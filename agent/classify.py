"""Deciding what each piece of text is, and whether it should survive.

Two resolution paths:

  LOCAL   -- a learned anchor identifies the field and an earned rule decides
             it. Zero model calls, zero latency.
  MODEL   -- anything unresolved goes to the model, batched into ONE call.

The whole point of the memory is to move work from the second path to the
first. On a mature memory this function makes no network request at all.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import heuristics
from .config import MODEL, client
from .memory import Rule

# GLM-4.7-Flash spends a large, variable budget on internal reasoning before
# emitting any content. Measured: ~1800 output tokens for a 6-field document.
# Too low a cap does not truncate the answer -- it produces NO answer at all,
# with finish_reason "length" and content None.
MAX_OUTPUT_TOKENS = 6000

# Field name used when classification failed. These must never be learned from.
UNRESOLVED_PREFIX = "unresolved_"

# The model names the same field differently across runs -- percentile/score,
# registration_no/registration_number. Left alone this fragments memory into
# near-duplicate rules that each need their own two observations before they
# can act. Canonicalising costs nothing and roughly halves the time to
# autonomy.
CANONICAL = {
    "percentile": "score",
    "marks": "score",
    "total_score": "score",
    "rank": "all_india_rank",
    "air": "all_india_rank",
    "registration_no": "registration_number",
    "reg_number": "registration_number",
    "application_no": "application_number",
    "roll_no": "roll_number",
    "dob": "date_of_birth",
    "name": "candidate_name",
    "student_name": "candidate_name",
    "plate_number": "license_plate",
    "vehicle_number": "license_plate",
    "number_plate": "license_plate",
}


def canonical(field_name: str) -> str:
    key = field_name.strip().lower().replace(" ", "_")
    return CANONICAL.get(key, key)

# Words that mark a token as document furniture rather than a personal value.
# Matched as substrings, so "GATE 2025 - SCORECARD" is recognised as a heading
# without having to enumerate every issuer. Headings carry no label to their
# left, so without this every run would pay a model call to re-classify the
# title of a document it has already seen a dozen times.
HEADING_WORDS = ("scorecard", "score card", "certificate", "offer letter",
                 "result", "admit card", "marksheet", "statement")

# Text that is structural, not personal: headings, column labels, boilerplate.
# Never proposed for redaction and never sent for classification.
STRUCTURAL = {
    "scorecard", "score card", "candidate name", "registration no",
    "registration number", "roll number", "application no", "date of birth",
    "all india rank", "score", "percentile", "employee id", "job title",
    "salary", "start date",
}


@dataclass
class Proposal:
    field_name: str
    decision: str                     # "redact" | "keep"
    token_indices: list[int]
    rationale: str
    source: str                       # "memory" | "model"
    rule_id: str | None = None
    needs_user: bool = True


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model_calls: int = 0

    @property
    def free(self) -> bool:
        return self.model_calls == 0


def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum()).casefold()


def is_structural(text: str, structural: set[str] | None = None) -> bool:
    """Document furniture: a heading, a column label, boilerplate."""
    norm = _norm(text)
    if not norm:
        return True
    if norm in (structural if structural is not None else {_norm(s) for s in STRUCTURAL}):
        return True
    return any(_norm(word) in norm for word in HEADING_WORDS)


def _label_for(tokens, index: int) -> str | None:
    """The label token sitting to the left of `tokens[index]` on the same line.

    Documents put the label left of the value; this recovers that pairing
    without needing layout analysis.
    """
    target = tokens[index]
    ty0, ty1 = target.bbox[1], target.bbox[3]
    best, best_x = None, -1

    for j, other in enumerate(tokens):
        if j == index:
            continue
        oy0, oy1 = other.bbox[1], other.bbox[3]
        overlap = min(ty1, oy1) - max(ty0, oy0)
        height = min(ty1 - ty0, oy1 - oy0) or 1
        if overlap / height < 0.5:          # not on the same line
            continue
        if other.bbox[2] > target.bbox[0]:  # not to the left
            continue
        if other.bbox[2] > best_x:
            best, best_x = other.text, other.bbox[2]

    return best


def resolve_locally(
    tokens,
    rules: list[tuple[Rule, float]],
    earned: dict[str, Rule],
) -> tuple[list[Proposal], set[int]]:
    """Fields identifiable from learned anchors alone."""
    by_anchor: dict[str, Rule] = {}
    for rule, _score in rules:
        for anchor in rule.anchors:
            by_anchor.setdefault(anchor, rule)

    proposals: list[Proposal] = []
    claimed: set[int] = set()

    structural = {_norm(s) for s in STRUCTURAL}
    for i, tok in enumerate(tokens):
        if is_structural(tok.text, structural):
            claimed.add(i)
            continue

        label = _label_for(tokens, i)
        if not label:
            continue
        rule = by_anchor.get(_norm(label))
        if rule is None or rule.field_name not in earned:
            continue

        proposals.append(Proposal(
            field_name=rule.field_name,
            decision=rule.decision,
            token_indices=[i],
            rationale=f"Learned from {len(rule.support)} prior run(s): {rule.rationale}",
            source="memory",
            rule_id=rule.id,
            needs_user=False,
        ))
        claimed.add(i)

    return proposals, claimed


PROMPT = """You are identifying fields in text extracted from an image that \
someone is about to post publicly.

Context: {context}

Numbered text tokens found in the image. Where the layout put a label to the left of a value, that label is shown -- trust it over your own guess about what the value means:
{tokens}

For each token that holds a PERSON-SPECIFIC value (an identifier, a number, a \
date, a name, a plate), name the field it belongs to and say whether a \
privacy-conscious person would redact it before posting publicly.

Ignore tokens that are labels, headings, or boilerplate.

Use snake_case field names like: registration_number, roll_number, \
application_number, candidate_name, date_of_birth, all_india_rank, score, \
license_plate, employee_id, salary.

Return strict JSON only, no prose:
{{"fields": [{{"field_name": "...", "token_index": 0, "decision": "redact", \
"rationale": "one short clause"}}]}}"""


def _parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def propose(
    tokens,
    context: dict[str, str],
    rules: list[tuple[Rule, float]],
    earned: dict[str, Rule] | None = None,
) -> tuple[list[Proposal], Usage]:
    """Propose a decision for every meaningful token in the image."""
    earned = earned if earned is not None else {}
    usage = Usage()

    proposals, claimed = resolve_locally(tokens, rules, earned)

    remaining = [i for i in range(len(tokens))
                 if i not in claimed and not is_structural(tokens[i].text)]
    if not remaining:
        return proposals, usage            # fully handled from memory

    # Name what local rules can name. These never decide alone -- a pattern
    # match says what a value is, not what this person wants done with it --
    # but a correctly named field the user confirms once becomes a rule, and
    # the model is never needed for it again.
    named: list[Proposal] = []
    still_unknown: list[int] = []
    for i in remaining:
        guess = heuristics.identify(tokens[i].text, _label_for(tokens, i))
        if guess is None:
            still_unknown.append(i)
            continue
        field_name, decision = guess
        named.append(Proposal(
            field_name=field_name, decision=decision, token_indices=[i],
            rationale="Recognised from the document's own labelling.",
            source="model", needs_user=True,
        ))

    c = client()
    if c is None or not still_unknown:
        proposals.extend(named)
        for i in still_unknown:
            proposals.append(Proposal(
                field_name=f"{UNRESOLVED_PREFIX}{i}", decision="redact",
                token_indices=[i],
                rationale="Not recognised; deferring to you.",
                source="model", needs_user=True,
            ))
        return proposals, usage

    remaining = still_unknown
    proposals.extend(named)

    lines = []
    for i in remaining:
        label = _label_for(tokens, i)
        if label:
            lines.append(f'{i}: "{tokens[i].text}"   [label to its left: "{label}"]')
        else:
            lines.append(f'{i}: "{tokens[i].text}"')
    listing = "\n".join(lines)
    messages = [{"role": "user", "content": PROMPT.format(
        context=json.dumps(context), tokens=listing)}]

    started = time.time()
    payload, last_error = None, None

    for attempt in range(2):
        try:
            resp = c.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
            )
            usage.model_calls += 1
            if resp.usage:
                usage.tokens_in += resp.usage.prompt_tokens
                usage.tokens_out += resp.usage.completion_tokens
            choice = resp.choices[0]
            if choice.finish_reason == "length" or not choice.message.content:
                raise ValueError(
                    f"model produced no content (finish_reason={choice.finish_reason}); "
                    "reasoning budget exhausted"
                )
            payload = _parse(choice.message.content)
            break
        except Exception as exc:
            last_error = exc
            messages.append({"role": "user", "content":
                             f"That was not valid JSON ({exc}). Return only the JSON object."})

    usage.latency_ms = int((time.time() - started) * 1000)

    if payload is None:
        for i in remaining:
            proposals.append(Proposal(
                field_name=f"{UNRESOLVED_PREFIX}{i}", decision="redact", token_indices=[i],
                rationale=f"Model output unusable ({type(last_error).__name__}); deferring to you.",
                source="model", needs_user=True,
            ))
        return proposals, usage

    for entry in payload.get("fields", []):
        try:
            idx = int(entry["token_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx not in remaining:
            continue
        proposals.append(Proposal(
            field_name=canonical(str(entry.get("field_name", f"{UNRESOLVED_PREFIX}{idx}"))),
            decision="redact" if entry.get("decision") == "redact" else "keep",
            token_indices=[idx],
            rationale=str(entry.get("rationale", ""))[:160],
            source="model",
            needs_user=True,
        ))

    return proposals, usage
