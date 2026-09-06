"""Turning what the user did into what the agent believes.

The folding of observations into memory is deterministic and testable without
any model. The model is used only for the optional generalisation pass, which
looks across accumulated rules and proposes higher-order principles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import PolicyMemory, Rule


@dataclass
class Decision:
    """What the agent proposed for one field, and what the user settled on."""

    field_name: str
    proposed: str            # "redact" | "keep"
    final: str               # "redact" | "keep"
    source: str              # "memory" | "model"
    rule_id: str | None = None
    asked_user: bool = False
    anchor: str | None = None        # label text that located this field

    @property
    def was_wrong(self) -> bool:
        return self.proposed != self.final

    @property
    def was_free(self) -> bool:
        """Handled from memory without bothering the user or the model."""
        return self.source == "memory" and not self.asked_user


@dataclass
class ReflectionReport:
    run_id: str
    created: list[str] = field(default_factory=list)
    reinforced: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)
    skipped_unresolved: int = 0

    @property
    def learned_anything(self) -> bool:
        return bool(self.created or self.reinforced or self.contradicted)

    def summary(self) -> str:
        bits = []
        if self.created:
            bits.append(f"{len(self.created)} new rule(s)")
        if self.reinforced:
            bits.append(f"{len(self.reinforced)} reinforced")
        if self.contradicted:
            bits.append(f"{len(self.contradicted)} corrected")
        if self.skipped_unresolved:
            bits.append(f"{self.skipped_unresolved} unresolved (not learned)")
        return ", ".join(bits) or "nothing new"


def _rationale(decision: Decision, context: dict[str, str]) -> str:
    """A short, honest explanation written from the observation itself.

    Deliberately not model-generated: this text is shown in the UI as the
    agent's justification, and it must never claim more than we observed.
    """
    verb = "redacts" if decision.final == "redact" else "keeps"
    where = context.get("doc_type", "this kind of content")
    if decision.was_wrong:
        return (
            f"User overrode the agent and {verb} {decision.field_name} "
            f"on a {where}."
        )
    return f"User confirmed that they {verb} {decision.field_name} on a {where}."


def reflect_on_run(
    memory: PolicyMemory,
    run_id: str,
    context: dict[str, str],
    decisions: list[Decision],
) -> ReflectionReport:
    """Fold one run's outcomes into persistent memory.

    Every decision teaches something -- including the ones the agent got right,
    which is what lets confidence climb and interventions fall.
    """
    report = ReflectionReport(run_id=run_id)

    for decision in decisions:
        # A field the classifier could not name teaches nothing. Learning from
        # it would write a confident rule keyed to a meaningless name, and the
        # agent would then apply that rule unattended on later runs.
        if decision.field_name.startswith("unresolved_"):
            report.skipped_unresolved += 1
            continue

        rule, action = memory.learn(
            field_name=decision.field_name,
            decision=decision.final,
            context=context,
            rationale=_rationale(decision, context),
            run_id=run_id,
        )
        if decision.anchor:
            rule.learn_anchor(decision.anchor)
            memory.save(rule)

        {"created": report.created,
         "reinforced": report.reinforced,
         "contradicted": report.contradicted}[action].append(rule.id)

        if decision.was_wrong:
            report.surprises.append(
                f"proposed {decision.proposed} for {decision.field_name}, "
                f"user chose {decision.final}"
            )

    return report


GENERALISE_PROMPT = """You are reviewing an agent's learned rules about one \
user's content-sharing preferences.

Rules observed so far:
{rules}

Identify at most 2 higher-order principles that explain several rules at once.
A good principle is predictive: it should let the agent handle a field it has
never seen before.

Return strict JSON only:
{{"principles": [{{"statement": "...", "explains": ["rule_id", ...], \
"predicts": "how this applies to an unseen field"}}]}}

Return {{"principles": []}} if the rules are too few or too unrelated. Do not \
invent patterns that are not there."""


def generalise(memory: PolicyMemory, client: Any = None, model: str = "glm-4-7-flash") -> list[dict]:
    """Optional self-reflection pass across accumulated rules.

    Returns [] when there is no client or too little evidence, so the pipeline
    never depends on this succeeding.
    """
    rules = [r for r in memory.all_rules() if r.confidence >= 0.5]
    if client is None or len(rules) < 4:
        return []

    rendered = "\n".join(
        f"- {r.id}: {r.decision} {r.field_name} "
        f"(context={r.context}, confidence={r.confidence:.2f})"
        for r in rules
    )

    import json

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": GENERALISE_PROMPT.format(rules=rendered)}],
            temperature=0,
        )
        payload = json.loads(resp.choices[0].message.content.strip()
                             .removeprefix("```json").removeprefix("```")
                             .removesuffix("```").strip())
        return payload.get("principles", [])
    except Exception:
        # Reflection is a bonus, never a dependency.
        return []
