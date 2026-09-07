"""Persistent, cross-run policy memory.

This is the part of the system that makes run N+1 different from run N.
Rules are learned from what the user actually did, not hardcoded.
Retrieval is deliberately explainable (weighted context overlap, no embeddings)
so the UI can show *why* a rule fired.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .store import LocalJSONStore, RuleStore

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"

# How much each matching context dimension contributes to a retrieval score.
CONTEXT_WEIGHTS = {"doc_type": 3.0, "platform": 2.0, "issuer": 1.0}

# Dimensions that GOVERN the preference rather than merely describe the
# instance. A person's posture genuinely differs between LinkedIn and
# Instagram, and between a scorecard and a vehicle photo -- so a rule may not
# act unattended across those. Which exam board issued the scorecard does not
# change whether they want their roll number hidden, so `issuer` informs
# ranking but never blocks a rule from applying.
GOVERNING_CONTEXT = ("doc_type", "platform")

# A rule must clear this to be applied without asking the user.
AUTO_APPLY_CONFIDENCE = 0.75

# ...and rest on at least this many agreeing observations. Confidence alone is
# not enough: one data point should never be sufficient to stop consulting the
# user, however unambiguous that single decision looked.
MIN_SUPPORT_FOR_AUTO = 2
# Recent observations outweigh old ones by this factor per step back.
RECENCY_DECAY = 0.72

# Below this a rule is treated as a weak hint only.
MIN_USEFUL_CONFIDENCE = 0.35


@dataclass
class Rule:
    """One learned belief about how this user wants a field handled."""

    field_name: str                     # e.g. "registration_number"
    decision: str                       # "redact" | "keep"
    context: dict[str, str]             # doc_type / platform / issuer
    rationale: str                      # the agent's own words, shown in UI
    support: list[str] = field(default_factory=list)        # run ids that agreed
    contradictions: list[str] = field(default_factory=list) # run ids that disagreed
    anchors: list[str] = field(default_factory=list)        # labels that identified it
    id: str = field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        """Recency-weighted agreement rate.

        Observations are ordered; recent ones count more (geometric decay).
        A flat lifetime ratio made rules far too sticky -- a user whose
        preference genuinely changed had to correct the agent ~7 times before
        it updated. With decay it adapts in 2-3, which is both better UX and
        an honest model of "this person's posture changed"."""
        events = [(seq, True) for seq in self.support]
        events += [(seq, False) for seq in self.contradictions]
        if not events:
            return 0.5
        events.sort(key=lambda e: self._order(e[0]))
        agree = total = 0.0
        for rank, (_, ok) in enumerate(reversed(events)):
            w = RECENCY_DECAY ** rank
            total += w
            if ok:
                agree += w
        return (agree + 0.5) / (total + 1.0)

    @staticmethod
    def _order(run_id: str) -> int:
        """Sort run ids like run_007 chronologically."""
        digits = "".join(ch for ch in run_id if ch.isdigit())
        return int(digits) if digits else 0

    @property
    def auto(self) -> bool:
        return (
            self.confidence >= AUTO_APPLY_CONFIDENCE
            and len(self.support) >= MIN_SUPPORT_FOR_AUTO
        )

    def learn_anchor(self, label: str) -> None:
        """Remember a label that identified this field.

        Decisions alone are not enough to skip the model: knowing you redact a
        registration number does not tell you which pixels are one. Storing the
        labels that located it before is what lets a later run resolve the
        field locally and make no model call at all.
        """
        norm = "".join(ch for ch in label if ch.isalnum()).casefold()
        if norm and norm not in self.anchors:
            self.anchors.append(norm)

    def applies_strictly(self, context: dict[str, str]) -> bool:
        """True only when no shared context dimension disagrees.

        Only GOVERNING_CONTEXT dimensions are checked. Confidence measures
        "how sure am I of this preference"; it says nothing about whether the
        preference transfers to a different setting. Acting unattended requires
        both. Without this a rule learned from LinkedIn offer letters will
        silently govern an Instagram post -- precisely the failure this system
        exists to prevent.
        """
        for key in GOVERNING_CONTEXT:
            mine, theirs = self.context.get(key), context.get(key)
            if mine and theirs and mine != theirs:
                return False
        return True

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        d["auto_apply"] = self.auto
        return d


class PolicyMemory:
    """File-backed store. One JSON per rule so the demo can show files appearing."""

    def __init__(
        self,
        directory: Path | None = None,
        store: RuleStore | None = None,
        namespace: str = "",
    ) -> None:
        """Back memory with `store`, or a local directory when none is given.

        The store is the only thing that knows where rules physically live, so
        moving to a hosted backend means passing a different object here.

        `namespace` scopes rules to one person. The whole premise is that
        privacy preferences are individual, so one user's learned policy must
        never govern another's post -- in a shared store that separation has to
        be explicit rather than assumed.
        """
        self.dir = Path(directory) if directory else MEMORY_DIR
        self.store: RuleStore = store or LocalJSONStore(self.dir)
        self.namespace = namespace

    # ---------- persistence ----------

    # Reserved context key carrying the owning user. Not a matching dimension:
    # it is absent from CONTEXT_WEIGHTS and GOVERNING_CONTEXT, so it partitions
    # the store without affecting how rules are ranked or applied.
    USER_KEY = "_user"

    def all_rules(self) -> list[Rule]:
        rules = []
        for raw in self.store.load_all():
            raw = dict(raw)
            owner = (raw.get("context") or {}).get(self.USER_KEY, "")
            if owner != self.namespace:
                continue
            # Derived fields are recomputed, never trusted from storage.
            raw.pop("confidence", None)
            raw.pop("auto_apply", None)
            try:
                rules.append(Rule(**raw))
            except TypeError:
                # A rule written by an older or newer schema is skipped rather
                # than crashing the run.
                continue
        return rules

    def save(self, rule: Rule) -> None:
        rule.updated_at = time.time()
        self.store.put(rule.id, rule.to_json())

    # ---------- retrieval ----------

    def _score(self, rule: Rule, context: dict[str, str]) -> float:
        """Weighted overlap. Mismatched values subtract, so a rule learned on
        Instagram does not silently govern a LinkedIn post."""
        score = 0.0
        for key, weight in CONTEXT_WEIGHTS.items():
            want, have = context.get(key), rule.context.get(key)
            if want is None or have is None:
                continue
            score += weight if want == have else -weight * 0.5
        return score

    def retrieve(self, context: dict[str, str], limit: int = 12) -> list[tuple[Rule, float]]:
        """Rules relevant to this run, best first, with their match scores."""
        scored = [
            (r, self._score(r, context))
            for r in self.all_rules()
            if r.confidence >= MIN_USEFUL_CONFIDENCE
        ]
        scored = [(r, s) for r, s in scored if s > 0]
        scored.sort(key=lambda rs: (rs[1], rs[0].confidence), reverse=True)
        return scored[:limit]

    def earned(self, context: dict[str, str]) -> dict[str, Rule]:
        """Rules permitted to act without consulting the user, by field.

        Requires confidence, supporting observations, AND strict context
        agreement. Anything failing the last test is still retrievable as a
        suggestion, but must be confirmed.
        """
        out: dict[str, Rule] = {}
        for rule, _score in self.retrieve(context):
            if rule.auto and rule.applies_strictly(context):
                out[rule.field_name] = rule
        return out

    def find_similar(self, field_name: str, context: dict[str, str]) -> Rule | None:
        """Existing rule this observation should reinforce rather than duplicate."""
        best, best_score = None, 0.0
        for rule in self.all_rules():
            if rule.field_name != field_name:
                continue
            score = self._score(rule, context)
            if score > best_score:
                best, best_score = rule, score
        return best

    # ---------- learning ----------

    def reinforce(self, rule: Rule, run_id: str, agreed: bool) -> Rule:
        """Fold one new observation into an existing rule."""
        bucket = rule.support if agreed else rule.contradictions
        if run_id not in bucket:
            bucket.append(run_id)
        self.save(rule)
        return rule

    def learn(
        self,
        field_name: str,
        decision: str,
        context: dict[str, str],
        rationale: str,
        run_id: str,
    ) -> tuple[Rule, str]:
        """Record an observation. Returns (rule, "created"|"reinforced"|"contradicted").

        A user decision that opposes an existing rule does not delete it --
        it lowers its confidence, and if the rule flips below the auto-apply
        threshold the agent starts asking about that field again.
        """
        context = dict(context)
        if self.namespace:
            context[self.USER_KEY] = self.namespace
        existing = self.find_similar(field_name, context)
        if existing is None:
            rule = Rule(
                field_name=field_name,
                decision=decision,
                context=context,
                rationale=rationale,
                support=[run_id],
            )
            self.save(rule)
            return rule, "created"

        if existing.decision == decision:
            existing.rationale = rationale or existing.rationale
            return self.reinforce(existing, run_id, agreed=True), "reinforced"

        self.reinforce(existing, run_id, agreed=False)
        # If the old belief has collapsed, replace it with the new one.
        if existing.confidence < MIN_USEFUL_CONFIDENCE:
            flipped = Rule(
                field_name=field_name,
                decision=decision,
                context=context,
                rationale=rationale,
                support=[run_id],
            )
            self.save(flipped)
            return flipped, "contradicted"
        return existing, "contradicted"

    # ---------- reporting (drives the demo panel) ----------

    def stats(self) -> dict[str, Any]:
        rules = self.all_rules()
        return {
            "total_rules": len(rules),
            "auto_apply_rules": sum(1 for r in rules if r.auto),
            "mean_confidence": round(
                sum(r.confidence for r in rules) / len(rules), 3
            ) if rules else 0.0,
            "observations": sum(len(r.support) + len(r.contradictions) for r in rules),
        }
