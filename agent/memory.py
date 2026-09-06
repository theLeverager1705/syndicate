"""Persistent, cross-run policy memory.

This is the part of the system that makes run N+1 different from run N.
Rules are learned from what the user actually did, not hardcoded.
Retrieval is deliberately explainable (weighted context overlap, no embeddings)
so the UI can show *why* a rule fired.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"

# How much each matching context dimension contributes to a retrieval score.
CONTEXT_WEIGHTS = {"doc_type": 3.0, "platform": 2.0, "issuer": 1.0}

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

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        d["auto_apply"] = self.auto
        return d


class PolicyMemory:
    """File-backed store. One JSON per rule so the demo can show files appearing."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory) if directory else MEMORY_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------- persistence ----------

    def all_rules(self) -> list[Rule]:
        rules = []
        for p in sorted(self.dir.glob("rule_*.json")):
            raw = json.loads(p.read_text(encoding="utf-8"))
            raw.pop("confidence", None)
            raw.pop("auto_apply", None)
            rules.append(Rule(**raw))
        return rules

    def save(self, rule: Rule) -> None:
        rule.updated_at = time.time()
        (self.dir / f"{rule.id}.json").write_text(
            json.dumps(rule.to_json(), indent=2), encoding="utf-8"
        )

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
