"""Orchestration: retrieve -> propose -> act -> verify -> escalate -> reflect.

Imports of worker-owned modules are deferred so this file is importable and
testable before W1-W4 have landed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from .memory import PolicyMemory
from .reflect import Decision, reflect_on_run

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MAX_VERIFY_ATTEMPTS = 4


@dataclass
class RunRecord:
    run_id: str
    context: dict[str, str]
    proposals: int = 0
    auto_applied: int = 0
    user_interventions: int = 0
    verify_attempts: int = 0
    verify_failures: int = 0
    final_strength: int = 1
    leaked: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model_calls: int = 0
    latency_ms: int = 0
    rules_before: int = 0
    rules_after: int = 0
    learned: str = ""

    def save(self) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def next_run_id(runs_dir: Path | None = None) -> str:
    d = runs_dir or RUNS_DIR
    d.mkdir(parents=True, exist_ok=True)
    existing = [p.stem for p in d.glob("run_*.json")]
    n = max((int(s.split("_")[1]) for s in existing if s.split("_")[1].isdigit()), default=0)
    return f"run_{n + 1:03d}"


def process(
    image_path: str,
    context: dict[str, str],
    memory: PolicyMemory,
    ask_user: Callable[[Any, list], str] | None = None,
    out_dir: str | None = None,
) -> tuple[RunRecord, str]:
    """Run the full pipeline over one image.

    `ask_user(proposal, tokens) -> "redact" | "keep"` is called only for fields
    no earned rule covers. Pass None to run fully autonomously (the agent then
    follows its own proposal, which is what later demo runs do).
    """
    from . import classify, ocr, redact, verify

    started = time.time()
    run_id = next_run_id()
    record = RunRecord(run_id=run_id, context=context)
    record.rules_before = len(memory.all_rules())

    tokens = ocr.extract(image_path)
    rules = memory.retrieve(context)
    proposals, usage = classify.propose(tokens, context, rules)

    record.proposals = len(proposals)
    record.tokens_in = usage.tokens_in
    record.tokens_out = usage.tokens_out
    record.model_calls = usage.model_calls

    decisions: list[Decision] = []
    to_redact: list[int] = []

    for p in proposals:
        if p.needs_user and ask_user is not None:
            final = ask_user(p, tokens)
            record.user_interventions += 1
            asked = True
        else:
            final = p.decision
            asked = False
            if p.source == "memory":
                record.auto_applied += 1

        decisions.append(Decision(
            field_name=p.field_name, proposed=p.decision, final=final,
            source=p.source, rule_id=p.rule_id, asked_user=asked,
        ))
        if final == "redact":
            to_redact.extend(p.token_indices)

    out_path = str(Path(out_dir or RUNS_DIR) / f"{run_id}_safe.png")
    boxes = [tokens[i].bbox for i in to_redact]
    must_be_absent = [tokens[i].text for i in to_redact]

    # Act, then check our own work, escalating until the text is truly gone.
    strength = 1
    result = None
    while strength <= MAX_VERIFY_ATTEMPTS:
        redact.apply(image_path, boxes, out_path, strength=strength)
        record.verify_attempts += 1
        result = verify.check(out_path, must_be_absent)
        if result.ok:
            break
        record.verify_failures += 1
        strength += 1

    record.final_strength = strength
    record.leaked = list(result.still_readable) if result and not result.ok else []

    report = reflect_on_run(memory, run_id, context, decisions)
    record.learned = report.summary()
    record.rules_after = len(memory.all_rules())
    record.latency_ms = int((time.time() - started) * 1000)
    record.save()

    return record, out_path
