"""Does the agent actually get better? Measured, with an ablation.

Two arms over the same document sequence:
  MEMORY ON  -- rules persist and earn the right to act unattended
  MEMORY OFF -- the ablation; a competent non-learning PII detector

The headline metric is user interventions per run. The honesty check is
silent errors: decisions the agent made unattended that contradict what the
user actually wanted. A system that stops asking by getting reckless would
show interventions falling AND silent errors rising.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from agent.memory import PolicyMemory
from agent.reflect import Decision, reflect_on_run
from eval.fixtures import DOCS, Doc, naive_proposal, user_decision

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class RunMetrics:
    run: int
    doc_id: str
    fields: int
    interventions: int      # times the user had to be consulted
    auto_applied: int       # handled from earned memory, no user, no model
    model_calls: int        # fresh judgements needed
    silent_errors: int      # unattended decisions that were WRONG
    rules_after: int
    learned: str


def run_arm(use_memory: bool, docs: list[Doc]) -> list[RunMetrics]:
    tmp = Path(tempfile.mkdtemp())
    memory = PolicyMemory(tmp)
    out: list[RunMetrics] = []

    try:
        for i, doc in enumerate(docs, 1):
            earned = memory.earned(doc.context) if use_memory else {}

            decisions: list[Decision] = []
            interventions = auto = calls = silent_errors = 0

            for fname in doc.fields:
                truth = user_decision(fname, doc.context)

                if fname in earned:
                    # Memory is confident enough to act alone.
                    rule = earned[fname]
                    proposed = rule.decision
                    final = proposed
                    auto += 1
                    if final != truth:
                        silent_errors += 1
                    decisions.append(Decision(fname, proposed, final, "memory",
                                              rule.id, asked_user=False))
                else:
                    # No earned rule: think about it, then ask.
                    proposed = naive_proposal(fname, doc.context)
                    calls += 1
                    final = truth
                    interventions += 1
                    decisions.append(Decision(fname, proposed, final, "model",
                                              None, asked_user=True))

            if use_memory:
                report = reflect_on_run(memory, f"run_{i:03d}", doc.context, decisions)
                learned = report.summary()
            else:
                learned = "memory disabled"

            out.append(RunMetrics(
                run=i, doc_id=doc.doc_id, fields=len(doc.fields),
                interventions=interventions, auto_applied=auto,
                model_calls=calls, silent_errors=silent_errors,
                rules_after=len(memory.all_rules()) if use_memory else 0,
                learned=learned,
            ))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return out


def bar(value: int, scale: int = 3, char: str = "#") -> str:
    return char * (value * scale)


def chart(on: list[RunMetrics], off: list[RunMetrics]) -> str:
    lines = ["", "USER INTERVENTIONS PER RUN", ""]
    lines.append(f"  {'run':<4} {'document':<14} {'memory ON':<26} {'memory OFF (ablation)'}")
    for a, b in zip(on, off):
        lines.append(
            f"  {a.run:<4} {a.doc_id:<14} "
            f"{bar(a.interventions) + ' ' + str(a.interventions):<26} "
            f"{bar(b.interventions) + ' ' + str(b.interventions)}"
        )
    t_on = sum(m.interventions for m in on)
    t_off = sum(m.interventions for m in off)
    lines += [
        "",
        f"  total interventions   ON {t_on:>3}   OFF {t_off:>3}   "
        f"({(1 - t_on / t_off):.0%} fewer)",
        f"  total model calls     ON {sum(m.model_calls for m in on):>3}   "
        f"OFF {sum(m.model_calls for m in off):>3}",
        f"  silent errors         ON {sum(m.silent_errors for m in on):>3}   "
        f"(unattended decisions that contradicted the user)",
        f"  rules learned         {on[-1].rules_after}",
        "",
        "  Last 3 runs:",
    ]
    for m in on[-3:]:
        lines.append(f"    run {m.run}: {m.interventions} interventions, "
                     f"{m.auto_applied}/{m.fields} handled from memory, "
                     f"{m.model_calls} model calls")
    return "\n".join(lines)


def main() -> None:
    on = run_arm(True, DOCS)
    off = run_arm(False, DOCS)
    print(chart(on, off))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "curve.json").write_text(json.dumps(
        {"memory_on": [asdict(m) for m in on],
         "memory_off": [asdict(m) for m in off]},
        indent=2), encoding="utf-8")
    print(f"\n  wrote {RUNS_DIR / 'curve.json'}\n")


if __name__ == "__main__":
    main()
