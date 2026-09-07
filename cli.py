"""Verity -- the demo surface.

    python cli.py reset                     clear learned memory
    python cli.py run <image> [--auto]      one run, interactive unless --auto
    python cli.py memory                    show what has been learned
    python cli.py sequence                  the scripted demo sequence

Interactive runs ask only about fields no earned rule covers. As memory grows
the questions stop, and so do the model calls.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent import classify, ocr, redact, verify
from agent.memory import MEMORY_DIR, PolicyMemory
from agent.reflect import Decision, reflect_on_run

console = Console()
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runs" / "output"

SOURCE_STYLE = {"memory": "bold green", "model": "bold yellow"}

# --policy answers as a consistent person would, instead of prompting. It is a
# rehearsal aid: the demo is deterministic and repeatable, and typing mistakes
# during a recording do not silently teach the agent the wrong thing.
REHEARSAL_POLICY = {
    "candidate_name": "keep",
    "score": "keep",
    "all_india_rank": "keep",
    "job_title": "keep",
    "start_date": "keep",
    "registration_number": "redact",
    "application_number": "redact",
    "roll_number": "redact",
    "date_of_birth": "redact",
    "employee_id": "redact",
    "salary": "redact",
    "license_plate": "redact",
    "qr_code": "redact",
}


def infer_context(image_path: str, platform: str | None = None) -> dict[str, str]:
    """Context from the filename, which in the demo encodes the document type."""
    stem = Path(image_path).stem.lower()
    if any(k in stem for k in ("car", "bike", "vehicle", "plate")):
        return {"doc_type": "vehicle_photo",
                "platform": platform or "instagram", "issuer": ""}
    issuer = stem.split("_")[0].upper()
    return {"doc_type": "exam_scorecard",
            "platform": platform or "linkedin", "issuer": issuer}


def ask(field_name: str, value: str, proposed: str, rationale: str) -> str:
    """Consult the user. Returns the decision they settled on."""
    suggestion = "redact" if proposed == "redact" else "keep"
    console.print(
        f"    [bold]{field_name}[/bold] = [cyan]{value}[/cyan]"
        + (f"\n    [dim]{rationale}[/dim]" if rationale else "")
    )
    prompt = (f"    [{'R' if suggestion == 'redact' else 'r'}]edact / "
              f"[{'K' if suggestion == 'keep' else 'k'}]eep "
              f"(enter = {suggestion}): ")
    while True:
        try:
            answer = console.input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return suggestion
        if not answer:
            return suggestion
        if answer.startswith("r"):
            return "redact"
        if answer.startswith("k"):
            return "keep"


def run_one(image_path: str, memory: PolicyMemory, auto: bool = False,
            platform: str | None = None, label: str | None = None,
            policy: dict[str, str] | None = None) -> dict:
    context = infer_context(image_path, platform)
    started = time.time()

    console.rule(f"[bold]{label or Path(image_path).stem}[/bold]  "
                 f"[dim]{context['doc_type']} | {context['platform']}[/dim]")

    tokens = ocr.extract(image_path)
    rules = memory.retrieve(context)
    earned = memory.earned(context)

    with console.status("[dim]classifying...[/dim]"):
        proposals, usage = classify.propose(tokens, context, rules, earned)

    table = Table(box=None, pad_edge=False, show_edge=False)
    table.add_column("field", style="bold", width=22)
    table.add_column("value", style="cyan", width=22)
    table.add_column("source", width=8)
    table.add_column("decision", width=9)

    decisions: list[Decision] = []
    to_redact: list[int] = []
    asked_count = from_memory = 0
    pending: list = []

    for p in proposals:
        idx = p.token_indices[0] if p.token_indices else None
        value = tokens[idx].text if idx is not None else ""
        if p.needs_user and not auto:
            pending.append((p, idx, value))
            table.add_row(p.field_name, value,
                          Text("ASK", style="bold magenta"), Text("-", style="dim"))
        else:
            if p.source == "memory":
                from_memory += 1
            table.add_row(
                p.field_name, value,
                Text(p.source.upper(), style=SOURCE_STYLE.get(p.source, "white")),
                Text(p.decision, style="red" if p.decision == "redact" else "green"),
            )

    console.print(table)

    if pending:
        console.print(f"\n  [magenta]{len(pending)} field(s) I have not earned "
                      f"the right to decide alone:[/magenta]\n")

    for p, idx, value in pending:
        if policy is not None:
            final = policy.get(p.field_name, "redact")
            console.print(f"    [bold]{p.field_name}[/bold] = [cyan]{value}[/cyan]"
                          f"  ->  [{'red' if final == 'redact' else 'green'}]{final}"
                          f"[/{'red' if final == 'redact' else 'green'}] [dim](rehearsal)[/dim]")
        else:
            final = ask(p.field_name, value, p.decision, p.rationale)
        asked_count += 1
        if final == "redact":
            to_redact.extend(p.token_indices)
        decisions.append(Decision(
            p.field_name, p.decision, final, p.source, p.rule_id,
            asked_user=True,
            anchor=classify._label_for(tokens, idx) if idx is not None else None,
        ))
        console.print()

    for p in proposals:
        if any(d.field_name == p.field_name and d.asked_user for d in decisions):
            continue
        idx = p.token_indices[0] if p.token_indices else None
        if p.decision == "redact":
            to_redact.extend(p.token_indices)
        decisions.append(Decision(
            p.field_name, p.decision, p.decision, p.source, p.rule_id,
            asked_user=False,
            anchor=classify._label_for(tokens, idx) if idx is not None else None,
        ))

    # Act, verify, escalate.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUT_DIR / f"{Path(image_path).stem}_safe.png")
    boxes = [tokens[i].bbox for i in to_redact]
    secrets = [tokens[i].text for i in to_redact]

    strength, result = 1, None
    while strength <= 4:
        redact.apply(image_path, boxes, out_path, strength=strength)
        result = verify.check(out_path, secrets)
        if result.ok:
            if strength > 1:
                console.print(f"  [green]verified[/green] at strength {strength}")
            break
        console.print(
            f"  [yellow]verify failed[/yellow] at strength {strength}: "
            f"{', '.join(result.still_readable)} still readable -- escalating"
        )
        strength += 1

    rules_before = len(memory.all_rules())
    report = reflect_on_run(memory, f"run_{int(time.time()) % 100000:05d}",
                            context, decisions)
    rules_after = len(memory.all_rules())
    elapsed = time.time() - started

    summary = Table.grid(padding=(0, 3))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("asked", f"[bold]{asked_count}[/bold]")
    summary.add_row("from memory", f"[bold green]{from_memory}[/bold green]")
    summary.add_row("model calls", f"[bold]{usage.model_calls}[/bold]")
    summary.add_row("output tokens", f"[bold]{usage.tokens_out}[/bold]")
    summary.add_row("time", f"[bold]{elapsed:.1f}s[/bold]")
    summary.add_row("learned", report.summary())
    summary.add_row("rules", f"{rules_before} -> [bold]{rules_after}[/bold]")
    console.print(Panel(summary, title="[bold]run[/bold]", border_style="dim",
                        expand=False))
    console.print(f"  [dim]safe copy: {out_path}[/dim]\n")

    return {"asked": asked_count, "from_memory": from_memory,
            "model_calls": usage.model_calls, "tokens_out": usage.tokens_out,
            "seconds": elapsed, "out": out_path}


def show_memory(memory: PolicyMemory) -> None:
    rules = sorted(memory.all_rules(), key=lambda r: (-r.confidence, r.field_name))
    if not rules:
        console.print("[dim]no rules learned yet[/dim]")
        return

    table = Table(title="learned policy", box=None)
    table.add_column("", width=6)
    table.add_column("action", width=8)
    table.add_column("field", style="bold", width=22)
    table.add_column("context", style="dim", width=26)
    table.add_column("conf", width=6)
    table.add_column("seen", width=5)

    for r in rules:
        table.add_row(
            Text("AUTO", style="bold green") if r.auto else Text("ask", style="dim"),
            Text(r.decision, style="red" if r.decision == "redact" else "green"),
            r.field_name,
            f"{r.context.get('doc_type','')}/{r.context.get('platform','')}",
            f"{r.confidence:.2f}",
            str(len(r.support)),
        )
    console.print(table)
    s = memory.stats()
    console.print(f"\n  [dim]{s['total_rules']} rules | "
                  f"{s['auto_apply_rules']} acting unattended | "
                  f"{s['observations']} observations[/dim]\n")


SEQUENCE = ["gate_2024", "jee_main", "ssc_cgl", "gate_2025", "cat_score"]


def main() -> None:
    args = sys.argv[1:]
    if not args:
        console.print(__doc__)
        return

    command = args[0]

    # --neural backs memory with Evorozen Neural DB instead of local files.
    # Only rules travel: field names, decisions, confidences and layout
    # anchors. Documents and extracted identifiers never leave the machine.
    if "--neural" in args:
        from agent.neural_store import NeuralPulseStore
        store = NeuralPulseStore()
        store.ensure_schema()
        memory = PolicyMemory(store=store)
        console.print("[dim]memory backend: Evorozen Neural DB[/dim]")
    else:
        memory = PolicyMemory()

    if command == "reset":
        if MEMORY_DIR.exists():
            shutil.rmtree(MEMORY_DIR)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        console.print("[green]memory cleared[/green]")

    elif command == "memory":
        show_memory(memory)

    elif command == "run":
        if len(args) < 2:
            console.print("[red]usage: python cli.py run <image> [--auto][/red]")
            return
        run_one(args[1], memory, auto="--auto" in args,
                policy=REHEARSAL_POLICY if "--policy" in args else None)
        show_memory(memory)

    elif command == "sequence":
        # --teach N: the first N runs consult you, the rest run unattended.
        # That is the whole demonstration -- you teach it three times, then
        # stop being needed.
        teach = 0
        for i, a in enumerate(args):
            if a == "--teach" and i + 1 < len(args):
                teach = int(args[i + 1])
        if "--auto" in args:
            teach = 0
        if teach == 0 and "--auto" not in args:
            teach = len(SEQUENCE)

        stats = []
        for n, name in enumerate(SEQUENCE, 1):
            path = f"demo/images/{name}.png"
            stats.append(run_one(path, memory, auto=(n > teach),
                                 label=f"run {n}  |  {name}",
                                 policy=REHEARSAL_POLICY if "--policy" in args else None))
        show_memory(memory)

        table = Table(title="across the sequence", box=None)
        table.add_column("run"); table.add_column("asked")
        table.add_column("from memory"); table.add_column("model calls")
        table.add_column("tokens"); table.add_column("time")
        for i, s in enumerate(stats, 1):
            table.add_row(str(i), str(s["asked"]), str(s["from_memory"]),
                          str(s["model_calls"]), str(s["tokens_out"]),
                          f"{s['seconds']:.1f}s")
        console.print(table)

    else:
        console.print(f"[red]unknown command: {command}[/red]")
        console.print(__doc__)


if __name__ == "__main__":
    main()
