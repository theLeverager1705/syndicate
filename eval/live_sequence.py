"""Four real runs on real images, end to end, against the live model.

This is the measurement that matters: not a simulation of learning, but the
actual pipeline -- OCR, classification, redaction, verification -- with model
calls and latency falling as memory takes over.
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import classify, ocr, redact, verify
from agent.memory import PolicyMemory
from agent.reflect import Decision, reflect_on_run

# What this user actually wants. The agent must infer it from corrections.
POLICY = {
    "candidate_name": "keep",
    "registration_number": "redact",
    "application_number": "redact",
    "roll_number": "redact",
    "date_of_birth": "redact",
    "dob": "redact",
    "score": "keep",
    "percentile": "keep",
    "all_india_rank": "keep",
}

SEQUENCE = [
    ("demo/images/gate_2024.png", "GATE"),
    ("demo/images/jee_main.png", "JEE"),
    ("demo/images/ssc_cgl.png", "SSC"),
    ("demo/images/gate_2025.png", "GATE"),
]

tmp = Path(tempfile.mkdtemp())
memory = PolicyMemory(tmp)
out_dir = Path("runs/_proof")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"{'run':<5}{'doc':<12}{'asked':<7}{'from memory':<13}{'model calls':<13}"
      f"{'tokens out':<12}{'latency':<10}")
print("-" * 72)

try:
    for n, (img, issuer) in enumerate(SEQUENCE, 1):
        ctx = {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": issuer}
        t0 = time.time()

        tokens = ocr.extract(img)
        rules = memory.retrieve(ctx)
        earned = memory.earned(ctx)
        props, usage = classify.propose(tokens, ctx, rules, earned)

        decisions, to_redact, asked, from_mem = [], [], 0, 0
        for p in props:
            if p.needs_user:
                final = POLICY.get(p.field_name, "keep")   # the simulated user
                asked += 1
                was_asked = True
            else:
                final = p.decision
                from_mem += 1
                was_asked = False
            if final == "redact":
                to_redact.extend(p.token_indices)
            decisions.append(Decision(
                p.field_name, p.decision, final, p.source, p.rule_id,
                asked_user=was_asked,
                anchor=classify._label_for(tokens, p.token_indices[0]) if p.token_indices else None,
            ))

        boxes = [tokens[i].bbox for i in to_redact]
        secrets = [tokens[i].text for i in to_redact]
        out = str(out_dir / f"run{n}.png")
        strength = 1
        while strength <= 4:
            redact.apply(img, boxes, out, strength=strength)
            if verify.check(out, secrets).ok:
                break
            strength += 1

        reflect_on_run(memory, f"run_{n:03d}", ctx, decisions)
        elapsed = int((time.time() - t0) * 1000)

        print(f"{n:<5}{Path(img).stem:<12}{asked:<7}{from_mem:<13}"
              f"{usage.model_calls:<13}{usage.tokens_out:<12}{elapsed:>6}ms")

    print("-" * 72)
    rules = memory.all_rules()
    print(f"\nlearned {len(rules)} rules, {sum(1 for r in rules if r.auto)} now acting unattended:")
    for r in sorted(rules, key=lambda x: x.field_name):
        mark = "AUTO" if r.auto else "ask "
        print(f"  [{mark}] {r.decision:<7} {r.field_name:<22} conf={r.confidence:.2f} "
              f"anchors={r.anchors}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
