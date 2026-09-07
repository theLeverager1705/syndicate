# Verity — an agent that learns what you are willing to share

Evorozen Apex — Category 6: Agentic OS & Workflow Automation.

## One-line pitch

Verity screens content before you post it publicly, and **learns each user's
individual sharing posture from their corrections**, so it needs fewer and
fewer of them over time.

## Why this is a learning-agent project, not a PII detector

A PII detector is a fixed function: same input, same output, forever.
Verity's behaviour on run N depends on runs 1..N-1. It:

1. **Retrieves** prior rules relevant to this document's context.
2. **Proposes** redactions, auto-applying only rules it has earned confidence in.
3. **Observes** what the user accepts, rejects, or edits.
4. **Reflects** on disagreements and writes generalised rules.
5. **Verifies** its own output deterministically, and retries when it failed.

Run 1 asks about everything. Run 8 asks about nothing and is still correct.
That delta is the product.

## Non-negotiable design commitments

- **Deterministic verification.** After redacting, we re-run OCR on the output
  image and assert the target string is absent. We never ask a model whether
  its own work succeeded.
- **Explainable retrieval.** Weighted context overlap, no embeddings. The UI
  can always answer "why did this rule fire?".
- **Memory is per-user and contextual.** A rule learned for Instagram must not
  silently govern a LinkedIn post. Context = (doc_type, platform, issuer).
- **The agent may be wrong.** Contradictions lower confidence rather than
  deleting rules; a shaken rule returns to asking the user.

## Architecture

```
image ──► ocr.extract() ──► [Token(text, bbox, conf)]
                                  │
              memory.retrieve(context) ──► [Rule]
                                  │
                          classify.propose()          ← one LLM call
                                  │
                        [Proposal(field, decision, bbox, source)]
                                  │
                    ┌─────────────┴─────────────┐
             auto-apply (earned)          ask user (unearned)
                    └─────────────┬─────────────┘
                                  ▼
                        redact.apply() ──► output image
                                  │
                        verify.check() ──► ok? ──no──► escalate blur, retry
                                  │ yes
                                  ▼
                     reflect.learn() ──► memory/*.json grows
```

## Interface contracts (FROZEN — do not change without updating this file)

Workers build against these. Anyone who needs a change edits SPEC.md first.

```python
# agent/ocr.py
@dataclass
class Token:
    text: str
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 in pixels
    confidence: float                  # 0..1

def extract(image_path: str) -> list[Token]: ...


# agent/classify.py
@dataclass
class Proposal:
    field_name: str        # snake_case, e.g. "registration_number"
    decision: str          # "redact" | "keep"
    token_indices: list[int]  # indices into the Token list
    rationale: str
    source: str            # "memory" (rule fired) | "model" (fresh judgement)
    rule_id: str | None    # set when source == "memory"
    needs_user: bool       # True when no earned rule covers this

def propose(
    tokens: list[Token],
    context: dict[str, str],
    rules: list[tuple[Rule, float]],
) -> tuple[list[Proposal], Usage]: ...


# agent/redact.py
def apply(
    image_path: str,
    boxes: list[tuple[int, int, int, int]],
    out_path: str,
    strength: int = 1,      # escalation level; higher = larger + blurrier
) -> str: ...


# agent/verify.py
@dataclass
class VerifyResult:
    ok: bool
    still_readable: list[str]   # target strings OCR could still recover
    attempts: int

def check(out_path: str, must_be_absent: list[str]) -> VerifyResult: ...


# agent/memory.py  (BUILT — do not rewrite)
PolicyMemory.retrieve(context) -> list[tuple[Rule, float]]
PolicyMemory.learn(field_name, decision, context, rationale, run_id) -> (Rule, str)
PolicyMemory.stats() -> dict
```

## Run record (written to runs/run_NNN.json)

Every run emits this. The eval harness reads these to plot improvement.

```json
{
  "run_id": "run_003",
  "context": {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "GATE"},
  "proposals": 7,
  "auto_applied": 5,
  "user_interventions": 2,
  "verify_attempts": 1,
  "verify_failures": 0,
  "tokens_in": 1840,
  "tokens_out": 260,
  "latency_ms": 3100,
  "rules_before": 4,
  "rules_after": 6
}
```

## The metric that is the demo

`user_interventions` per run, falling to zero across the sequence, while
`verify_failures` stays at zero. Plus `tokens_in` falling as memory replaces
reasoning — getting smarter and getting cheaper are the same curve.

## Inference

TensorMux, OpenAI-compatible:
- Base URL: `https://api.tensormux.com/v1`
- Model: `glm-4-7-flash`
- Key in `.env` as `TENSORMUX_API_KEY`

Only `classify.py` calls the model. Everything else is local and free.

## Governing vs descriptive context (found by the eval, not by design)

The first working harness showed 1 silent error: a rule learned from LinkedIn
offer letters auto-applied `employer_name: keep` to an Instagram post, where
this user wants it hidden. Confidence was high and the rule was retrieved --
retrieval tolerates partial context matches by design.

The fix distinguishes two kinds of context:

- **Governing** (`doc_type`, `platform`) -- the preference genuinely differs
  across these, so a rule may not act unattended unless they agree exactly.
- **Descriptive** (`issuer`) -- informs ranking, never blocks application.
  Which board issued a scorecard does not change whether a roll number is
  private.

Requiring strict agreement on ALL dimensions dropped the benefit to 16%;
restricting it to governing dimensions restored 40% with silent errors at 0.

## Measured result on real images (live model, 2026-09-06)

Four scorecards through the full pipeline against GLM-4.7-Flash:

| run | asked | from memory | model calls | output tokens | latency |
|-----|-------|-------------|-------------|---------------|---------|
| 1   | 6     | 0           | 1           | 1799          | 27.0s   |
| 2   | 6     | 0           | 2           | 7821          | 82.4s   |
| 3   | 1     | 5           | 1           | 625           | 10.5s   |
| 4   | 0     | 6           | 0           | 0             | 4.2s    |

By run 4 the agent asks nothing, calls no model, and finishes in 4.2s -- 6.4x
faster than run 1 at zero marginal token cost. The remaining 4.2s is OCR,
redaction and verification, all local.

Run 2 needed two model calls because the first exhausted a 6000-token
reasoning budget without emitting content. That variance is why truncation is
treated as failure rather than parsed optimistically.
