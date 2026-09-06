# AO work packages

Read SPEC.md first. Interfaces there are frozen: build to them exactly.
Each task owns disjoint files, so workers can run in parallel without conflicts.

Do not edit `agent/memory.py`. It is built, tested, and other tasks depend on it.

---

## W1 — OCR adapter  ·  owns `agent/ocr.py`  ·  P0

Implement `extract(image_path) -> list[Token]` per SPEC.md.

Use `rapidocr-onnxruntime` (pip-installable, no system binary — do NOT use
pytesseract, we are not installing Tesseract on Windows during a hackathon).
Add it to requirements.txt.

- Return word- or line-level boxes with real pixel coordinates.
- Normalise confidence to 0..1.
- Must handle: a screenshot of a scorecard, a photo at an angle, low contrast.
- Cache results by file hash in `runs/.ocr_cache/` — the verifier re-OCRs
  constantly and we cannot afford to redo work.

Done when: `python -m agent.ocr demo/sample.png` prints tokens with boxes.

---

## W2 — Redaction engine  ·  owns `agent/redact.py`  ·  P0

Implement `apply(image_path, boxes, out_path, strength) -> str` per SPEC.md.

- Pillow only.
- `strength` escalation: 1 = box blur radius ~8 with 4px padding;
  each level up increases radius ~1.8x and padding ~6px.
- At strength >= 3, stop blurring and draw an opaque filled rectangle.
  Blur is defeatable; past a point we must guarantee removal.
- Never mutate the input file.

Done when: given a box, output image has that region visibly destroyed at each
strength level, and strength 3 is provably unrecoverable (solid fill).

---

## W3 — Classifier  ·  owns `agent/classify.py`  ·  P0

Implement `propose(tokens, context, rules) -> (list[Proposal], Usage)`.

This is the only file that calls the model. TensorMux, OpenAI-compatible
(see SPEC.md). Use the `openai` SDK pointed at the TensorMux base URL.

Critical behaviour:
- Rules passed in with `rule.auto == True` must be applied WITHOUT consulting
  the model for that field. Set `source="memory"`, `needs_user=False`.
  This is what makes later runs cheaper — do not skip it.
- Only fields not covered by an earned rule go to the model.
- If every field is covered by memory, make ZERO model calls and return
  `Usage(tokens_in=0, tokens_out=0)`. This is the headline result.
- Model output: strict JSON, validated. On parse failure retry once with the
  error appended, then fall back to `needs_user=True` for everything.

Return `Usage` as a dataclass with `tokens_in`, `tokens_out`, `latency_ms`,
`model_calls`.

Done when: with a populated memory the function returns proposals having made
no network call at all.

---

## W4 — Verifier  ·  owns `agent/verify.py`  ·  P0

Implement `check(out_path, must_be_absent) -> VerifyResult` per SPEC.md.

- Re-run `agent.ocr.extract` on the REDACTED image.
- For each string in `must_be_absent`, check it is not recoverable. Compare
  normalised (strip whitespace, casefold, drop punctuation) and also check
  fuzzy partial matches — "MH12AB1234" leaking as "MH12AB12" is still a leak.
- Return the specific strings that survived, not just a boolean.

Done when: blurring at strength 1 a large clear number is detected as STILL
READABLE and the loop escalates. Prove this with a real failing case — if your
verifier never fails, it is not verifying anything.

---

## W5 — Synthetic demo data  ·  owns `demo/generate.py`  ·  P0

Generate 8 realistic-but-fake inputs. No real personal data goes in the video.

- 4 exam scorecards (GATE / JEE / UPSC / SSC styling) with: candidate name,
  roll no, registration no, DOB, score, All India Rank, a QR-ish block.
- 2 offer letters / certificates with employee ID and salary.
- 2 vehicle photos — synthesise a plate onto a stock-ish car rectangle.
- Vary layout and font per document. If all 8 are identical templates the
  learning result is fake and a judge will spot it in one question.

Emit `demo/manifest.json`: for each image, its context dict and the ground-truth
list of strings that MUST be redacted. The eval harness scores against this.

Done when: 8 varied PNGs exist and OCR reads them.

---

## W6 — Eval harness  ·  owns `eval/harness.py`  ·  P1

Runs the 8 demo inputs in sequence against a fresh memory, simulating a
consistent user policy from `demo/manifest.json`, and emits:

- `runs/curve.json`: per-run interventions, tokens, latency, verify failures,
  and leak count vs ground truth.
- An ASCII chart to stdout of interventions falling to zero.

Also run an ABLATION: same 8 inputs with memory disabled. Interventions stay
flat. Two lines on one chart is the single most persuasive artifact you can
put in front of these judges.

Done when: `python -m eval.harness` prints both curves.

---

## W7 — Terminal dashboard  ·  owns `ui/dashboard.py`  ·  P1

`rich`-based live view for the demo video. Left: current run's proposals with
source badges (MEMORY / MODEL / ASKING). Right: the memory panel — rule count,
confidences, and rules highlighting as they fire. Footer: the three counters.

No web UI. A terminal that visibly shows memory files appearing is faster to
build and reads better on video than a half-finished web app.

---

## Owned by the integrator (do not take these)

- `agent/loop.py` — orchestration, run records, retry escalation
- `agent/reflect.py` — turning user corrections into generalised rules
- `cli.py` — entry point
