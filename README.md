# Verity

**An agent that learns what you're willing to share.**

Evorozen Apex — Category 6: Agentic OS & Workflow Automation

---

## The problem

People post exam scorecards, offer letters and photos of their cars to
celebrate something real. The same image carries a registration number, a roll
number, a date of birth, an employee ID, a licence plate. They didn't overlook
it — they were looking at the achievement, not the identifiers around it.

A generic PII detector doesn't solve this, because **there is no universal
right answer.** One person is proud to show their rank and hides their score.
Another shows their employer on LinkedIn and hides it on Instagram. A detector
that redacts everything identifier-shaped is wrong for everyone, just in
different directions.

Measured on our evaluation set, a competent generic policy agrees with one
specific user on **69% of field decisions**. The remaining 31% is the product.

## What Verity does

Before anything is posted, Verity reads the image, identifies each value,
decides what to do with it, acts, and then **verifies its own work**. Where it
hasn't earned the right to decide, it asks — and it remembers the answer.

```
image -> OCR -> retrieve memory -> propose -> act -> verify -> escalate -> reflect
                     |                |                            |
              learned rules      ask only what           re-OCR the output;
              + anchors          isn't earned            blur harder if the
                                                         text survived
```

## The result

Five real scorecards, in sequence, against GLM-4.7-Flash:

| run | asked | from memory | model calls | output tokens | time |
|-----|-------|-------------|-------------|---------------|------|
| 1 | 6 | 0 | 1 | 1799 | 60.8s |
| 2 | 6 | 0 | 2 | 7407 | 95.6s |
| 3 | 1 | 5 | 1 | 863 | 18.8s |
| 4 | **0** | 6 | **0** | **0** | **9.6s** |
| 5 | **0** | 5 | **0** | **0** | 16.2s |

By run 4 it asks nothing, calls no model, and finishes 6x faster at zero
marginal cost. Getting smarter and getting cheaper are the same curve.

Against an ablation with memory disabled over 12 documents: **40% fewer user
interventions, and zero silent errors.**

## How the learning actually works

**Confidence is recency-weighted.** A flat lifetime ratio made rules so sticky
that a user whose preference genuinely changed had to correct the agent seven
times. With geometric decay it adapts in two or three — and a rule that gets
contradicted loses the right to act unattended and starts asking again.

**Autonomy must be earned twice over.** A rule acts alone only when confidence
clears 0.75 *and* it rests on at least two agreeing observations. One decision,
however unambiguous it looked, is never enough to stop consulting the user.

**Memory stores recognition, not just preference.** Knowing you redact a
registration number doesn't tell the agent which pixels are one. Each rule also
records the labels that located that field before, so a later run resolves it
locally and makes **no model call at all**. This is why token cost goes to zero
rather than merely down.

**Context that governs vs context that describes.** Our first working eval
showed one silent error: a rule learned from LinkedIn offer letters
auto-applied to an Instagram post, where the user wanted the opposite. Acting
unattended now requires exact agreement on *governing* context (document type,
platform) while *descriptive* context (which exam board) only informs ranking.
Requiring exact agreement on everything dropped the benefit to 16%; this
distinction restored 40% with silent errors at zero.

**The agent never learns from its own unconfirmed guesses.** It briefly did,
and one unreviewed run taught it to redact the user's score. Acting on a guess
is reasonable; recording it as a confirmed preference is not.

## Verification is deterministic

After redacting, Verity re-runs OCR on its **own output** and asserts the
target strings are unrecoverable, including partial fragments — a plate leaking
as `MH12AB12` is still a plate. It never asks a model whether its own work
succeeded.

This matters because blur strength has no single right value. Measured on our
fixtures: document text becomes unreadable at blur radius 4, but a large
high-contrast licence plate **survives radius 4 and needs 6**. A fixed-strength
redactor ships that leak silently. Verity starts gentle, checks, and escalates
to an opaque fill if blur isn't enough.

## Running it

```bash
pip install -r requirements.txt
echo "TENSORMUX_API_KEY=tmx_..." > .env      # optional; see below
python -m demo.generate                       # synthetic fixtures, no real data

python cli.py reset
python cli.py sequence --teach 3 --policy     # the full demonstration
python cli.py memory                          # what it learned
python -m eval.harness                        # the ablation
```

Without an API key the pipeline still runs end to end — OCR, redaction and
verification are entirely local. Only fresh classification needs the model, and
by run 4 it isn't calling one anyway.

## Evorozen Neural DB is the backbone

```bash
echo "EVOROZEN_API_KEY=evo_live_..." >> .env
python cli.py sequence --teach 3 --policy   # Neural DB by default
python cli.py stats                          # run history, read from Neural DB
python cli.py memory                         # learned policy, read from Neural DB
python cli.py sequence --local               # offline fallback
```

Two tables, both live in Neural DB:

- **`policy_rules`** -- every learned rule: field, decision, confidence,
  supporting and contradicting run ids, and the layout anchors that locate the
  field. This is the agent's memory. Reading it is how run 4 answers without
  calling a model.
- **`run_history`** -- per-run telemetry: questions asked, fields resolved from
  memory, model calls, tokens, latency, verification attempts. The improvement
  curve is the product's central claim, so it lives in the database rather than
  a local file a judge has to take on trust. `cli.py stats` reads it back.

`agent/neural_store.py` implements both against
`https://pulse.evorozen.com/api/neural` using `create_schema`, `upsert_data`,
`select_data`, `insert_data` and `delete_data`.

Neural DB is the default backend. `agent/store.py::default_store()` falls back
to local files only when the service is unreachable or `--local` is passed --
the document-handling half of the pipeline is local by design and must keep
working without a network.

**Only rules travel.** A rule holds a field name, a decision, a confidence and
the layout anchors that located that field -- never the document, never the
OCR text, never the identifier itself. Images and extracted values stay local.
A privacy tool that uploaded people's scorecards to a third party in order to
remember their preferences would have defeated its own purpose, so the split
is deliberate rather than incidental.

Reads are cached in memory and only writes go over the wire: `all_rules()` is
consulted many times per run, and the free tier allows 50 calls.

Two things worth knowing if you build against this API yourself: `prompt` is
required on every request including the deterministic ones (the published
snippet omits it), and the `chat` action was returning
`All LLM providers failed` during development, so nothing here depends on it.

## Pluggable memory backend

`PolicyMemory` talks to a `RuleStore` (see `agent/store.py`), which is the
complete persistence contract: read all rules, write one, delete one. The
default `LocalJSONStore` keeps one JSON file per rule.

That seam exists so the backing store is a substitution rather than a
refactor. Swapping the local directory for a hosted virtual-database service
means implementing three methods; nothing in the learning logic changes.

Local JSON was chosen over a vector database deliberately: with rules
numbering in the tens, embeddings would add infrastructure and remove the
ability to explain why a rule fired. Every rule is a file you can open and
read, which matters for a system whose central claim is that its decisions are
auditable.

## Built with AO

Work was decomposed into frozen-interface work packages (`SPEC.md`, `TASKS.md`)
so AO could run leaf modules as parallel workers in isolated worktrees while
integration stayed serial.

## Honest limitations

- Verification proves a string is unreadable **by one OCR engine**. It is not a
  claim that the original pixels are unrecoverable by any means.
- QR codes are detected as regions but their payload is never decoded, so the
  agent reasons about them without knowing what they encode.
- No live platform integration. The Chrome/LinkedIn interception described in
  our original design is not built.
- Field naming depends on a 30B model and is inconsistent between runs; we
  canonicalise known synonyms, which is a patch over the underlying variance.
- The rehearsal policy in `cli.py` is a demo aid, not a learned artefact.

## Layout

```
agent/memory.py      persistent policy memory, confidence, anchors
agent/store.py       pluggable persistence backend (RuleStore protocol)
agent/reflect.py     corrections -> rules; refuses to learn from guesses
agent/classify.py    the only file that calls a model
agent/ocr.py         RapidOCR with content-hash caching
agent/redact.py      escalating blur, opaque fill past strength 3
agent/verify.py      deterministic re-OCR check
agent/loop.py        orchestration
eval/harness.py      the ablation
eval/live_sequence.py  live measurement on real images
cli.py               the demo surface
```
