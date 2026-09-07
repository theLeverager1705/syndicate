# Demo script (target 3:30-4:00; Devpost requires 3-5 minutes)

REQUIRED by Devpost: the video must show the AO dashboard including the
total number of AO sessions used during development. That is Shot 0 below
and it is not optional.

Total spoken words: ~430. Read at a normal pace, do not rush. Every number
below is real and reproducible — do not round them up.

Record in three takes and cut them together. Do not attempt one continuous
take; you will lose an hour to retries.

---

## Before you record

```bash
cd ~/syndicate
python cli.py reset
python cli.py sequence --teach 3 --policy
```

Run this once first. It warms the OCR cache, so your recorded run is faster
and cleaner. Then `python cli.py reset` again immediately before recording.

- Terminal font size 16pt or larger. Judges watch on laptops.
- Maximise the window. Dark background.
- Close Slack, Discord, notifications.
- Have `runs/before_after_scorecard.png` open in a second window for shot 1.

---

## Shot 0 — AO (0:00 – 0:30)  [REQUIRED]

**On screen:** the Agent Orchestrator window. Show the Projects sidebar with
`syndicate`, the session list, and the ACTIVITY panel. Pause long enough that
the session count is legible.

> "The project was built through AO. This is the workspace it created against
> the repository, and these are the sessions that ran during the build."

Say only what AO actually did. Do not describe work it did not do — that is the
one claim a judge can check directly.

---

## Shot 1 — the problem (0:30 – 0:55)

**On screen:** the unredacted `gate_2024.png`, full frame. Cursor circles the
registration number, roll number, date of birth.

> "Every year, thousands of students post their exam scorecards to celebrate a
> rank. This one also shows a registration number, a roll number, and a date of
> birth — everything you need to impersonate them on the results portal.
>
> They didn't miss it. They just weren't looking for it, because they were
> looking at their rank."

**Cut.**

---

## Shot 2 — teaching it (0:25 – 1:15)

**On screen:** terminal. Run `python cli.py sequence --teach 3 --policy`.
Let run 1 play. Do not speed it up — the pauses are the point.

> "This is Verity. Before anything gets posted, it reads the image, works out
> what each value is, and asks me what I'm willing to share.
>
> Run one, it asks about all six fields. I keep my name, my score, my rank —
> that's the achievement. I hide the registration number, the roll number, the
> date of birth.
>
> Watch the bottom line. It says 'verify failed at strength one — escalating'.
> It blurred the fields, then re-read its own output with OCR, found the text
> was still machine-readable, and blurred harder. It doesn't trust itself."

**Point at the ASK badges, then the escalation line.**

---

## Shot 3 — it stops needing you (1:15 – 2:00)

**On screen:** runs 3, 4, 5 playing. Let the source column fill with green
MEMORY badges.

> "Run three, it only asks once. Run four, it asks nothing — every field came
> from memory. And look at the counters: zero model calls. Zero tokens.
>
> Run one took sixty seconds and eighteen hundred tokens. Run four took nine
> point six seconds and cost nothing, because it no longer needs to reason
> about a document it has already understood.
>
> It didn't just learn what I want redacted. It learned how to *find* those
> fields — which labels point at them — so it can resolve them locally without
> calling a model at all."

**Cut to `python cli.py memory` — the policy table, AUTO badges, confidences.**

---

## Shot 4 — the part that's actually hard (2:00 – 2:35)

**On screen:** the eval output. Run `python -m eval.harness`.

> "Getting an agent to stop asking is easy. Getting it to stop asking without
> getting reckless is the hard part.
>
> Two things stop that here. First, it never learns from its own unconfirmed
> guesses — only from decisions I actually made. Early on it was recording its
> own proposals as my preferences, and one unreviewed run taught it to redact
> my score. That's fixed.
>
> Second, context. It learned from LinkedIn that my employer name is fine to
> show. When the same field appeared on an Instagram post, it refused to apply
> that rule and asked me instead — because a preference isn't transferable just
> because you're confident about it.
>
> Against an ablation with memory switched off: forty percent fewer
> interruptions, and zero silent errors."

---

## Shot 5 — close (2:35 – 3:00)

**On screen:** the before/after composite.

> "Deterministic verification, not a model grading its own homework. Memory
> that earns the right to act and loses it when it's wrong.
>
> Built with AO orchestrating the work packages in parallel worktrees.
>
> It gets faster, cheaper, and less annoying every time you use it. That's the
> whole idea."

---

## Numbers you may quote (all measured, all reproducible)

| claim | value | where |
|---|---|---|
| run 1 → run 4 questions | 6 → 0 | `cli.py sequence --teach 3 --policy` |
| run 1 → run 4 model calls | 1 → 0 | same |
| run 1 → run 4 output tokens | 1799 → 0 | same |
| run 1 → run 4 latency | 60.8s → 9.6s | same |
| interventions vs ablation | 40% fewer | `python -m eval.harness` |
| silent errors | 0 | same |
| blur defeating OCR | doc text r=4, plate r=6 | documented in `agent/redact.py` |

## Do not claim

- Do not say "100% accurate" or "guaranteed". The verifier proves a specific
  string is unreadable by one OCR engine, not that the image is unrecoverable.
- Do not say it works on video, or on any platform live. It does not yet.
- Do not imply the LinkedIn/Instagram example ran in the video demo — it comes
  from the eval harness. Say "in evaluation" if you mention it.
