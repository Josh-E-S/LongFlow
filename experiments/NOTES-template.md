# <experiment name> — NOTES

> Copy this file to `experiments/<name>/NOTES.md`. Fill in everything above the
> Results line BEFORE running anything (pre-registration). Fill in the rest after,
> even on FAIL — negative results are first-class citizens in this project.

## Hypothesis

What we expect to be true, stated so that the gate criteria below can refute it.

## Setup

- Date / hardware / cost budget:
- Model + weights pin (repo ID, revision):
- Code pin (git commit of this repo):
- Config (or path to `configs/...`):
- Data (source, subset, sample count):

## Gate criteria — written before the run

| Verdict | Condition |
|---|---|
| PASS | <objective, checkable conditions — thresholds, not vibes> |
| PARTIAL | <what a salvageable middle result looks like and how the plan reshapes> |
| FAIL | <what refutes the hypothesis; what the project does instead> |

Automated checks (script output attached below): <e.g. WER delta, ECAPA floor, F0 monotonicity>
Listening step (never skipped, never automated): <what to listen for, how many samples>

## Results

- Automated metrics:
- Listening notes (who listened, what was heard):
- Surprises / anything the metrics missed:

## Verdict

PASS / PARTIAL / FAIL — one paragraph justifying it against the pre-registered criteria.
If the criteria themselves turned out to be wrong, say so here rather than bending them.

## Artifacts

- Audio: `experiments/<name>/audio/` (gitignored; curated keepers -> `samples/` via git-lfs)
- Checkpoints / cached tensors: <where>
- Git tag (if a phase gate): <e.g. p0-pass>

## Follow-ups

What this result changes about the next experiment, the docs, or `docs/negative-results.md`.
