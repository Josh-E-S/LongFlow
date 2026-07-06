# p0_steering — NOTES

Spec: `docs/experiments/p0-steering.md`. Resource pins: `docs/resources.md`.

## Hypothesis

Activation steering vectors, extracted from VibeVoice's own contrast-pair generations
and injected into the frozen Qwen2.5 backbone, can audibly shift the emotional
character of generated speech without degrading intelligibility or speaker identity.

## Setup

- Date started: 2026-07-05
- Hardware: Google Colab L4 (Stage 0 sanity); Vast.ai RTX 3090 or Colab for the sweep
- Cost budget: ≤ $15 total
- Model + weights pin: `microsoft/VibeVoice-1.5B` (HF, MIT, ungated) — record `revision` after first download
- Backbone code pin: `vibevoice-community/VibeVoice` @ `07cb79fea`
- This repo's commit: record at each stage
- Data: 10 neutral scripts (Stage 1), 3 held-out scripts (Stage 3) — written by hand, varied domains

## Gate criteria — pre-registered (from the P0 spec)

| Verdict | Condition |
|---|---|
| PASS | Some (α, layer) gives clearly audible affect shift, WER within +10% relative of unsteered, ECAPA to reference ≥ 0.85× unsteered, and the multi-speaker probe leaves speaker A unchanged |
| PARTIAL | Audible shift exists but control range is narrow or axes collapse into one — C2 reshapes to single-axis "intensity steering" |
| FAIL | No audible shift before intelligibility breaks at any α/layer — C2 dropped; paper proceeds on C1+C3+C4; write the negative result |

Automated checks: Whisper-large-v3 WER vs script, ECAPA cosine to reference, F0 mean/var + energy monotonicity with α.
Listening step: every sweep sample; note α where affect becomes audible and α where speech degrades.

## Stage 0 — environment + baseline sanity

- [ ] Model/version pins recorded (weights revision, fork commit, transformers version)
- [ ] Single-speaker sample generated and listened to — sounds normal
- [ ] 2-speaker dialogue sample generated and listened to — sounds normal
- [ ] Hook map verified against `docs/resources.md` §1 (layers path, hidden_size 1536, generate() entry point, speaker→position mapping)
- [ ] Measured actual default inference steps + CFG behavior (README reconciliation #1)

Findings:

(fill in)

## Results

(fill in after the sweep)

## Verdict

(fill in against the pre-registered criteria)

## Artifacts

- Audio: `experiments/p0_steering/audio/` (gitignored)
- Steering vectors: `experiments/p0_steering/vectors.pt` (gitignored via *.pt)

## Follow-ups

(fill in)
