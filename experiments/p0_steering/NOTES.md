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

## Stage 0 — environment + baseline sanity — **DONE 2026-07-06** (2 carry-overs into Stage 1: weights revision, solver step count)

- [x] Model/version pins recorded (fork `07cb79fea`, `transformers 4.51.3` confirmed in-session; weights revision: not printed — capture next session)
- [x] Single-speaker sample generated and listened to — sounds normal (Josh, 2026-07-06: "sounded pretty good")
- [x] 2-speaker dialogue sample generated and listened to — sounds normal (same session)
- [x] Hook map verified against `docs/resources.md` §1
- [ ] Measured actual default inference steps + CFG behavior *(CFG default 3.0 confirmed; step count still unmeasured)*

Findings (2026-07-05 Colab L4 run; executed notebook committed as the run record):

- Environment: Colab L4 23GB, CUDA 13.0, fork @ `07cb79fea`, `transformers==4.51.3`
  installed cleanly. Voice presets shipped: en-Alice_woman, en-Carter_man,
  en-Frank_man, en-Mary_woman_bgm, en-Maya_woman, in-Samuel_man + 3 zh voices.
- **Hook map verified against the live model — all asserts passed:**
  `model.model.language_model.layers` → 28 × `Qwen2DecoderLayer`, hidden_size **1536**,
  head `VibeVoiceDiffusionHead` at **123.28M params**, acoustic latent dim **64**,
  `sample_speech_tokens(condition, neg_condition, cfg_scale=3.0)` → CFG default 3.0.
- **Design observation:** VibeVoice's own head is 123M params; our planned flow head
  is ~15M — an 8× shrink *on top of* the step reduction. Favorable for the paper if
  it works; a capacity risk to watch at the P1 gate (per constraint 5, evaluate the
  thin MLP fully before adding capacity).
- Automated acoustic sanity (local, `soundfile`): 24 kHz confirmed; single: 12.0s,
  −28.7 dBFS RMS, ZCR 3654/s; dialogue: 12.4s, −25.2 dBFS, ZCR 3540/s. **Both ZCRs
  are inside the 3,000–8,000/s speech band — the April 7 failure signature (~1,250/s)
  is absent.** Energy mildly front-weighted (~2:1 halves), 50–60% low-amplitude
  frames (pauses/turn gaps) — plausible for short scripted clips.
- Still open before Stage 0 closes: listening verdict (both clips), weights
  `revision`, and the actual default DDPM/DPM-Solver step count (search the demo
  script's args / generation config next session).

## Stage 1 — contrast-pair generation (2026-07-06, Colab L4)

- Calibration falsified the fixed-calls-per-step assumption: 116 hook calls for
  61 generated tokens (1 prefill + 60 positive + 55 negative) — the CFG negative
  pass fires only on speech-frame steps. Recorder rewritten position-aware
  (classification by cache_position chain); verify cell asserts n_gen−1 recovery.
- Per-turn speech_start/speech_end markers confirmed in the token stream →
  lead-in exclusion uses exact turn masking, not the drop-fraction fallback.
- GitHub-token clone friction in Colab (fine-grained PAT + VM recycle) — switched
  to drag-and-drop upload of contrast_pairs.py + p0_contrast.json; simpler, keep it.
- **Honesty check (Stage 1.4, Josh listening):** arousal contrast audible —
  pos lead-in gives excited delivery, neg noticeably flatter. Valence contrast
  NOT clearly audible. Matches the pre-registered PARTIAL trajectory (possible
  axis collapse to single "intensity" axis); Stage 2 consistency/independence
  diagnostics will decide quantitatively.
- Capture loop stats: (fill in: records saved / failures)
- vectors.pt: (fill in when placed in experiments/p0_steering/)

## Results

(fill in after the sweep)

## Verdict

(fill in against the pre-registered criteria)

## Artifacts

- Audio: `experiments/p0_steering/audio/` (gitignored)
- Steering vectors: `experiments/p0_steering/vectors.pt` (gitignored via *.pt)

## Follow-ups

(fill in)
