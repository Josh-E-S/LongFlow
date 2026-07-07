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
- Capture loop stats: **80/80 records, 0 failures**, 35.5 min on L4. Frames kept
  per record: 57–149 (turn masking active; lead-in excluded).
- vectors.pt in experiments/p0_steering/ (13.8 MB, gitignored); honesty wavs in audio/.
- Weights revision: `c00898d257e6b46004e3e2866a47534085fb685a`.
- **Solver steps resolved (README reconciliation #1):** model default
  `ddpm_inference_steps=20`, but the demo sets 10 (`inference_from_file.py:365`).
  Head runs cond+neg CFG pair per step → baseline is 10 steps × 2 head passes/frame.
- Calibration quirk: the tiny 2-sentence script emitted only ONE speech_end this
  session (vs 3 segments in the first session) — segment-marker emission is
  variable on short scripts. Real two-turn capture scripts produced boundaries
  reliably (0 failures, healthy frame counts). Keep the 0-frames guard.

## Stage 2 — extraction + diagnostics (2026-07-06, local CPU)

- Naive consistency (spec's expectation was >0.4 mid-layers): **0.023–0.042 ≈ the
  1/√1536 = 0.026 random baseline** on both axes. At face value: noise.
- More sensitive tests found real transferable signal the consistency stat missed:
  - Permutation test (2000 label-flips): valence per-layer p<0.05 at L0, L1, L17;
    arousal global p=0.33 (norm-test underpowered).
  - **Leave-one-script-out AUC: arousal 0.70 @L17/L18; valence 0.825 @L17**
    (chance 0.5) — a direction from 9 scripts classifies the held-out script's
    poles well above chance, concentrated exactly in the predicted mid-stack.
- Interpretation: real but weak shared emotion component; K=2 per pole makes
  per-script directions noise-dominated (norms swamped by generation-to-generation
  variance), which floors the pairwise-cosine stat. Valence transfers better than
  arousal internally — opposite of the listening impression; not contradictory
  (states vs rendered audio).
- Decision: **Stage 3 smoke test before any recapture** — inject the L17/L18
  directions at a few α values, 3 clips, ~15 min GPU. Injection is the actual
  go/no-go question; if inaudible, THEN recapture with K=4 + stronger lead-ins,
  then Expresso fallback in that order.
- directions.pt saved (extract_all output: unit directions, norms, consistency,
  candidates, independence).
- **Known artifact, deliberately not chased:** spontaneous podcast-jingle/BGM
  right after the opening line in almost all generations — inherited from
  VibeVoice's podcast training data, generated through the same latent stream
  as speech (no off switch short of retraining = out of scope). Harmless to
  extraction: appears in both poles so the pos−neg subtraction cancels it, and
  it clusters in the lead-in turn we already mask out. Conventions adopted:
  clean voice presets only (never `*_bgm`), and for eval/demo audio add a
  throwaway first sentence and trim it before metrics/listening.

## Stage 3 — injection (2026-07-06, in progress)

- Smoke test v1 (UNIT directions, α∈{2,8,−8}, layers 17+18 simultaneously):
  **all steered outputs unintelligible gibberish; baseline clean.** Mechanism
  confirmed (injection reaches the speech pipeline); scale mis-calibrated —
  unit-vector α is the wrong unit. Raw pos−neg contrast norms are ~1.1, so
  α=2 × unit × 2 compounding layers ≈ 4–8× the largest natural emotion gap,
  every frame, further amplified by CFG. The spec's α∈{0.5..8} grid implicitly
  assumed natural units; recorded as an empirical finding.
- Smoke test v2: single layer (L17), directions at natural scale
  (unit × raw norm), α∈{0.5,1,2,4}. Scale measurements: **typical hidden norm
  @L17 = 7.6; raw contrast norms @L17: arousal 0.4622, valence 0.4636** — the
  natural emotion contrast is ~6% of the residual stream. (v1 gibberish therefore
  injected ~4–8× the natural gap per layer, compounding across 2 layers.)
  Near-identical norms across axes = noise-power signature (directions still
  noise-dominated; real signal is a small rider). ~95 steered steps/clip.
- **v2 listening verdict (Josh): AUDIBLE EFFECT.** α=1.0 best on both axes —
  subtle but definite emotion: pitch rises in places, louder, present across the
  whole clip ("9/10 blind listeners would call arousal α=1 more expressive").
  α=4 degrades voice quality on both axes (still audible, sounds weird) →
  usable control window ≈ α∈[1, 3] natural units. Caveat: both axes described
  as "more expressive/excited" — axis-collapse (PARTIAL scenario) still live;
  F0/energy stats in the full sweep to arbitrate.
- Full sweep launched: 3 held-out scripts × {baseline; 2 axes × α{0.5,1,2,3} @L17;
  2 axes × α{1,2} spread across L14–19 band (per-layer α divided by band size)}
  = 39 clips + manifest + reference voice → sweep.zip; metrics run locally.
- Keep one over-steered gibberish clip as the failure sample (spec deliverable).

## Results

**Full sweep (39 clips, 3 held-out scripts), objective metrics — 2026-07-07, local
faster-whisper large-v3 int8 + ECAPA + parselmouth (`sweep_metrics.json`):**

| condition | WER | sim× (vs base) | ΔF0 | ΔF0σ | gates |
|---|---|---|---|---|---|
| arousal α0.5 L17 | 0.000 | 0.97 | +14 | +14 | PASS/PASS |
| arousal α1.0 L17 | 0.000 | 1.00 | +6 | +11 | PASS/PASS |
| arousal α2.0 L17 | 0.427* | 0.89 | +14 | +14 | fail/PASS |
| arousal α3.0 L17 | 0.000 | 0.64 | +6 | +15 | PASS/fail |
| valence α0.5 L17 | 0.000 | 1.03 | +6 | −5 | PASS/PASS |
| valence α1.0 L17 | 0.000 | 1.06 | +7 | −1 | PASS/PASS |
| valence α2.0 L17 | 0.000 | 0.83 | +10 | +14 | PASS/fail |
| valence α3.0 L17 | 0.010 | 0.63 | +35 | +16 | PASS/fail |
| arousal α1.0 band | 0.010 | 0.98 | +1 | +2 | PASS/PASS |
| arousal α2.0 band | 0.000 | 1.00 | +5 | +3 | PASS/PASS |
| valence α1.0 band | 0.010 | 0.90 | +14 | +2 | PASS/PASS |
| valence α2.0 band | 0.000 | 0.67 | +27 | +15 | PASS/fail |

\* one clip (airport_arousal_a2.0_L17) Whisper-hallucinated completely
("I trust the temple. Sadist, Matt, Claire, McGarge...") — kept as the
over-steer failure sample. Baselines: sim 0.63–0.77, F0 199–218.

- **Usable window confirmed: α∈[0.5, 1] @L17 passes all gates on both axes**
  (WER 0.000, sim 0.97–1.06×), matching the listening verdict exactly.
- **Band (L14–19, α split per layer) is more robust than single-layer**: arousal
  α=2 band = WER 0.000 / sim 1.00×. Wider control range via multi-layer spread.
- **Against full axis collapse:** arousal raises F0 *variability* (+11..+15 σ)
  at all α; valence at usable α shifts F0 mean without widening σ (−5..−1).
  Distinct prosodic fingerprints.
- Gate scorecard: audible ✓, WER ✓, speaker-sim ✓; multi-speaker probe below.

**Multi-speaker probe (2026-07-07):** 2-turn Alice→Frank dialogue; SegmentGate
(embed_tokens END-watch) flips steering ON after Alice's turn. L17 α=1 arousal:
44 steps steered, all in Frank's turn. Split at the token-derived turn boundary:
- Alice segment: 0.689 vs Alice ref (baseline's own Alice: 0.663); 0.707 vs
  baseline Alice segment → **within normal generation variation, unchanged.**
- Frank segment: 0.460 vs baseline Frank → **steering moved the steered speaker
  substantially while leaving the unsteered one alone.**
- Band-config probe void: generation degenerated (4.7s, single segment,
  ~0 sim to everything) — 1/6 degenerate rate at this config not established,
  logged only. Turn-marker emission variability remains a known quirk.
- Listening confirmation of the probe pair: PENDING (Alice identical? Frank
  more expressive?).

## Verdict

(fill in against the pre-registered criteria)

## Artifacts

- Audio: `experiments/p0_steering/audio/` (gitignored)
- Steering vectors: `experiments/p0_steering/vectors.pt` (gitignored via *.pt)

## Follow-ups

(fill in)
