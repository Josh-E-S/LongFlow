# p1_flow_head — NOTES

Gate-check stage of P1 (flow-head baseline). Full-run protocol follows only if
this gate passes. Resources: `docs/resources.md` §2; predecessor failure this
refutes-or-confirms: finding N2 in `docs/negative-results.md`.

## Hypothesis

A ~16.6M-param OT-CFM head at 4 NFE, conditioned on VibeVoice's own cached
hidden states (captured from full inference including the tokenizer feedback
loop), produces intelligible speech through the frozen σ-VAE decoder. This is
the pre-registered conditioning experiment: identical head recipe to the April 7
failure (N2), with the conditioning source as the only changed variable.

## Setup

- Date started: 2026-07-07
- Hardware: Colab L4; cache persisted to Google Drive (resumable across VM recycles)
- Cost budget: ≤ $10 GPU for the gate
- Model pin: `microsoft/VibeVoice-1.5B` @ `c00898d257e6b46004e3e2866a47534085fb685a`, fork @ `07cb79fea`
- This repo: gate run used the bundle built at `96c645f` (trainer commit);
  batched-caching validation used `bb0736a`; 10K run uses `1cb8c24` or later —
  record the actual bundle commit in this line for every run
- Data: LibriTTS-R `train.clean.360` (streaming), ~800 utterances, voice prompt =
  first 3s of each utterance's own ground-truth audio, text = `text_normalized`
- Head: width 640, 4 layers, 16.64M params; Adam lr 2e-4, betas (0.9, 0.95),
  bs 1024, 5K steps, **EMA 0.999** (0.9999 would stay ~at init over 5K — see
  trainer test), fp32

## Gate criteria — pre-registered

| Verdict | Condition |
|---|---|
| PASS | (a) ROUNDTRIP SANITY: cached ground-truth latents decode to normal VibeVoice-quality speech (proves capture+decode path); (b) training loss decreases smoothly, no NaN; (c) flow-head samples at 4 NFE decode to intelligible speech, recognizably the prompt speaker, ZCR in the 3–8k/s band |
| PARTIAL | Intelligible but clearly degraded vs roundtrip → try NFE 8/16, more steps, or width 768 before re-gating; diagnose before scaling |
| FAIL | April-7 signature (sharp but unintelligible, ZCR ≪ 3k) at any NFE despite healthy loss → conditioning hypothesis wrong even with pre-validated states; stop, re-examine capture correctness first (most likely bug class), then N2 implications |

Listening step: roundtrip clip AND flow-decoded clips, A/B against the original
generation. Never skipped.

## Stage log

- 2026-07-07/08 (overnight, Colab L4): caching complete — **800/800 utterances**
  at ~9s/utt, filters healthy, fully resumable via Drive (survived at least one
  session boundary). One notebook bug: cold-start only imports on the READY
  branch, so a skipped-DRAG first run left `SampleCapture` undefined and the
  loop's try/except mislabeled it as per-utterance skips — fixed live; make
  setup errors fail loudly in future notebooks.
- 2026-07-08: **Roundtrip sanity PASS (gate criterion a)** — cached ground-truth
  latents decoded through the frozen σ-VAE sound like normal VibeVoice speech
  (Josh listening). Un-scale + decode call worked first try. Capture and decode
  paths validated; training data is clean by construction.

## Results

- Training (5K steps, bs 1024, 41,450 pairs from 800 utts): loss 2.02 → 0.29,
  smooth monotonic, no NaN. **Criterion (b) PASS.**
- Listening (Josh): flow clips sound good; one small start-of-speech artifact
  that clears ("warm-up"); teacher-quality otherwise. **Criterion (c) ears PASS.**
- Objective parity vs teacher (5 utts, `gate_metrics.json`; WER reference =
  Whisper transcript of the teacher clip):

| config | mean WER vs teacher | mean sim to teacher |
|---|---|---|
| flow4 | **0.030** | **0.984** |
| flow16 | 0.030 | 0.978 |

- Durations identical, F0 within a few Hz. **flow4 ≡ flow16 — 4 NFE already
  saturates this head**; strong signal for the P2 1–2 NFE push.
- Known minor, correctly attributed (Josh, on re-listen): the brief onset
  artifact is present **identically in the teacher clips** — it is VibeVoice's
  own utterance-onset behavior (same family as the P0 jingle/BGM inheritance),
  carried into the cache and faithfully reproduced by the head. NOT a
  distillation defect; more data won't and shouldn't change it. Covered by the
  existing convention: throwaway first sentence + trim for demo/eval audio.
  (Earlier "small-data signature, watch at 75K" note was wrong — corrected.)

## Verdict

**PASS (2026-07-08).** All three pre-registered criteria met: roundtrip clean,
training healthy, 4-NFE samples intelligible and speaker-faithful (0.984 sim,
0.030 WER vs teacher). **N2's conditioning hypothesis confirmed in both
directions** — identical head recipe to the April 7 failure succeeds when
conditioned on VibeVoice's own pre-validated hidden states. Proceed to the full
P1 run: 75K caching, full training, DDPM-baseline eval (WER, WavLM SIM,
wall-clock), then tag `p1-baseline`.

## Artifacts

- Cache: Google Drive `longflow_p1_cache/` (~1 GB at 800 utts)
- Checkpoint: Drive `longflow_p1_ckpt/`
- Audio: `experiments/p1_flow_head/audio/` (gitignored)

## Batched caching (2026-07-07, validated on L4)

- `BatchedSampleCapture` + token-stream row attribution: all invariants held on
  the real model, 8/8 utterances split with correct dims; batched-captured
  latents decode to clean speech (Josh listening) — attribution proven by ear.
- **Speedup 4.9× (8.4 → 1.7 s/utt at batch 8).** VibeVoice batch generate uses
  left-padding (confirmed via attention-mask edges), so generated streams align
  at the shared padded prompt length.
- New cost math: 75K ≈ 35 GPU-h (was 187); **10K ≈ 5 h — one overnight Colab
  session.** Vast.ai deferred until a run actually exceeds Colab session limits.
- Not done (deliberate): length-bucketing batches to cut padding waste — note
  for the 75K run if it happens; ~5h for 10K doesn't justify the complexity.

## 10K scaling run (2026-07-08)

- Caching: 10,000 utts complete overnight (~18 skipped batches ≈1.4%, all the
  known edge case, fixed in `3bac875` for future runs). 478,191 frame pairs.
- Training: 20K steps, bs 1024, EMA 0.9995 — loss 2.00 → 0.965, smooth, **still
  declining at stop** (~0.004/1K steps) → undertrained; longer run warranted.
  NB: the 800-run's final loss 0.29 is a memorization artifact, NOT comparable —
  ~0.97 is the honest population-level CFM loss with this conditioning.
- **First held-out (generalization) evaluation** — 5 never-trained sentences,
  teacher-forcing parity (`eval10k_metrics.json`, `ab_scaling_metrics.json`):

| model | held-out WER | held-out sim | wins |
|---|---|---|---|
| 10K-utts | **0.105** | **0.824** | **5/5** |
| 800-utts (gate ckpt) | 0.146 | 0.677 | 0/5 |

- **Conclusions:** (1) the gate's 0.984 train-set sim was partly memorization —
  held-out is the metric that matters from now on; (2) data scaling decisively
  works and is not saturated at 10K; (3) full P1 run should scale BOTH data
  (→75K per the original plan) and steps (≥50K), with the fixed batched capture.
- Caveats logged: n=5, single held-out speaker (consecutive sorted files —
  next eval should hold out utterances across many speakers); listening
  calibration of sim 0.82 pending (Josh).

## Follow-ups

- ~~Josh: listen/calibrate~~ **DONE (2026-07-08):** Josh rates flow4 vs teacher
  "75–85% close — sounds good, just a bit of noise, almost like the voice is
  doubled and offset slightly, not as clear." Ears match the ECAPA number
  (0.82) almost exactly → the sim metric is perceptually calibrated in this
  range. Artifact signature (phasey/doubled) → two candidate causes to split
  with an NFE-16 A/B on the same checkpoint: integration error (more steps
  fixes) vs velocity-field imprecision (full run fixes). Pending.
- Full P1 run: 75K cache (~35 GPU-h with batching — exceeds Colab sessions →
  Vast.ai onboarding, or chunked Colab) + dialogue caching (turn-boundary
  coverage per data plan) + ≥50K-step training + proper multi-speaker held-out
  eval + wall-clock vs DDPM baseline → tag `p1-baseline`.
- Then P2 (MeanFlow 1–2 NFE) — timeline priority per dots.tts (docs/related-work.md).
