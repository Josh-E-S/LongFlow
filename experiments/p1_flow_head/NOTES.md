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

## Integration findings (2026-07-08, first closed-loop runs)

- **NFE-16 A/B (Josh):** 16 ≈ 4, barely better — the "stepping + light
  ghosting/doubling" artifact is velocity-field imprecision (underfit), not
  integration error. Teacher-forced long clips otherwise "great and clear."
- **First end-to-end generation through our head (FlowHeadPatch replacing
  sample_speech_tokens live):** audio starts strong then progressively fades to
  near-silent digital noise by ~60s; generations run long (64s audio vs
  parent's 39s for the same text — no clean termination). Classic feedback
  drift / exposure bias: the loop consumes OUR latents for the first time.
- **Profile (L4, per frame):** parent head 67ms = 35% of ~190ms step; our head
  **8ms = 6%** of ~129ms step; everything-else ~122ms (LM + 340M acoustic
  feedback encoder + semantic encoder + loop overhead + CFG negative stream).
  End-to-end speedup from head swap alone: **1.47×** (Amdahl-capped; matches
  measurement). cfg_scale=1.0 did NOT disable the negative stream (identical
  per-frame cost) — real removal needs a deeper patch. Consequences: README's
  15–20× e2e target needs honest revision (head-cost ×8.4 + a bottleneck map
  is the defensible claim); P2's e2e wall-clock value is ~6ms/frame — its worth
  is the scientific claim + first-packet latency, not RTF; the RTF levers are
  CFG-stream removal, P4 encoder distillation, and batching (5×, built).
- **Drift diagnosis:** NOT progressive variance collapse — our latents are
  born ~35% under-dispersed (std ≈0.65 vs teacher ≈1.0, stable across the
  sequence). Mean-regression signature of an underfit few-NFE head; the fade
  emerges from the loop reacting to persistently muted latents.
- **Variance-calibration band-aid: FAILED, informatively.** Global ×1.79
  rescaling made it worse (rapid gibberish → loud digital pulsing): it
  amplified the signal content, not the missing spread — off-manifold latents.
  Conclusion: the missing 35% is unlearned FINE STRUCTURE, not amplitude;
  no linear inference-time correction exists. The treatments are (1) train
  better — full run, more data + steps, shrink under-dispersion at the source —
  and (2) if drift persists after that, on-policy distillation: drive the loop
  with the flow head while computing teacher targets from the same conditions
  (DAgger-style), which attacks exposure bias directly. Synergy note: the
  energy/quality-over-time curves needed to evaluate this ARE the C4 benchmark
  machinery — build once, use for both.

## E1 — teacher-forced dispersion audit (2026-07-08, decisive)

228 held-out frames, teacher std in normalized space 0.937:

| sampler | marginal std | per-cond spread |
|---|---|---|
| euler4 sway−1 (shipped) | 0.798 | 0.420 |
| euler4 sway0 | 0.824 | 0.462 |
| euler16 sway0 | 0.906 | 0.584 |
| euler64 sway0 | 0.932 | 0.619 |
| **heun8 sway0** | **0.941** | 0.631 |

**Verdict: the sampler was the thief — reviewer-2's perfect-field simulation
confirmed almost to the decimal.** Heun-8 (16 evals, ~16ms/frame, still 4×
cheaper than parent head) matches teacher dispersion exactly; the field is NOT
collapsed; training/capacity acquitted for the dispersion deficit. Remaining
gap between teacher-forced 0.85 ratio (shipped sampler) and the 0.56 measured
in-loop = the drift amplification mechanism (conditions leave distribution).
**E1b — closed loop with heun8 (2026-07-08): THE FADE IS DEAD.** 34.7s audio
for the long paragraph (parent ~39s; yesterday's runaway: 64s), clean
termination, drift curve 1.03→1.21 around teacher's ~1.16 (yesterday: flat
0.65). Head cost 30ms/frame (vs parent 67; NFE/solver tradeoff tunable).
Josh's ears: "definitely gone... full 34 second duration." Residual quality
gap unchanged and known ("digital cold" = the underfit stepping/ghosting
texture, 75–85% of teacher) — that's the polish problem, owned by training
scale/recipe, no longer a survival problem. Watch item: mild upward dispersion
creep (1.21 by the final quarter) — re-check on 5–10 min generations.

**P1 blocker resolved at ~$0 training cost.** The 75K decision is now purely
about closing the quality gap: order of attack per reviews = E3 (steps
continuation + lr tail on existing 10K, one overnight) → capacity probe →
75K with the upgraded schema (teacher noise + neg_condition + cfg_scale for
paired map distillation) only if still short.

## Pre-75K review round 2 (2026-07-08, post-E3)

Three clean-context reviewers (code audit / design gaps / data strategy) —
full distillation + merged plan: **`review-pre75k-2.md`**. Headlines: no code
bug pollutes existing results (audit fixes landed same day: shape guards,
patch-nesting restoration, standardization-proof tests); ~7–9 GPU-h of
BLOCKING gates before the big cache (teacher determinism + teacher-reseed
metric floor → 1K gate on the NEW recipe → val machinery + retro-validation →
scaling curve from cache subsets → context-length OOD spec); capture mix
rebudgeted by FRAMES: 30% short read / 30% long monologue (2–20 min) / 25%
dialogue / 15% conversational (75K-from-clean.360-as-written would reproduce
the OOD failure). cfg_scale covariate flagged (cache 1.3 vs shipped default
3.0 — decide and be consistent).

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
- **P3 idea (Josh, 2026-07-08): windowed/anchored context for long generation** —
  StreamingLLM-style attention sink: keep the sequence START (which in VibeVoice
  IS the voice-reference prompt → identity stays permanently in view) + a
  recent sliding window; evict the drifting middle history. Potential double
  win: stability (drift can't accumulate through evicted context) AND speed
  (attention stops growing with length). Test in P3 as a C4-measured arm:
  anchoring (C3) vs windowed context vs both. Caveat to measure: long-range
  prosody/discourse coherence may need the window not-too-small.

## Endurance test (2026-07-08, first C4-style measurement)

- 1,668-word script (~10 min expected) → **5.3 min audio, early termination**;
  365s wall. Drift curve over 8 segments:
  [1.099, 1.193, 1.185, 1.249, 1.327, 1.384, 1.542, 1.572] — steady ~linear
  UPWARD accumulation (over-dispersion), opposite sign from the cured fade,
  milder dynamics (no collapse), but past 1.4 by segment 7 and likely audible;
  early stop consistent with loop-state wander.
- Current stability envelope: clean ≈1–2 min; degraded by ~5; 10 min not yet.
- Treatments in order: polish training (sharper field → less per-frame bias to
  compound); sampler balance probe (heun8 runs slightly hot teacher-forced,
  euler16 slightly cold → possible loop-neutral setting between); P3
  anchoring/windowed-context (see follow-up idea above) as the structural cure.
- Pending listens: late-clip harshness vs start; full-script coverage vs cutoff.
- Listening (Josh) on the 5.3-min endurance clip: **no fade — instead it slowly
  sped up and got louder, less intelligible near the end but still
  understandable.** Matches the hot-drift curve exactly; also reframes the
  "early termination": the clip likely COMPRESSED the full script by
  accelerating (10 min of words in 5.3), rather than truncating. Runaway in
  the energetic direction, milder dynamics than the old fade.
## E3 — polish run (2026-07-08/09 overnight, training DONE, eval pending)

- Warm start from the 20K ckpt (fresh Adam), 60K further steps, bs 1024, cosine
  lr tail 2e-4 → 2e-5, on the existing 10K cache (478,191 pairs).
  **Loss 0.975 → 0.693**, smooth, no NaN; end slope ~0.0015/1K with the tail
  mostly responsible for the flattening. Saved `full10k_80k.pt` (Drive).
- Read: the 20K loss (~0.965) sat almost exactly at the reviewer's Gaussian-toy
  bound for σ_cond ≈ 0.61 — the field had learned little beyond conditional
  means + noise. 0.693 is well below that bound: the head now captures real
  conditional fine structure it previously left as "irreducible" noise. This is
  the quantity that owns the "digital cold" doubling/ghosting texture.
- Caveats: 60K × 1024 ≈ +128 epochs on 478K pairs → memorization watch; train
  loss can't adjudicate. Multi-t (LatentLM ~4 t-draws/condition) was NOT in
  this run — still unimplemented; bank for the next continuation or the 75K.
- **Eval protocol: `eval80k_colab.ipynb`** (built 2026-07-09) — (1) fresh
  multi-speaker held-out capture, 20 speakers from test.clean (fixes the n=5
  single-speaker caveat); (2) E1-protocol dispersion re-audit 80K vs 20K;
  (3) parity A/B renders for ears + Mac WER/ECAPA; (4) E1b closed-loop
  replication; (5) endurance re-baseline on the same 1,668-word script,
  8-segment curve vs [1.10 … 1.57]. The endurance slope after polish is the
  baseline the thermostat probe is queued against.

### E3 verdict: FAIL — steps-scaling on 10K data overfits (2026-07-08, decisive)

Mac analysis of `eval80k_bundle` (Whisper large-v3 + ECAPA, 6 held-out
speakers; artifacts in the eval bundle + `analysis.json`):

| held-out (teacher-forced) | teacher | 20K ckpt | 80K ckpt |
|---|---|---|---|
| mean WER vs text | 0.051 | **0.088** | 0.209 |
| mean ECAPA sim to teacher | — | **0.743** | 0.716 (1/6 wins) |

- **The 80K head is WORSE than 20K on held-out despite train loss 0.965→0.693.**
  ~128 additional epochs on the same 478K pairs memorized the training set.
  The dispersion table's per-cond spread shrink (0.714→0.509) was misread
  in-session as "learned fine structure" — it is actually **overconfidence on
  held-out conditions** (sharper conditionals, worse placed). Same
  self-persuasion trap as N2's mid-experiment retraction; logged accordingly.
- **Endurance run (9.8 min audio, hit the 4410-frame cap): collapse FROM FRAME
  ZERO, not stability.** The flat std≈2.0 curve is a stable NOISE attractor.
  Josh's ears (tier-3): zero intelligible speech from the very start — the
  "75 words" my segment-1 Whisper pass found were hallucinated too (61 wpm,
  sim-to-prompt 0.05; segments 2–8 hallucinate "Thank you" loops, voiced
  fraction 0.22–0.34). Metrics missed it, a human caught it — April 7 lesson
  re-confirmed. Closed-loop 35s paragraph: WER 0.222 (vs 20K near-parity at
  E1b).
- **Why parity clips work, 35s closed loop degrades, endurance is instant
  garbage (same ckpt):** conditioning distance. Parity = cached teacher states
  (training distribution). Short closed loop = self-generated conditions from
  a short prompt (near-distribution). Endurance = the full 1,666-word script
  in the LM context BEFORE frame one — hidden states from a text prefix ~10×
  longer than any training sample, OOD from the first frame. The overfit 80K
  head maps unfamiliar conditions straight off-manifold; the 20K head on the
  SAME script (2026-07-08) degraded gracefully for 5+ min. Overfitting cost
  robustness-to-OOD-conditions — the property a closed loop lives or dies by.
  Corollaries: long-context capture is near-mandatory in the 75K schema
  (else 90-min conditions are permanently OOD), and P3 windowed context gains
  a second mechanism (keeps conditioning stats inside the trained regime).
- **Conclusions:** (1) steps alone on 10K data are ruled out — past ~20K steps
  the head trades generalization for memorization; the binding constraint is
  DATA, exactly the pre-registered 75K trigger ("only if still short" — we are
  short and steps are eliminated); (2) **20K remains the operating checkpoint**
  for all in-loop work; (3) intermediate checkpoints (every 5–10K, per the
  review's dispersion-vs-steps ask) were not saved — any steps sweet spot
  between 20K–80K is unmeasured; save them in ALL future runs; (4) the
  segment-wise quality-over-time pipeline (words/min, RMS, F0, ECAPA drift
  per segment) now exists and worked — this IS the C4 machinery, first real
  exercise; (5) 75K schema requirements stand (teacher noise + neg_condition +
  cfg_scale for paired-map distillation) and long-context/dialogue capture is
  now motivated by drift too, not just turn coverage (drift onset ~30–90s ≈
  where conditions exit the short-utterance training distribution).

- **Queued probe: statistical anchoring ("thermostat")** — inference-time
  feedback controller on latent statistics: track running std over recent
  frames; when it creeps past anchor stats (from the cache), scale residuals
  back a few % per frame. Gentle closed-loop correction, NOT the failed static
  x1.79 gain. If a dumb controller flattens the drift curve, drift is
  controllable at inference — P3 preview. Test AFTER the polish run re-baselines
  the slope. P3 arm order: polish -> thermostat -> C3 speaker anchor ->
  windowed context; all judged on C4 curves.
