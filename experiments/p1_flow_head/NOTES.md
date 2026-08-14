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

## 2026-08-10 — project review session (no new experiments)

First session after ~1 month dormant. Full audit + field re-check + first-
principles re-derivation of the approach: **`docs/review-2026-08-10.md`**.
No experimental results to log; state entering the review = state at the
2026-07-09 "Gate Night 1 notebook" commit (notebook built, never run).

Decisions affecting this experiment line:
- **Next run is unchanged: Gate Night 1** (stats equality, teacher
  determinism, reseed floor, heun8-vs-euler4 held-out A/B) **+ the data-
  scaling curve** (2.5K/5K subsets of the existing cache). ~5 GPU-h combined.
  The curve is now also the **venue scope gate**: 5K≈10K within the reseed
  noise band → compact ICASSP 2027 paper (deadline 2026-09-16); data still
  binding → skip ICASSP, pace toward Interspeech 2027 (see review §5).
- Capacity probe scope widened: width-960 AND a shallow local-attention
  variant (2–4 frames) — conditioning is acquitted, so constraint 5's
  rationale no longer bars it (review §3.4).
- Held-out design rule added: stratify by **context length** as well as
  speaker (E3 showed prefix-length OOD is the dominant failure axis; the
  current eval would not catch a long-prefix regression).
- On-policy ("DAgger-style") follow-up formally identified as **Self Forcing**
  (Huang et al. 2025) / Causal-rCM (2606.25473) — import the recipe rather
  than invent it; windowed-context idea = DySink (2605.21028) family.
  Recipe-alignment table: review §4.
- README/CLAUDE.md stale claims (15–20× e2e, Interspeech 2026, C2-as-live,
  private-repo rule) struck through with dated corrections in the same commit.
- ~~**Speed-claim sampler discipline (added 2026-08-11).** The 8.4× head-level /
  1.47× e2e pair (line ~141 above) is **euler4**, which E1/E1b showed is
  dispersion-incorrect. The shippable sampler is heun8: head ~30ms/frame →
  **2.2× head-level, ~1.24× e2e**. Always name the sampler next to the number;
  the heun8 pair is the headline.~~ **[RETRACTED same day by Gate Night 1 cell 5
  — heun8 costs 3× WER and is not shippable. See the Gate Night 1 entry below;
  the headline-sampler question is OPEN.]** The surviving half: always name the
  sampler next to the number, and **P2 MeanFlow is load-bearing** — it is the
  only route to a large head-level number at correct dispersion.

## 2026-08-11 — GATE NIGHT 1 RESULTS (Colab L4 generate; Mac-side scoring)

Notebook `gate_night1_colab.ipynb`, built 2026-07-09, finally run. ~$2, 1122 s
wall for the endurance cell alone. Bundle `gate_night1_bundle.zip` (33 files)
extracted to `audio/gate_night1/` (git-ignored). Machine-readable results:
`gate_night1_metrics.json`, `endurance_episode_map.json`,
`endurance_transcript.json`.

| cell | check | result | verdict |
|---|---|---|---|
| 2 | 20K/80K standardization stats equality | mean & std maxdiff **0.0** | **PASS** — E3 dispersion table was a fair comparison; overfit verdict stands |
| 3 | teacher determinism given RNG state | same seed 0.0 / diff seed 2.81 | **PASS** — paired noise→latent 75K schema is viable |
| 4 | teacher-vs-teacher reseed floor | mean 0.076 WER, **median 0.000** | **UNUSABLE at n=6** — recalibrate before the scaling curve |
| 5 | sampler A/B on the 20K head | heun8 0.238 / euler4 0.079 / euler16 0.081 | **BLOCKING** — no sampler is clean; see below |
| 6 | batched-vs-unbatched condition parity | mean shift 0.20–0.50σ, frame counts diverge | **BLOCKING** — do not cache 75K at batch-8 until resolved |
| 7 | 20-min teacher endurance | rate inflation 1.5×; voice sim decline in back half | **NEW FINDING N8** + harvest-cap constraint |

### Cell 5 — the sampler dilemma (blocking, and it reframes the speed story)

Answers the question NOTES never recorded: **0.088 was euler4** (0.079 measured
here on 6 held-out utterances). But the A/B exposes a trap:

- **euler4** — WER 0.079, intelligible; under-dispersed (std ≈0.65 vs teacher
  ≈1.0), and that under-dispersion is the closed-loop fade of E1/E1b.
- **heun8** — matches teacher dispersion and cures the fade, but WER **0.238**,
  worse in **5 of 6** utterances (median 0.235). Not fuzzy edges — semantic
  corruption: *"And what was the subject of the poem?"* → *"a notable subject of
  Moab."*
- **euler16** — 0.081, no better than euler4 for 4× the NFE.

**There is no sampler setting that is both intelligible and dispersion-correct.**
This is not a sampler-selection problem; it is the underfit velocity field
surfacing through whichever error the integrator accumulates. Consistent with
the NFE-16 A/B (16≈4) and the "digital cold" texture — all three point at
underfit, not integration error.

Consequence: **P2 MeanFlow is load-bearing for correctness, not merely speed**
— a stronger claim than the 2026-08-10 amendment 1 made. The headline-sampler
question is **open**: we cannot quote heun8 (unintelligible) or euler4
(under-dispersed, fades in closed loop) as "the shipped configuration". Resolve
by fixing the head, not by picking a sampler.

Speaker sim vs teacher was ~0.575–0.596 for all three samplers, against a
teacher-vs-itself ECAPA of **0.734** — see cell 4 caveat before reading anything
into that gap.

### Cell 4 — the floor is bimodal and unusable at n=6

Mean pair-WER 0.076 matches the review's "~0.07 → 20K is at the ceiling"
prediction, but the **median is 0.000**: 4 of 6 utterances transcribe
identically across seeds. The mean is carried by one utterance (121, pair-WER
0.400) whose text is dense with proper nouns (*"Mother Eddy and Brother
Dowie"*) that Whisper mangles — so part of the "floor" is ASR unreliability,
not generation variance. Teacher-vs-itself ECAPA is only 0.734 on clips this
short, so the similarity floor is noisy too.

**The noise band is not a smooth band — it is usually zero with occasional
catastrophe.** The data-scaling curve cannot be judged against it, and the curve
is the venue scope gate. **Blocking action: rerun the reseed floor at n=30–50,
report median and IQR (not mean), and either exclude or separately bucket
proper-noun-heavy texts.** Teacher-only, no training, cheap.

### Cell 6 — batched capture parity (blocking)

Per-dim mean shift 0.20–0.50 with std ratio 0.97–1.14: spread preserved, mean
moved. Worse, frame counts diverge (1995: **124 unbatched vs 53 batched**);
6 of 8 shorter when batched, which is not the symmetric pattern pure stochastic
divergence would give. The one equal-length case has per-frame cosine 0.288.
The check cannot separate "left-padding pollutes conditioning" from "generation
diverged" by construction — but this is the pollute-everything signature the
cell was built to catch, for $0.30. **Do not capture the 75K cache at batch-8
left-padded until this is resolved** (suggested: fixed-seed batched-vs-unbatched
on identical conditions, or capture unbatched and eat the 4.9× cost).

### Cell 7 — endurance: N8 rate inflation + a real back-half voice decline

Full write-up as **finding N8** in `docs/negative-results.md`. Headline: the
teacher renders a 3229-word script complete (3365 Whisper words) in 13.35 min at
**252 wpm** vs **168 wpm** on short clips — **1.50× rate inflation**, present
from the first 30 s, saturating by ~min 2. It stopped at 6009/12000 tokens on its
own, so this is a pacing defect, not truncation or a lost-place failure.

Separately and genuinely progressive: per-4s-window ECAPA vs the run's median
identity declines in the back half — segment means 0.759 / 0.801 / 0.833 / 0.846
/ **0.852** (min 6:40–8:20) → 0.797 → 0.775 → **0.735**, worst-window 0.612.
This tracks the latent-std climb (1.224 → 1.58 over the same span). Four of the
five flagged voice episodes fall in the final 100 s. One early outlier at
**0:54** (sim 0.573) is the largest single anomaly in the run and the
segment-averaged std curve rates that segment as normal — **latent std is the
wrong detector for episodic defects; use per-window ECAPA + rate.** Relevant
because latent std was the candidate automated quality filter for the 75K cache.

**Harvest-cap constraint:** long captures are usable for *long-context
conditioning* research but carry N8's pacing defect; they are not clean
prosody targets. Josh listening (13.35 min, full): "overall sounds pretty
good", voice changes noted ~min 1, 8, 9 that recover — consistent with the
0:54 episode and the seg-5→6 transition (8:20) where the decline begins.

**0:54 confirmed by ear (Josh, 2026-08-11): "it sounds like an anomaly."**
Two independent detectors agree on it — Josh unprompted during full-length
listening, and per-window ECAPA (sim 0.573, z−3.9) — while the segment-averaged
latent std rates that segment (1.137) as the *cleanest* in the run. That is the
decisive evidence for **replacing latent std with per-window ECAPA + speaking
rate as the 75K cache quality filter** (blocking item 3): the std filter would
have passed the single worst audible defect in 13 minutes of teacher audio.

### Retractions from this session (kept visible per the strike convention)

Three claims made and withdrawn on 2026-08-11 while analyzing this bundle:

1. ~~heun8 is the shippable/headline sampler~~ — cell 5: 3× WER. Retracted in
   README, CLAUDE.md, review §3.1 and above.
2. ~~The teacher dropped ~1/3 of the script~~ — transcript is 3365 words vs a
   3229-word script; coverage complete. The 13.35-vs-20 min gap is entirely the
   1.5× rate.
3. ~~The teacher lost its place / failed text-audio alignment~~ — it completed
   the script and terminated correctly at 6009/12000 tokens.

Also **withdrawn: the 9 "speed-up" episode timestamps** in
`endurance_episode_map.json` (1:22, 2:00, 2:22, 2:38, 4:54, 6:20, 7:58, 8:12,
10:48). The onset-rate proxy measured articulation density, not speaking rate —
local wpm at those points is at or near the 252 average, while the one real rate
outlier (**350 wpm at 3:00**) went unflagged. The `voice` flags (ECAPA-based)
stand; the `fast` flags do not. Treat that file's fast column as invalid.

### Blocking list before any 75K spend

1. Reseed floor at n=30–50, median+IQR (gates the scaling curve → gates venue).
2. Batched-capture parity resolved, or capture unbatched.
3. Cache quality filter switched from latent std to per-window ECAPA + rate.
4. Rate-correction decision for long-context captures (N8).

Not blocking but queued: rate-vs-script-length sweep (N8 caveat: n=1 script,
1 prompt, 1 CFG scale), and the capacity probe (width-960 + shallow
local-attention 2–4 frames).

## 2026-08-11 — GATE NIGHT 2 PRE-REGISTRATION (teacher-only; not yet run)

Notebook: `gate_night2_colab.ipynb`. Mac scorer: `score_gate_night2.py`.
Teacher-only — touches nothing on the 75K blocking list; runs in parallel with
it. L4, ~90–120 min, ~$2–3. Motivated by N8 and the windowed-context promotion
(review §3.2): before building KV-cache eviction, establish (a) that upcoming-
text length is the pacing driver and (b) whether naive text windowing already
cures it.

### Hypotheses

- **H1 (dose-response):** VibeVoice's speaking rate is driven by the length of
  the *visible upcoming script*, not rollout history. Design: nested prefix
  scripts (100/500/1500/~3229 words) sharing the identical first ~100 words,
  2 voice prompts, seed=0, cfg 1.3 — rate measured on the shared segment, so
  text ahead is the only variable. (N8 predicts this: inflation present in the
  first 30 s, before any audio accumulates.)
- **H2 (text windowing cures pacing):** rendering the same full script in
  ~320-word chunks (same original voice prompt each chunk, identity held
  constant) restores near-natural rate. Seam identity cost is measured, not
  assumed away.

### Gate criteria — written before the run

| Verdict | Condition |
|---|---|
| H1 CONFIRMED | shared-segment wpm(full)/wpm(100) ≥ 1.2 on both prompts |
| H1 REFUTED | ratio ≤ 1.05 on both prompts → pacing driver is elsewhere (cfg? sampler steps?); windowed context loses its N8 justification but keeps the drift one |
| H1 AMBIGUOUS | mixed/between → add prompts before concluding |
| H2 pacing CURED | chunked wpm ≤ 1.10× standalone-500 AND ≤ 0.85× full-script |
| H2 seams ACCEPTABLE | mean seam ECAPA ≥ within-chunk adjacent-window baseline − 0.10 |
| H2 seams FAIL | below that → naive chunking unusable; proceed to KV-cache sink+window (position re-indexing work) |

Automated checks: `score_gate_night2.py` (shared-segment wpm via Whisper word
timestamps; seam-vs-within-chunk ECAPA). Listening step (never skipped): Josh
listens to `chunked_full.wav` for seam audibility and one full sweep clip
against the Gate Night 1 endurance render.

Also banked by this run for free: seeded replication of the N8 full-script
condition (N8 was unseeded, n=1); teacher hidden-state captures at 4 context
lengths × 2 prompts → E4 context-length-OOD probe conditions; N8's robustness
caveat (1 prompt → 2 prompts) partially discharged.

### Results (2026-08-11 — run and scored same day; ~$3, L4, ~100 min)

Whisper-scored (`gate_night2_metrics.json`); raw report in
`audio/gate_night2/gate_night2_report.json`.

- **Sweep, whole-clip wpm** (119/514/1500/3229 words): p0 199.5 / 234.5 /
  241.3 / 230.7; p1 166.7 / 190.2 / 238.3 / 233.3. Monotone-to-saturation on
  both prompts; ceiling ~230–240 by 1500 words. Onset is prompt-dependent:
  p0 is already elevated at 119 words (~200 vs its ~158 natural), p1 still
  natural at 119 (167 vs ~164).
- **H1 on the registered metric (shared first 119 words):** full/100 ratios
  **1.03 (p0), 1.06 (p1)** → **AMBIGUOUS** per pre-registration. The criteria
  were mis-designed, said plainly per the template's escape clause rather than
  bent: whole-clip ≫ shared-opening in every long condition, i.e. **the
  inflation ramps with position over the first ~1–2 min of a render** — the
  shared opening is the *least*-affected segment, so the registered isolator
  measured where the effect is smallest. The dose-response itself is
  unambiguous in whole-clip rates and replicated that of N8.
- **H2 pacing: NOT CURED.** chunked_full **237.1 wpm** vs standalone-500 234.5
  vs full 230.7 (cure required ≤196.1). Consistent with the dose curve — a
  single ~320-word chunk from a cold start renders at 237; practical chunk
  sizes are already fully inside the fast register.
- **H2 seams: metric-fail, ears-pass.** Seam ECAPA **0.456** vs within-chunk
  baseline 0.72 → formally NEEDS KV-CACHE VERSION. Josh listened *before*
  seeing the metric: "the stitch is barely noticeable." Dissociation noted:
  the seam windows compare end-of-chunk wind-down prosody against fresh-onset
  prosody (3 s windows; cf. GN1's 0.734 teacher-self-sim floor on short
  clips), so the embedding gap is likely mostly prosodic, not identity. Ears
  are the tier-3 authority: treat as **PARTIAL — listenable, metrically
  flagged**.
- **Prior art found the same day** (full citations in N8): the *symptom* is in
  Microsoft's own docs ("if the generated voice speaks too fast, try
  chunking...") and open issue microsoft/VibeVoice#85; the dominant ComfyUI
  wrapper auto-chunks >250 words by default "to prevent audio acceleration
  issues". No quantification exists anywhere. GN2 is the first measurement —
  and it shows the folk remedy (which is also the official tip and the
  wrapper's silent default) does **not** restore natural rate.

### Verdict

**H1: AMBIGUOUS on the registered metric; the dose-response is nevertheless
established** by whole-clip rates on both prompts (criteria flaw acknowledged,
not bent — the opening-segment design assumed a constant rate offset; the data
show an early positional ramp instead). **H2: FAIL on pacing, PARTIAL on
seams.** Consequences:

1. **The pacing fix is now an open research question.** Text windowing at any
   practical size does not work; the fast register triggers at low hundreds of
   words and ramps within the first minutes. Candidate directions (unscheduled):
   rate/duration conditioning, prompt-side pacing exemplars, CFG-scale arm.
   **Found in the wild (Josh, 2026-08-11):** VibeVoice-ComfyUI v1.5.0
   (discussion #142) ships `voice_speed_factor` — **time-stretching the
   reference audio steers output rate** (pitch-preserved; ±20% range, ±5%
   recommended; voice-cloning only; unmeasured, zero replies). Implies the
   model partly mimics the prompt's pace → direct support for the prompt-side
   direction. Cheap probe (Gate Night 3 candidate): 0.8×-stretched prompt on
   the full script, measure output wpm + ECAPA identity cost. Note the gap:
   canceling the ~1.4× register needs ~0.7×, well past their recommended
   band — the question is whether prompt-rate transfer holds on *long*
   scripts at all, and how much identity it costs.
   **Two further candidates (2026-08-11 discussion):**
   (a) **Rate steering via the retained `src/steering/` machinery.** N7 killed
   affect steering because the backbone lacks affect range; but the backbone
   demonstrably HAS rate range (165→240 wpm, GN2-measured), so the ceiling
   argument does not apply. Extract fast-vs-slow contrast pairs (same text,
   register forced via script length — we now know how to trigger both
   registers on demand), find the rate direction, steer negatively during long
   renders. Training-free; a principled fix, not a prompt bandaid; would also
   resurrect the C2 machinery as a live contribution.
   (b) **7B replication arm.** All N8/GN2 evidence is VibeVoice-1.5B; the 7B
   is untested (and docs recommend it "for stability"). Same sweep on 7B =
   cheap GN3 arm; "defect scales with size" and "defect vanishes at 7B" are
   both paper sentences. Also explains why Josh has not heard it on his 7B
   only if his tooling chunks silently — check how the 7B is being invoked.
   **Mechanism hypothesis (unproven, consistent with all GN2 data):** pace is
   an unsupervised free variable (no duration model, no loss penalty); long
   text prefixes match the long-form/podcast register in training data
   (~200–240 wpm is normal podcast pace, hence the ceiling); the early ramp is
   self-reinforcing prompt-pace mimicry — generated audio re-enters the
   context and the model matches its own accelerating pace (same
   exposure-bias family as E1/E3; and the mimicry channel is exactly what
   voice_speed_factor exploits).
   **(c) The untested third condition — turn-splitting within ONE call
   (2026-08-11, from Josh's conference-generator app).** GN2 tested monolithic
   blocks (sweep) and separate calls (chunked arm). Microsoft's official
   remedy is neither: many short `Speaker N:` turns inside a single call.
   Josh's own app has run exactly that shape for months (e.g. 1,097 words as
   18 turns averaging 58 words, one generate call, 1.5B AND 7B) with no
   audible rushing — informal evidence the remedy works even though the full
   script sits in context. If a measured test confirms it, the mechanism
   refines from "total-script-length register" to **per-turn pace scoping**,
   which would also mean multi-speaker dialogue is naturally immune and N8
   bites hardest on single-speaker long-form narration. GN3 cell: the same
   3,229 words in one call split as ~60-word same-speaker turns, measure wpm.
   Also explains "never heard it on the 7B": the app never hands either model
   the trigger condition — usage was accidentally self-immunizing.
2. **Windowed context keeps its promotion (review §3.2) but on narrower
   grounds:** back-half drift mitigation + bounding conditioning stats — not
   pacing.
3. **N8 upgraded:** prior-art citations + folk-remedy refutation; claim is now
   "first quantification of a community-worked-around defect, and the standard
   workaround does not restore natural rate."
4. Follow-up for H1 disambiguation (cheap, queued): ≥3 prompts + position-
   resolved rate curves (wpm vs position within each condition) — the scorer
   already extracts word timestamps, so this is analysis, not new generation.

## 2026-08-11 — GATE NIGHT 3 PRE-REGISTRATION (teacher-only; not yet run)

Notebook: `gate_night3_colab.ipynb`. Mac scorer: `score_gate_night3.py`.
Teacher-only, parallel to the 75K blocking list. L4 (~$3–5, ~2–2.5 h); the 7B
cell may need A100. Baselines reused from GN2 (same stack, seeded): natural
165–169 wpm, monolithic-full 230.7/233.3, 119w 199.5/166.7, 1500w 241.3.
Rate-steering (candidate (a)) deliberately deferred — needs its own build.

### Hypotheses and gate criteria — written before the run

- **T1 — turn-split within one call** (Microsoft's actual remedy; the shape
  Josh's conference app runs daily without audible rushing). Same 3229 words,
  one generate call, ~60-word `Speaker 1:` turns, both prompts.
  | Verdict | Condition |
  |---|---|
  | CURED | wpm ≤ 185 on both prompts → pace is **per-turn scoped**; N8 localizes to single-speaker long-form narration; multi-speaker dialogue naturally immune |
  | PARTIAL | 185–215 |
  | FAIL | > 215 → the official tip is as unverified-and-wrong as the wrapper default; mechanism stays total-length |
- **T2 — stretched-prompt transfer** (ComfyUI v1.5.0 `voice_speed_factor`
  trick, measured at long context). Prompt time-stretched 0.8×/0.9×
  (pitch-preserved), monolithic full script.
  | Verdict | Condition |
  |---|---|
  | TRANSFERS | monotone wpm(0.8×) < wpm(0.9×) < 230.7 AND wpm(0.8×) ≤ 208 |
  | WEAK/PARTIAL | monotone but shallower |
  | NO TRANSFER | not monotone |
  Identity cost reported (whole-clip ECAPA vs the GN2 unstretched render),
  judged by ear as the authority (GN1/GN2 lesson: short-window ECAPA
  over-flags prosody).
- **T3 — 7B replication** (mirror weights `vibevoice/VibeVoice-7B`; all
  N8/GN2 evidence is 1.5B; Josh's null experience is confounded by his app's
  turn-split shape). 119w + 1500w monolithic, prompt p0.
  | Verdict | Condition |
  |---|---|
  | DEFECT PRESENT | wpm(1500)/wpm(119) ≥ 1.2 |
  | DEFECT ABSENT | ≤ 1.05 → N8 is 1.5B-specific; hunt for what 7B does differently |
  | AMBIGUOUS | between |

Listening step (never skipped): `t1_turnsplit_p0.wav` (naturalness, per-turn
seam artifacts) and `t2_stretch80.wav` (does the voice still sound like p0?).

### Results — PROVISIONAL from the run log (2026-08-12; audio lost to a runtime
### disconnect before bundling — see incident note)

**Incident:** Josh's machine slept after T2; the Colab runtime recycled before
the bundle cell ran. GN3 v1 did not mirror to Drive (GN1/GN2 saved captures to
Drive; GN3 was built lean — a mistake). The printed run log survives and words
÷ audio_s gives the primary metric directly (script word count known = 3229;
coverage assumption per N8, unverified until the rerun). Notebook patched
same day: every wav now mirrors to Drive immediately + runs are resume-safe
(re-running skips anything already on Drive). T3 (7B) never ran.

| run | audio_s | wpm (3229 words) | pre-registered verdict |
|---|---|---|---|
| t1_turnsplit_p0 | 1176.8 | **164.6** | **CURED** (bar ≤185) |
| t1_turnsplit_p1 | 1176.8 | **164.6** | **CURED** |
| t2_stretch80 | 952.1 | **203.5** | ~~≤208 ✓~~ did not replicate — see below |
| t2_stretch90 | 843.3 | **229.7** | ~~monotone ✓ → TRANSFERS~~ overturned |

**OFFICIAL (rerun bundle, Whisper-scored 2026-08-12):** T1 **CURED** — 176.7
wpm (p0, 3391 words = full coverage) / 159.9 (p1, 3135 = 97%, consistent with
the 9:40 babble); T2 **NO TRANSFER** — 228.1 / 226.8 ≈ baseline 230.7; T3
**DEFECT PRESENT AT 7B** — 189.6 (119w) → 248.4 wpm (1500w), ratio **1.31**.
Machine-readable: `gate_night3_metrics.json`.

- **T1 CURED, provisionally — the headline.** The same 3229 words that render
  monolithically at 230.7/233.3 wpm come out at **exactly natural rate
  (164.6)** on both prompts when split into ~60-word same-speaker turns in ONE
  call. Microsoft's un-verified official tip works; **pace is per-turn
  scoped**, not total-script; N8 localizes to single-speaker continuous
  narration (the exact use case long-form TTS exists for), and multi-speaker
  dialogue is naturally immune — explaining why nobody, Microsoft included,
  ever chased the defect: their flagship demo format hides it.
  Curious detail: both prompts produced *identical* durations (1176.8 s) —
  consistent with seeded, text-driven termination.
- ~~**T2 TRANSFERS, sub-linearly:** 0.8× prompt → 0.88× output rate; 0.9× →
  ~no effect.~~ **[OVERTURNED by the official rerun + Whisper scoring,
  2026-08-12: T2 = NO TRANSFER.]** Rerun renders: stretch80 **228.1 wpm**,
  stretch90 **226.8** — statistically the baseline (230.7). The airport log's
  apparent effect did not replicate (same nominal seed, different session);
  the prompt-pace channel is noise-dominated at long context. The
  `voice_speed_factor` lever is dead for our purposes (may still work on
  short clips — untested, not our problem). Turn-split stands alone as the
  cure, which is fine: it is complete on its own.
- **Pending the rerun:** Whisper coverage confirmation (did it speak all 3229
  words), Josh's listening steps (turn seams; stretch80 identity), and all of
  T3. Seeded determinism means the rerun reproduces last night's audio
  exactly, so provisional numbers should confirm bit-for-bit.

### The throughput reframe (2026-08-12, Josh)

Goal restated: **not streaming realtime — beat wall-clock on a 90-min batch
job** ("a 90-minute script in under 90 minutes"). Sequentially those are the
same line (wall < audio), but batch unlocks the parallel-segment lever:
- 90 min = 40,500 frames. Sequential: original ~190ms/frame → **128 min
  (FAILS the goal)**; our head ~129ms → **87 min (passes)**; +CFG removal
  ~69ms → ~47 min.
- **Parallel-segment generation:** split the script into ~8 segments, each
  internally turn-split (GN3 pacing cure), same voice prompt per segment
  (GN2 seams: ears-pass), generate as one batch (batch-8 ≈ 4.9× throughput,
  measured July). → **~18 min wall today; ~10 min with CFG removal ≈ 9–13×
  effective**, system-level and honest. Segmenting also dodges the >8-min
  drift zone and the KV-growth per-frame slowdown.
- **Load-bearing prerequisite:** blocking item 2 (batch parity) now gates the
  headline speed claim, not just caching. Plus a seam listen at 90-min scale.
- **Quant lever (queued behind GN4, 2026-08-12):** weight-only int8/int4 on
  the frozen backbone is the only lever touching the non-head 65% (sequential
  decode is bandwidth-bound → quant ≈ direct speedup; also KV-cache quant for
  the 90-min memory growth). Rule-#1-legal (compression, not training). Risk
  is OURS specifically: the flow head conditions on bf16-captured stats, and
  quant shifts hidden-state distributions — the N1/N2 failure axis in
  miniature. Test = GN1 cell-6 parity methodology verbatim (same seeds, quant
  vs bf16, conditioning stats + rendered WER/sim). Fallback if it hurts:
  re-capture the cache from the quantized backbone. Stacked projection with
  segments+batch+CFG removal: 90-min job in ~6–10 min.

### Feedback into the main line (2026-08-12)

1. **T1 likely RESOLVES blocking item 4** (rate-correction for long-context
   captures): capture long-context training data as **turn-split scripts** —
   natural pace by construction, no post-hoc correction, long context
   preserved. The N8 detour pays back into the 75K cache design. Confirm on
   the rerun audio (coverage + seams by ear), then mark item 4 closed.
2. **The 19.6-min T1 renders double as a drift probe at natural pace.** Run
   the per-window ECAPA episode map (GN1 tooling) on `t1_turnsplit_p0.wav`:
   does the back-half identity decline (0.85→0.73 after ~min 8, GN1) persist
   when pacing is cured? Distinguishes "drift is independent of the fast
   register" from "the two defects share a cause." Free analysis, no GPU.
   **Josh's bet (2026-08-12, pre-registered before the analysis): turn-split
   fixes likeness too** — reasoning: the model's long-form reputation rests
   on turn-shaped usage (conversations, chunking tools, his app), which never
   audibly drifted. If T1 audio holds flat where monolithic declined, the
   windowed-context mechanism demotes toward bonus/ablation.
   **VERDICT (same day, GN3 T1 audio, per-window ECAPA vs run median): BET
   CONFIRMED, both voices.** Position-matched against GN1's monolithic
   segments: where mono declined 0.852→0.797→0.775→0.735 (seg 5→8), turn-split
   holds 0.841→0.853→0.849→**0.860** (p0; final segment is the run's BEST) and
   0.825→0.790→0.813→0.817 (p1). Back half ≥ front half on both (0.848/0.839;
   0.811/0.806); final 3 min at run average; flat through 19.2/19.6 min — 6+
   min past mono's audible deterioration. **Pacing and drift were ONE disease:
   the monolithic operating mode.** Windowed context demotes to
   bonus/ablation; P3's KV-cache engineering is no longer load-bearing.
   Caveats: two isolated one-window dips need ears (p0 0:12 — early prompt
   settling, same as GN1; p1 **9:40**, sim 0.241 — likely a pause/silence
   ECAPA artifact, LISTEN to rule out a real glitch); claim bounded at ~20 min
   until a 90-min render exists. Curves: `t1_drift_curves.json`.
   **Listening verdicts (Josh, 2026-08-12):** overall turn-split audio "sounds
   good"; p1 9:40 is a REAL glitch — ~2 s of babble ("decided to speak in
   tongues"), which he has also seen occasionally in his HF space (turn-shaped
   usage) → a third, rare defect class: **transient babble**, stochastic,
   pre-existing, NOT turn-split-induced and not cured by it (~1 event / 39 min
   here). Detector scorecard: per-window ECAPA has now caught both confirmed
   audible defects (GN1 0:54, this) that latent-std missed — it is the cache
   quality filter, settled. Cache implication: filter must EXCISE babble
   windows or the head learns babble; C4 should report a glitch-rate-per-hour
   metric alongside drift curves.
3. **GN4 candidate — THE high-stakes cheap test: our 20K head under
   turn-split conditions.** Everything that condemned the head on long-form
   (E3 frame-zero collapse on long prompts, endurance fade) was measured under
   **monolithic** long prompts — which we now know flip the backbone into a
   register the training cache (short utterances, natural register) contains
   zero samples of. The head's long-form failures may be largely **register
   OOD, not context-length OOD**. Test: FlowHeadPatch + 20K ckpt, turn-split
   full script end-to-end, listen + score vs the July endurance collapse.
   One cell, ~$1–2. If the head holds up: long-form works TODAY with the
   current checkpoint, "context-length OOD" reframes as "register OOD", and
   the C4 story simplifies to an operating-mode finding. The short-clip
   underfit verdict (sampler dilemma) is untouched either way — that was
   measured in-distribution and still needs P2 + data.
   **Sequencing (agreed 2026-08-12): GN4 runs BEFORE any retraining spend.**
   It re-baselines the head under the corrected operating mode, and its
   outcome sets the 75K cache design: head holds → amendment 3's long-context
   bias partially unwinds (mostly short/medium clips + turn-split long
   captures — cheaper, already-proven pipeline); head collapses → register
   OOD refuted, long-biased mix and windowed context stay. The teacher's GN3
   T1 renders are the direct A/B reference (same script, seeds, conditions).

## 2026-08-12 — GATE NIGHT 4 PRE-REGISTRATION (the head's re-baseline; not yet run)

Notebook: `gate_night4_colab.ipynb`. Mac scorer: `score_gate_night4.py`.
L4, ~1.5 h, ~$2–3. The critical-path night: the 20K head's first-ever run in
the turn-split operating mode. Direct A/B reference: GN3's
`t1_turnsplit_p0.wav` (same script recipe, same prompt, teacher head).

### Arm A — 20K head via FlowHeadPatch, turn-split full script, euler4 + heun8

Two pre-registered bets collide:
- **Register OOD** (E3 reframe): the head's long-form collapse came from
  out-of-register hidden states, not context length per se. Turn-split keeps
  the backbone in the register the cache trained on → the head should hold.
- **Turn-reset** (new): E1's closed-loop fade accumulated over ~60 s of the
  head consuming its own latents; a 60-word turn is ~25 s, so each turn
  boundary re-anchors on text *before* the fade horizon → euler4 may survive
  turn-split despite fading monolithically.
Also expected: pacing should match the teacher's turn-split rate (~165–168
wpm) regardless of head — frame count is decided by the LM, not the head; a
big wpm deviation would itself be a finding.

| Verdict (per sampler) | Condition |
|---|---|
| PASS | Whisper coverage ≥95% of 3229 words AND no fade (back-half RMS ≥ front − 3 dB) AND identity flat (back-half ECAPA ≥ front − 0.05) |
| PARTIAL | intelligible but fades or drifts |
| FAIL | coverage <60% / E3-style collapse → register-OOD refuted; long-biased data mix and windowed context re-promote |

### Arm B — per-module latency profile (measurement, no gate)

Timer hooks on every top-level child of `model.model`, ~60 s generation,
teacher head vs flow head. Splits the ~122 ms non-head lump; becomes the
paper's measured bottleneck map.

### Arm C — batch parity at the audio level (blocking item 2)

4 held-out texts, solo vs one padded batch-4, teacher head, seed 0. Per-utt
solo-vs-batched WER/sim. PARITY OK if mean pair-WER ≤ 1.5× the GN1 reseed
floor (0.076 — noisy at n=6, noted); DEGRADED otherwise. Gates both the 4.9×
throughput claim and batch-8 cache capture.

Listening step (never skipped): euler4 render vs the GN3 teacher render,
same script same prompt — texture ("digital cold"?), fade, seam naturalness.

### Results — Arm A euler4 (partial; bundle + scorer pending)

Run vitals: 1184.0 s audio / 1144 s wall (**first faster-than-realtime render
in the project**), 8880 head calls at 7.7 ms = 6.0% of wall — July's profile
reproduced exactly. Full-length generation, no truncation.
latent_std_8seg [0.503, 0.767, 0.894, 1.019, 1.125, 0.823, 0.95, 1.262],
overall 0.943 — looks like "recovery to teacher dispersion" and is NOT: it is
the E3 stable-noise-attractor illusion again (second catch; the listening
rule pays again).

**Verdict by ear (Josh, 2026-08-12): FAIL — E3-style collapse, faster.**
Intelligible (grainy) speech only to ~**0:14** ("I have found the diary of
Mrs. Gilroy" — less than one turn), then, verbatim: silence for a long time,
then random noise "like an alien noise reverberating through a microphone,"
fading in and out "seemingly randomly with silence in between, sounding more
different as time goes on, like a helicopter coming in and out with a voice
in the wind... toward the last 20% really creepy louder alien noises but much
slower reverberation, to ultimately a swinging reverb with weird low hums and
random cat-meow-like sounds."

**Reading:** classic unstable-feedback resonance. First ~14 s ride the clean
voice-prompt acoustic context; as self-generated latents fill the context the
backbone leaves the speech manifold and the loop settles into quasi-periodic
attractor states (the in/out swells and "swinging reverb" = the loop's
ringing modes; the escalating std = energy accumulating in them).

**Diagnosis sharpened: the head's disease is feedback-OOD, not register-OOD
and not prompt-length.** Turn-split did not help and death was FASTER than
E1's ~60 s fade horizon (< one 25 s turn). Text re-anchoring cannot fix a
poisoned acoustic channel. Register-OOD (GN4 bet 1) and turn-reset (bet 2):
both REFUTED for the head.

**Consequence — the single most important redirect of the training budget:**
offline distillation provably cannot survive its own rollout regardless of
data scale; a naive 75K retrain of the same recipe would improve texture and
still collapse at ~second 20. **The on-policy stage (Self Forcing / DAgger —
queued in July, named via the video recipe in the August review) is
MANDATORY, not optional.** P2/75K planning must include self-generated
rollout context in the training distribution. The capture-wraps-patch
nesting (integration audit finding 4) was built for exactly this collection.

What survives untouched: all teacher-side findings (N8, turn-split pacing +
drift cure, 7B scale result); the head's teacher-forced quality (July: 0.030
WER / 0.984 sim); the speed ledger (6% head share, faster-than-realtime
loop). heun8 arm + B (profile) + C (parity) verdicts pending bundle.

### OFFICIAL CLOSE-OUT (scored 2026-08-12; `gate_night4_metrics.json`)

**Arm A — both samplers FAIL, with a materially different failure depth:**
| | euler4 | heun8 |
|---|---|---|
| Whisper coverage | 95 words (**2.9%**) | 1711 words (**53.0%**) |
| last voice-like window (vs teacher ref) | 0:14 | 0:52 |
| post-collapse character | quiet unstable resonance (−30..−60 dB) | loud garbled speech throughout (−11..−17 dB) |
| closed-loop latent std | 0.503→1.262 (under→noise) | 1.274→1.302 (over, stable) |

The loop is an amplifier of whatever bias the head injects: under-dispersion
(euler4) spirals to near-silent resonance; heun8's dispersion-correct field
*degrades* rather than dies — half the script survives as garbled speech.
Collapse horizons + decile curves: `gn4_collapse_horizons.json`. For the
Self Forcing stage this is the starting line: on-policy fine-tuning begins
from 53% in-loop intelligibility (heun8), not zero.

**Arm B — bottleneck map (measured, L4, ms/frame):** teacher 187.2 = LM 63.8
+ DDPM head 50.7 + connectors 0.6 + unhooked glue/tokenizers ~72. Flow head
127.5 = LM 64.2 + head 7.7 + connectors 0.6 + glue ~55. Ratio **1.468×
measured** (July's estimate confirmed per-module). Biggest remaining targets:
the LM's 64 ms (carries the CFG double stream) and the ~55–72 ms glue bucket.

**Arm C — PARITY OK. Blocking item 2 CLEARED.** Solo-vs-batch-4 WER per utt:
0.000 / 0.071 / 0.000 / 0.000 (mean 0.018, well inside 1.5× reseed floor);
sim 0.639–0.717 ≈ the teacher's own reseed band (0.734). The frame-count
divergence (audio lengths ±16–35%) is different-but-valid renditions, not
degradation. Batch throughput measured ~2.9× at batch-4 (35 s solo → 12 s
batched). **Consequence: the teacher-side product pipeline (turn-split ×
batch × [optionally our head for the 1.47×]) has now passed every quality
gate — pacing, drift, parity. It is shippable with the stock model today.**

**Blocking-list state after GN4:** (1) reseed floor n=30–50 — OPEN, → GN5;
(2) batch parity — CLEARED; (3) cache filter — settled (ECAPA+rate);
(4) rate correction — settled (turn-split capture).

## 2026-08-12 — GATE NIGHT 5 PRE-REGISTRATION (not yet run)

Notebook: `gate_night5_colab.ipynb`. Scorer: `score_gate_night5.py`. L4,
~1.5 h, ~$2–3. Merges the external handoff docs' best proposals with our
queue; assessment of those docs (what was adopted vs corrected) is in the
2026-08-12 session log. **Venue decision reaffirmed by Josh: ICASSP stays
dead; Interspeech 2027.** Literature basis verified: Causal Forcing
(2602.02214, ICML 2026 — AR-teacher init + DMD stage 2) and Causal Forcing++
(2605.15141 — causal consistency distillation, frame-wise 1–2 step, no cached
trajectories; our regime exactly). Reading of dots.tts + CF++ happens
off-GPU in parallel; stage-2 method decision follows it.

### Arms and gate criteria — written before the run

- **F — feedback-path ablation** (the night's headline; dots.tts hypothesis).
  euler4 closed loop (most sensitive assay), 5-min turn-split script, acoustic
  connector output: base / zeroed / running-mean (the fair in-distribution
  null) / noised (σ=0.5·running std).
  | Verdict | Condition |
  |---|---|
  | ACOUSTIC CHANNEL CONFIRMED | any non-base condition survives ≥3× base's horizon → an inference-time (plumbing) fix exists; windowed/anchored context re-promotes as co-headline |
  | NO PLUMBING RESCUE | nothing reaches 3× → Self Forcing–family training confirmed as the only path |
  Note: if *noise* collapses like base while *mean/zero* survive, corruption
  (not absence) of acoustic feedback is the poison — cleanest possible
  mechanism statement.
- **D — Fréchet-distance backfill.** σ-VAE re-encode of GN4 renders + GN3
  teacher; per-10s-window FD vs the teacher's first-half distribution.
  Deliverable, not a gate: the euler4/heun8/teacher dose-response figure with
  an honest metric (scalar std retired after two failures: E3, GN4).
- **R — reseed floor, n≤40** (blocking item 1). Teacher, seeds 0/1 per
  held-out utt. Deliverable: **median + IQR** pair-WER (GN1's n=6 mean was
  carried by one proper-noun utterance). Floor is usable if IQR width <
  ~2× median.
- **S — chunked-parallel seam test** (product path). 2-speaker ~750-word
  dialogue, 4 chunks split at turn boundaries, rendered as ONE batch, 0.25 s
  crossfade. Teacher head. Metric: seam ECAPA vs within-chunk baseline;
  **Josh listens to the 3 seams — ears are the authority** (GN2 lesson:
  short-window ECAPA over-flags prosody at boundaries).
- ~~**H — batch hygiene: the [A,A] test.** Two identical rows in one batch,
  same seed — a FAIR determinism criterion (unlike solo-vs-batch
  bit-equality, which floating-point batching cannot satisfy). Identical
  outputs → masking clean; divergent → genuine cross-contamination.~~
  **[CRITERION RETRACTED 2026-08-14, before scoring, on first sight of the
  result (rows 10.0 s vs 7.9 s).]** The test is flawed by the same disease it
  was meant to cure: sampling noise is drawn as one stream across the batch,
  so identical rows receive DIFFERENT noise slices by construction —
  divergence is expected, not evidence of leakage. No bitwise criterion can
  separate benign numeric jitter from mask leakage in a stochastic AR system,
  because any infinitesimal difference amplifies. **The correct and only
  instrument is distributional quality parity — GN4 Arm C — which PASSED.**
  The scorer's "CONTAMINATION" verdict for arm H is void. Both the handoff
  docs' bit-determinism demand and this "fairer" variant fail the same way;
  retiring the entire determinism-test genre for this system.

Listening steps (never skipped): `s_chunked_stitched.wav` seams; the
best-surviving `f_abl_*` render (is "survives" actually speech?).

### Results — partial (2026-08-13; disconnect after Arm F; D/R/S/H pending rerun)

**Arm F v1: INVALID as a feedback ablation — design error, caught by ear.**
Josh: `f_abl_zero` has "no voice at all, just millisecond artifacts" from
second one → the acoustic connector carries the **voice-prompt encoding**;
v1's hook fired during prompt processing and erased the speaker rather than
isolating feedback. All four v1 conditions corrupt input conditioning.
v1 wavs kept as evidence. **v2 shipped same day** (`f2_abl_*` tags): the
intervention gates on `patch.calls > 0` so the prompt is encoded untouched
and only per-frame feedback traffic is modified.

**New mechanism findings from the v1 run (valid despite the design error):**
1. **Bit-identical frame counts across all four conditions (2296)** —
   pacing and stop timing are **text-anchored**: the LM marches through the
   script on schedule regardless of acoustic-channel chaos. (Also the
   base render's collapse boundary is a *clean stop* after "diary of Mrs
   Gilroy" — phrase-aligned, not a drift-out.)
2. **Speech RE-EMERGES mid-collapse.** Josh, base render: at ~0:32,
   oscillating noise gives way to intelligible *later script content* ("will
   that be honorable... she knew too much about the murder") before washing
   away again. GN4's "voice in the wind" was this same phenomenon. The
   collapse is **bistable orbiting of the speech manifold, not divergence**
   — the system keeps re-approaching the speech attractor (plausibly at turn
   boundaries) even deep in collapse. Encouraging for the training path:
   Self Forcing needs to widen a basin the loop already re-enters, not drag
   the system back from infinity. The FD curves (Arm D, pending) should show
   these re-entries as transient dips.

Latent stds (v1: base 0.942 = GN4's 0.943 ✓ consistency): zero 0.821, mean
0.857, noise 1.101 — different conditioning did change head output, but with
the prompt path corrupted no feedback conclusion can be drawn.

**Arm F v2 zero — verdict by ear (Josh, 2026-08-13): WORSE than base. ~3
millisecond blips in the first seconds, then nothing.** With the prompt
untouched and only per-frame feedback zeroed, speech dies essentially at
frame one. Ranked by what the model hears in its feedback channel:
teacher-real → 19+ min clean; head-imperfect → 14 s + bistable flicker;
**absent → instant permanent silence. Imperfect ≫ absent: the acoustic
feedback channel is load-bearing by training** (VibeVoice never saw a
position without real acoustic context; a zeroed embedding is maximal OOD,
not "silence"). The dots.tts semantic-only wiring works for them because
they TRAINED that way — it cannot be retrofitted at inference. **Amputation-
style plumbing fixes are dead.** (mean/noise ear-verdicts + FD pending;
noise is the interesting residual — mild corruption of REAL feedback.)

**v3 shipped (`f3_renorm`): the one gentle plumbing candidate left.** Don't
remove the feedback — CORRECT it: per-frame whiten the head's latent and
re-color to the cache's per-dim teacher stats before it re-enters the loop
(also emitted as audio → doubles as post-hoc dispersion correction). Directly
targets the measured amplifier mechanism: no per-frame bias left to compound.
| f3 outcome | meaning |
|---|---|
| horizon ≫ 14 s | inference-time stabilization exists; also strong evidence Self Forcing will work (same principle, learned) |
| ≈ 14 s | the compounding error is structural (content/direction, not frame statistics) → training path only, and stage-2 must fix more than dispersion |

**f3_renorm verdict by ear (Josh, 2026-08-14): WORSE than base — ~3 s of
speech, slow fade into a STEADY white-noise/hum with slight reverb, constant
to the end (306.1 s, full length, text-anchoring again).** The steady-hum
signature is self-diagnosing: per-frame whitening forces silence-class frames
up to speech-level statistics → constant-energy noise floor; and destroying
legitimate per-frame structure killed even the early good frames (3 s vs
base's 14 s). Second-worst branch of the pre-registered table →

~~**GATE NIGHT 5 HEADLINE VERDICT: NO PLUMBING RESCUE — the plumbing era is
closed by elimination.**~~ **[AMENDED hours later, 2026-08-14, by the noise
condition — the criteria measured the wrong axis.]** The verdict splits:

- **Identity axis (what the pre-registered ECAPA criterion measured): NO
  RESCUE stands.** No condition keeps the reference voice: zero/mean die
  instantly, renorm is worst-of-program (FD med 4170), noise never clears
  voice-likeness 0.5.
- **Content axis: `f2_abl_noise` is a MASSIVE partial rescue.** Whisper
  recovers **747/~803 words (93%)** vs base's 82 (10%); FD climbs slowly
  (673 → med 990 → 1289) instead of jumping. **Josh, by ear (2026-08-14,
  verbatim):** "starts a little muddy like normal... voice is still there...
  goes into like a raspy deep creepy voice... gradually into a sort of loud
  whisper... like when someone loses their voice, **I can understand it**...
  by minute four more raspy, and between speech pauses a digital breath,
  almost like a record pulled in reverse for a split second... but speech
  resumes. **It actually makes it full 5 minutes.**"

**Mechanism nailed: the loop's killer is specifically the CORRELATED
component of the head's error.** σ=0.5 feedback noise decorrelates the
systematic bias → compounding never takes hold → content survives the full
render (14 s → 5 min, >20×, zero training); the cost is gradual erosion of
fine acoustic identity (clean → raspy → whisper), i.e. the two things the
feedback carries — content and identity — fail on different axes. Contrast
completes the picture: v1 noise (prompt+feedback noised) was WORSE than base
(FD med 2046) — the prompt must stay clean; only the feedback tolerates
(benefits from!) noising.

**Consequences:**
1. **GN6 candidate — the σ sweep** (0.1/0.2/0.3/0.5 + maybe a schedule): does
   a sweet spot exist that breaks compounding AND keeps the voice listenable?
   If yes → deployable inference-time stabilizer for the fast head TODAY. If
   no → the content-vs-identity-vs-σ tradeoff curve is a novel figure anyway.
2. **Stage-2 de-risked:** noise-augmented rollout context demonstrably keeps
   this system on-manifold — direct pre-evidence the Self Forcing family will
   bite. Training's precise job is now: achieve what σ-noise achieves
   (decorrelate the error) without paying the identity cost (make the head's
   residual error small AND isotropic).
3. ~~Whisper-vs-teacher WER verification of the 747 words pending~~
   **VERIFIED (2026-08-14): real script, not hallucination.** Transcripts
   open word-for-word identically ("fancy said lucy running to meet
   coniston..."); full 803 words attempted; **WER 0.296 vs the teacher's own
   rendition** — the raspy register costs ~30% word accuracy but the content
   is unambiguously the script. Three-tier confirmation: FD curve, transcript
   WER, Josh's ear.

Prior conclusion that survives intact: training (stage-2) remains the road to
SHIPPABLE closed-loop quality — noise plumbing rescues content, not the
voice. Next: dots.tts + CF++ reading → stage-2 method decision; σ sweep as
the cheap parallel probe.

**Novelty calibration + third stage-2 candidate (2026-08-14 discussion).**
Honest positioning of the noise finding: the *principle* (noisy history
fights AR drift) has video precedent — Diffusion Forcing trains under
per-frame noise; GameNGen fought world-model drift with training-time noise
augmentation of context frames (verify both citations in the reading pass).
What is OURS: (1) the **inference-time causal isolation** — noise injected
into a frozen system + untrained head, no training confound, proving the
collapse driver is the CORRELATED error component (renorm-fails-worse is the
matched control: same statistics, wrong structure, opposite outcome);
(2) the **content/identity axis split** — speech feedback carries two
streams that fail independently (words survive σ-noise, voice does not); no
video analog; (3) first documentation on TTS. Consequence for stage-2: the
method menu is now THREE options — full Self Forcing/CF++ distribution
matching (heavyweight), DAgger-style teacher labeling (2024-vintage), and
**GameNGen-style noise-augmented-feedback training** (lightweight, closest
to existing infra, and the GN5 noise result is direct evidence it targets
the right mechanism). The reading pass decides; the σ sweep informs the
noise-augmentation schedule either way.

**Arm S verdict by ear (Josh, 2026-08-14): PASS.** "Sounds fine and clear,
stitching is good, paced between them" — seams inaudible. One transient at
**1:10–1:15**: Speaker 2 "moans and then sounds like she's throwing up," a
one-time artifact. Placement analysis: chunks run ~65–70 s, first seam ≈1:07 —
the event sits a few seconds INTO chunk 2's fresh generation, i.e. the known
early-generation-settling window, and the character matches the
**transient-babble defect class** (third confirmed sighting: GN3 p1 9:40,
Josh's HF space, now here). Not a stitching failure.
**Product consequence — detect-and-reroll:** chunked architecture converts
transient babble from a 90-min-take catastrophe into a ~70 s chunk re-roll
with a different seed; the ECAPA+rate filter is the detector (2-for-2 on this
class by machine + 3 sightings by ear). At GN3's observed rate (~1 event /
40 min), a 90-min render needs ~2–3 re-rolls ≈ minutes of extra compute.
**The product path is now validated end to end by ear: pacing ✓ drift ✓
batch parity ✓ seams ✓ glitch-repair story ✓.**

**FIRST PRODUCTION BENCHMARK (2026-08-14, Josh's Conference-Generator Space +
Modal A100-40GB, deployed this day):** VibeVoice-**7B**, 4 speakers, 17-turn
955-word script → parallel mode, 4 chunks in 1 wave → **120.4 s for 322.1 s
of audio = 2.67× realtime** (warm container; pure generation). Community
reported experience for the same model: RTF 0.5 on an H200 — i.e. ~5×
slower on a bigger GPU. Chunked-parallel pipeline live in production;
projections: 90-min 4-speaker ≈ 34 min at 7B quality, likely <20 min on
1.5B/batch-8. Seam/quality listen on 7B pending (parity was gated on 1.5B).

**Field check (2026-08-14 searches): no competing "record" exists.** Wild-side
TTS speed today: official podcast demo 1.8× SLOWER than realtime; open issue
microsoft/VibeVoice#268 reports RTF 0.5 (2× slower) on an idle H200 for both
1.5B and 7B — i.e. the community's 90-min render costs 2–3 h of premium GPU.
No published timed 90-min 4-speaker generation exists at any speed; ComfyUI
wrappers chunk sequentially with no speed claims. The chunk+batch playbook
exists in the ecosystem ONLY on the ASR side (JacobLinCool/modal-vibevoice:
VAD-aware chunking + auto-batching + speaker unification, 15× realtime —
verified zero TTS functionality). Chunked-parallel GENERATION is unclaimed;
shipping it (~90 min in ~25 min on modest hardware) would be, on public
evidence, the fastest demonstrated long-form multi-speaker TTS — enabled
specifically by the GN2–GN5 quality receipts (turn-split pacing, anchored
no-drift, seam ear-pass, batch parity) that nobody else has.

(D FD curves / R floor official numbers pending bundle + scorer)

## Gate Night 1 continued — venue read (updates review §5)

The scaling curve was the ICASSP-vs-Interspeech decision gate, and it is
**unreadable until the floor is recalibrated**. With three blocking items ahead
of it and ICASSP 2027's 2026-09-16 deadline five weeks out, **ICASSP is off the
table on schedule grounds** — not on results. Target **Interspeech 2027
(~Mar 2027)** at full scope, arXiv when ready.
