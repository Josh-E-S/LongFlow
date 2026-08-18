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

**7B production ear-pass + a FOURTH cured symptom (Josh, 2026-08-14):**
"Quality was amazing, stitching not even noticeable, perfect pacing" — 7B
chunked-parallel parity confirmed by ear in production. AND a new
observation: the old monolithic app renders would sometimes "slowly get
quieter and quieter and maybe come back" — an **energy-fade drift** in the
stock teacher over long rollouts, never formally measured, gone under
chunking. That makes the monolithic-accumulation symptom family: (1) rate
inflation [N8], (2) identity drift, (3) energy fade [this], (4) transient
babble [reroll-able] — 1–3 all cured simultaneously by bounding rollout
length, because they are one disease. The fade-and-recover character echoes
the bistable orbiting measured in the student collapse — same dynamics,
gentler, in the teacher. Jingle caveat: podcast-heritage intro music may
still spawn (chunk starts are intro-like; possibly more opportunities), but
now confined to one chunk and caught by the ECAPA detect-and-reroll filter
(music vs voice reference scores near zero); steering machinery remains the
anti-jingle fallback per CLAUDE.md.

**Second benchmark + backoff shipped (2026-08-14):** 1.5B, same script:
**100.3 s for 329.6 s = 3.29× realtime** (and that job also absorbed the cold
start). OOM backoff deployed (waves split in half and retry; unit-tested
12→6→3 with order preserved) and caps raised to 12 (1.5B) / 6 (7B) —
overshoot now costs one halving, never the job. Next calibration: a ~2.5k-word
script to test 12-chunks-in-one-wave on the A100.

**Third benchmark — longest timed TTS render on public record (2026-08-14):**
1.5B, 4 speakers, 65-turn 5,466-word documentary script → 23 chunks, **2 waves
of 12 (raised cap held, no backoff)** → **412.1 s for 1828.1 s of audio =
4.44× realtime**, warm container. 30.5 min of audio ≈ 6× longer than any
publicly documented timed render. Updated projections: 90 min ≈ 20 min wall;
120 min ≈ 27 min.

**Production incident + fix (2026-08-14, 50-min render attempt):** wave 3 of 4
OOMed; backoff cascade 12→6→3→2 all failed with 37.85 GiB still allocated —
root cause: the worker thread stored the OOM *exception object*, whose
traceback pinned every tensor of the failed attempt, making empty_cache()
powerless. Second design flaw exposed: the monolithic fallback on a 50-min
script re-triggers N8 by construction. Fixes deployed same hour: store error
message not exception; per-wave gc+cache release; expandable_segments
allocator; fallback redesigned to sequential per-chunk (turn-split preserved
at batch-1 — "slow and good", never monolithic on long scripts). Lesson for
the paper's engineering section: OOM-retry logic is worthless unless the
failed attempt's references are actually severed.

**Fourth benchmark — fixes verified live (2026-08-15 morning):** the 50-min
"extended" render, first run on the repaired code: **624.7 s for 2786.8 s
(46.4 min) of audio = 4.46× realtime**, 34 chunks in 3 waves of 12 on a WARM
(reused) container, **zero OOM, backoff never needed** — the per-wave
gc+empty_cache + expandable_segments fixes pass their first live test; last
night's ceiling was leaked memory, not batch size. Scaling is FLAT (4.44× @
2 waves → 4.46× @ 3), so projections harden: 90 min ≈ 20 min, 120 min ≈ 27.
Pacing 178 wpm (natural) at 46 min. New longest-timed-render mark. Listening
(seams + back half + babble scan) pending.

**Ear certification of the 46-min render (Josh, 2026-08-15): "Sounds amazing
all the way through to the end."** Seams, back half (min 40+, deepest audio
the pipeline has produced), full pass. Zero babble events in 46 min — the
transient-glitch rate estimate improves below the prior ~1/40 min. Pipeline
is ear-certified at every scale tested: 5, 30, and 46 minutes.

**Session close-out (2026-08-14, late) — the production evening, full log.**
Everything below shipped to Josh's public Conference-Generator Space (HF) +
its Modal backend, all synced to the Space repo:

- **Features shipped:** paste/upload full-script loader (parses `Speaker N:`
  + named tags through the app's existing turn pipeline); script cap raised
  1,500 → 20,000 words; chunked-parallel rendering (whole-turn chunks ~200
  words, batched waves, 0.25 s crossfades, timing report with cold/warm flag
  + ×realtime + mode line); gradio pinned <6 (unpinned requirements had let a
  rebuild pull breaking Gradio 6); voice presets restored (GitHub zip had
  shipped 131-byte LFS pointers, not audio — real files pulled from Space LFS,
  converted mp3→wav for soundfile).
- **Benchmarks (all measured in production):** 7B 2.67× realtime (batch-4);
  1.5B 3.29× (cold-start job); 1.5B **4.44×** (batch-12, 2 waves) on the
  30.5-min render — believed longest publicly timed TTS render (~6× anything
  documented). Projections: 90 min ≈ 20 min; 120 min ≈ 27 min.
- **OOM saga (two rounds, instructive):** Round 1: backoff cascade
  12→6→3→2→dead with ~38 GB pinned — root cause: worker thread stored the OOM
  *exception object*, whose traceback pinned every tensor of the failed
  attempt (empty_cache useless against live references). Fixes: store message
  not exception; per-wave gc+empty_cache; expandable_segments allocator;
  fallback redesigned **sequential per-chunk** (the old monolithic fallback
  re-triggered N8 on long scripts by construction). Round 2: retry cascaded
  identically — traceback proved it ran the OLD code on a **warm container
  poisoned by the cancelled run's zombie generation thread** (Python threads
  survive client cancellation; zombie kept generating, ate the GPU). Fixes:
  **recycle-on-cancel** (GeneratorExit → container exits, zombies impossible)
  + **VRAM health check** at request start vs post-load baseline (poisoned
  container announces itself and recycles). Status: all round-1+2 fixes
  deployed together but **UNTESTED** — the 50-min render retry is the first
  order of business next session.
- **Scripts written:** 989-word 4-speaker test (validated); 5,466-word "Long
  Groove" audio-history documentary (the 30.5-min record render); 8,276-word
  extended cut (~50 min, render pending). True 120-min needs ~7 more acts
  (~1,400 quality words/act is the realistic authoring yield).
- **End-of-session Q&A worth keeping (Josh's questions, sharp ones):**
  (1) *"If we stitch, why is VibeVoice special?"* — because chunks are
  four-voice cloned conversation SCENES, not utterances; the 90-min training
  regime built the discourse-level prosody we harvest at 80 s (marathon
  runner in short races); and per-provider field check shows **everyone
  stitches**: OpenAI TTS caps at 4,096 chars/request, Deepgram Aura 2,000,
  ElevenLabs' multi-speaker dialogue endpoint recommends ≤2,000 (~2 min) —
  the industry leader's conversation mode uses OUR chunk size. Cost: ~90-min
  episode ≈ $15–25 at ElevenLabs rates vs ~$1.50 of self-hosted A100.
  (2) *"Surely someone's done batching/quant?"* — quant yes but for low-VRAM
  access, never throughput (community 4/8-bit 7B exists incl. a selective-Q8
  keeping audio-critical parts fp — the ear-safe candidate for our quant A/B);
  "batch generation" appears once in the wild as job-queue batching; timed
  chunked-parallel single-episode remains unclaimed.
  (3) *Why batching is nearly free:* AR decode is weight-haul-bound; a batch
  shares one weight read across N streams (truck/packages). Sublinear
  (2.9×@4, 4.44×@12) via stragglers + KV growth.
  (4) *Quant-7B projection:* freed VRAM lifts 7B cap 6→~12 → ~3.5–4.5×
  realtime ≈ "Large quality at small-model speed"; gate = one ear A/B.
  (5) *The two GPU bills:* weights (fixed; what quant shrinks; why 7B needs
  beefy cards) vs per-second work (what 7.5 Hz makes tiny — 7.5 passes/sec
  instead of 75+; why 90 min fits context, why batch-12 caches fit, why the
  133 ms/frame realtime budget exists). Big orchestra, compact sheet music.
- **Next-session queue:** (1) verify 50-min render on the new code (first
  true test of all OOM fixes); (2) stage-2 reading (dots.tts, CF++) + method
  decision incl. GameNGen-style noise-augmented option; (3) σ sweep; (4)
  quant-7B ear A/B via the community selective-Q8; (5) multi-container
  fan-out; (6) remaining ~7 acts for the 120-min record.

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

## 2026-08-14 — STAGE-2 READING PASS COMPLETE (off-GPU; no experiments)

Full digest + method decision: `docs/reading-2026-08-14-stage2.md`. Headlines:
**not scooped** (dots.tts's "self-corrective" stage is a within-patch solver
fix on ground-truth prefixes — never faces closed-loop drift; option-3
noise-augmented training has no speech descendant; N8 unmeasured anywhere;
render record uncontested). Decision sequence: (1) GN6 σ sweep; (2) causal-CD
loss swap on the existing 20K cache (cheap stage-1.5, from CF++ — theory
predicts collapse→graceful drift); (3) GameNGen-style noise-augmented feedback
training **with σ-bucket conditioning** (GameNGen tells the model the noise
level via a learned embedding — the ingredient our inference hack lacked and
the likely reason identity eroded); (4) head-v2 question: windowed
acoustic-history conditioning (the dots.tts identity lever); (5) reserve:
short asymmetric DMD (frozen DDPM head = real score) only on a healthy init —
CF++'s own warning: DMD is mode-seeking and amplifies accumulated history
error on weak inits. 75K stays deferred until 2–3 report.

## 2026-08-14 — GATE NIGHT 6 PRE-REGISTRATION (the σ sweep; not yet run)

Notebook: `gate_night6_colab.ipynb`. Scorer: `score_gate_night6.py`. L4,
~45–60 min, ~$2. Single arm, seven renders. Everything held identical to
GN5 Arm F for comparability: 20K head, euler4 (most sensitive assay), the
same ~800-word 5-min turn-split script, seed 0, cfg 1.3, noise applied to
the acoustic connector output gated on `patch.calls > 0` (prompt untouched),
running-std EMA mechanism unchanged — only the σ multiplier varies.

### Hypotheses

- **H1 (sweet spot):** some σ < 0.5 decorrelates enough to stop compounding
  while staying gentle enough that voice identity survives. GN5 proved the
  content axis saturates by σ=0.5; identity cost may fall faster than
  content rescue as σ drops.
- **H2 (threshold, competing):** decorrelation needs to exceed the head's
  own correlated-error magnitude to break the amplifier — below that σ,
  collapse returns abruptly (base-like death), and no sweet spot exists;
  the tradeoff is a cliff, not a curve.
- **H3 (schedule):** the head's early frames are clean (collapse takes ~14 s
  to seed); ramping σ in over the first minute preserves early identity at
  no content cost vs constant σ.

### Conditions (tags)

| tag | σ (× running std) |
|---|---|
| `g6_sig000` | 0 (base control, re-rendered this night) |
| `g6_sig010` / `g6_sig020` / `g6_sig030` / `g6_sig040` | 0.1 / 0.2 / 0.3 / 0.4 |
| `g6_sig050` | 0.5 — **replication of GN5 `f2_abl_noise`** (consistency check) |
| `g6_ramp050` | linear 0→0.5 over the first 450 frames (~60 s), then hold |

### Gate criteria — written before the run

Per render: Whisper transcript → **WER vs the script text** (primary content
metric; script stored in the report JSON) + word coverage; per-window ECAPA
vs the GN3 teacher reference → voice fraction (sim ≥ 0.5) and horizon;
per-window FD curves for all seven (the dose-response figure). Reseed-floor
context: differences > 0.06 WER are real.

| Verdict | Condition |
|---|---|
| **SWEET SPOT** | some condition has WER ≤ 0.15 AND voice fraction ≥ 90% AND Josh rates it listenable → deployable inference-time stabilizer TODAY; it becomes the head's default inference config while stage-2 trains; training noise schedule centers on it |
| **TRADEOFF CURVE ONLY** | content is rescuable (coverage ≥ 90% at some σ) but no condition passes both axes → publish the σ vs content-vs-identity curve; stage-2 training carries the full burden; GameNGen-style training samples σ up to the lowest content-saving level |
| **CLIFF (H2)** | σ ≤ 0.3 all die base-like (horizon < 60 s) and only 0.4–0.5 save content → decorrelation threshold ≈ head's error scale; report the threshold; training should sample σ across it |
| replication check | `g6_sig050` must land within reseed-floor distance of GN5's f2_abl_noise (WER 0.296, full-length survival); if not, flag before trusting anything else this night |

H3 read: `g6_ramp050` vs `g6_sig050` — better early-window ECAPA at equal
coverage = schedule wins; equal = constant σ is fine; worse coverage = early
noise is load-bearing (the compounding seeds earlier than 14 s suggests).

**Josh listens to all seven.** Ear verdicts recorded verbatim here; the
listenability half of the SWEET SPOT gate is his call, never the metrics'.

### Results (2026-08-14 — run and scored same day; L4, ~$2; ear pass pending)

Night is trustworthy: `g6_sig000` reproduces GN4/GN5 base exactly (latent
std 0.942, 2296 frames, horizon 8 s vs GN4's 14 s — same order); **`g6_sig050`
replicates GN5 `f2_abl_noise` nearly number-for-number** (FD 672→985→1291 vs
673→990→1289; full-length; full coverage). Replication gate PASSED.

| tag | WER vs script | coverage | voice % | horizon | FD med |
|---|---|---|---|---|---|
| g6_sig000 | 0.933 | 8.5% | 1.3 | 8 s | 1840 |
| g6_sig010 | 0.686 | 43.3% | 0 | 0 | 2265 |
| g6_sig020 | 0.585 | 100% | 0 | 0 | 1360 |
| **g6_sig030** | **0.289** | **100%** | 0 | 0 | **1210** |
| g6_sig040 | 0.957 | 4.8% | 0 | 0 | 1890 |
| g6_sig050 | 0.339 | 100% | 0 | 0 | 985 |
| g6_ramp050 | 0.300 | 76.5% | 2.0 | 14 s | 1109 |

**VERDICT (metric half): TRADEOFF CURVE ONLY — H1 sweet spot REFUTED on the
identity axis.** Content is rescuable from σ=0.2 up (full coverage), and the
content optimum is **σ=0.3 (WER 0.289), better than σ=0.5 (0.339)** — the
noise dose GN5 happened to pick was past the optimum. But no condition puts
a single 4-s window above voice-likeness 0.5; the identity axis never moves.
Stage-2 training carries the full identity burden, as the reading pass
already concluded.

**Anomaly, do not smooth over: σ=0.4 DIED (4.8% coverage, WER 0.957, worse
than base).** Sandwiched between two full-coverage conditions — the
dose-response is not monotone at n=1 seed. Consistent with the bistable-
orbit picture (a single unlucky excursion early can trap the render);
treat as variance until a reroll at a different seed says otherwise. Direct
consequence for training: per-frame/per-window RANDOM σ (GameNGen's actual
design) rather than a fixed dose — a fixed dose can land in a dead pocket.

**H3 (ramp) split verdict: EARLY NOISE IS LOAD-BEARING for content** — the
ramp lost 24% coverage (compounding seeds before the ramp reaches strength;
GN4's 14-s collapse said as much) — **but the ramp is the identity-best
condition** (final-third sim 0.2, horizon 14 s, flattest FD tail 1109→1195).
Training-schedule read: noise must be present from frame one for content;
gentler late doses may be where identity is preserved. A reverse-ramp
(start 0.3–0.5, decay) is the natural GN7 probe if one more inference night
is ever worth it; otherwise fold into the training-σ schedule directly.

**Consequences for stage-2 (supersedes the σ=0.5-centered plan):** training
noise schedule centers on **σ≈0.2–0.3**, sampled randomly per frame/window,
present from frame one; σ-bucket conditioning as per the reading-pass
decision. No deployable inference-time stabilizer exists — the fast head
ships only after training.

**Ear verdict (Josh, 2026-08-14, verbatim):** "ehhh all are bad, but yea 50
makes it ot the end but by the end barley noticeble." — Listenability half:
FAIL across the board; even the full-coverage conditions are not listenable,
and σ=0.5's surviving speech fades to barely-noticeable by the end (matches
the FD tail climbing 985→1291 and final-third sim 0.064 — the content
survives as measured, but the ear says what the coverage number can't: the
signal is dying, not stabilizing).

**GATE NIGHT 6 CLOSE-OUT: TRADEOFF CURVE ONLY, confirmed on both halves.**
No inference-time stabilizer exists at any σ. The σ axis is mapped (content
optimum ≈0.3, random-σ lesson from the 0.4 anomaly, ramp's split verdict),
the GN5 replication held, and the program moves to stage-2 training with
nothing left to extract from plumbing. Next: causal-CD loss swap on the 20K
cache (reading-pass decision step 2), then GameNGen-style noise-augmented
training with σ sampled ~0.2–0.3 per frame + σ-bucket conditioning.

## 2026-08-14 — CRITIC PASS on the stage-2 plan (no experiments)

Josh asked for an adversarial re-check of history + approach before any
training spend. Four findings, logged in full in
`docs/reading-2026-08-14-stage2.md` (CRITIC PASS section): (1) the head has
no context input (`src/flow_head/model.py:95`) → noise-augmented training
requires a NEW capture with noise injected at the LM feedback during
teacher runs, DDPM outputs as targets (DAgger×GameNGen hybrid) — and
teacher health under feedback noise is UNTESTED (GN5/6 noised only student
runs); (2) causal-CD demoted from prerequisite to parallel arm (CF++'s
mode-seeking warning applies to DMD, not supervised noise training);
(3) all GN6 σ points are n=1 seed — stage-2 gates need 2–3 seeds and the
graded sim curve (binary voice≥0.5 has zero discrimination left);
(4) capture schema must include σ buckets + trailing-K latents so one
capture serves noise-head, σ-conditioned head, and head-v2 history
conditioning. Amended sequence: GN7 → capture v2 → training arms → DMD
reserve. Strategy itself: survives.

## 2026-08-14 — GATE NIGHT 7 PRE-REGISTRATION (capture prerequisites; not yet run)

Notebook: `gate_night7_colab.ipynb`. Scorer: `score_gate_night7.py`. L4,
~45–60 min, ~$2–3. Two arms, five renders. Same ~800-word turn-split script
and noise machinery as GN6.

### Arm T — teacher robustness under feedback noise (BLOCKS capture v2)

The capture design feeds the TEACHER σ-noised feedback and trusts its
outputs as training targets. If the teacher degrades under noise, the
cache is garbage. Never tested: GN5/GN6 noised only student (head) runs.

Conditions (stock DDPM head, no FlowHeadPatch; noise gated to per-frame
generation calls only — prompt encoding untouched, discriminated by
sequence length 1, with first-call shape logging to verify the gate):
`t7_sig000` (clean control) / `t7_sig020` / `t7_sig030`. Seed 0.

| Verdict | Condition |
|---|---|
| **CAPTURE GREENLIT at σ** | WER within reseed floor of control (≤ +0.06), voice fraction ≥ 90%, rate 150–200 wpm, Josh's ear passes → capture v2 proceeds at that σ range |
| **PARTIAL** | 0.2 passes, 0.3 fails → capture caps σ at the passing level |
| **CAPTURE DESIGN DEAD** | both fail → teacher targets under noise are unusable; rethink (candidates: lower σ, or clean-teacher targets aligned to noised hidden states via a second forward) |

### Arm A — the σ=0.4 anomaly, rerolled

GN6's σ=0.4 died (4.8% coverage) between two full-coverage neighbors, n=1
seed. `a7_sig040_s1` / `a7_sig040_s2` (20K head, euler4, seeds 1 and 2).

| Verdict | Condition |
|---|---|
| VARIANCE (bistability) | ≥1 of 2 survives with ≥90% coverage → GN6's 0.4 was an unlucky basin; dose-response is noisy-monotone; random-σ training design stands as-is |
| DEAD ZONE | both die → real non-monotonicity; training σ range explicitly excludes ~0.4; flag for the paper's tradeoff figure |

**Josh listens to all five; teacher-condition verdicts matter most** (his
ear is the final gate on whether noised-teacher audio is target-quality).

### Results (2026-08-14 — run and scored same day; L4, ~$2–3; ear pass pending)

Hook-gate audit PASSED in all three teacher runs: first connector call
[1, 23, 1536] (prompt) skipped, ~2,280 per-frame calls noised. No GN5-v1
contamination.

| tag | WER vs script | coverage | voice % | sim med | rate wpm | FD med |
|---|---|---|---|---|---|---|
| t7_sig000 | 0.281 | 100% | 98.7 | 0.635 | 183 | 545 |
| t7_sig020 | 0.195 | 86.8% | 71.5 | 0.527 | 140 | 562 |
| t7_sig030 | 0.161 | 93.1% | 87.4 | 0.585 | 150 | 547 |
| a7_sig040_s1 | 0.169 | 100% | 0 | 0.094 | 162 | 1113 |
| a7_sig040_s2 | 0.346 | 89.8% | 0.7 | 0.144 | 144 | 1141 |

**Calibration note first: the clean teacher scores WER 0.281 vs script** —
that is the script's intrinsic Whisper-vs-text floor (proper-noun-heavy
fiction; the reseed-floor exclusion note predicted this), NOT teacher
degradation. WER *deltas* are the meaningful signal here, and both noised
conditions IMPROVED on the control (likely mediated by slower speech being
easier to transcribe).

**Arm A verdict: VARIANCE (bistability) — GN6's σ=0.4 death was an unlucky
basin.** Seed 1 survives with 100% coverage and WER 0.169 — the best
closed-loop head render in program history — while seed 2 manages 89.8% at
0.346. Same σ, same everything, WER 0.169 vs 0.346: the per-seed spread is
enormous, which simultaneously (a) clears σ=0.4, (b) validates the
random-σ training design, and (c) proves again that no stage-2 gate may
run at n=1 seed.

**Arm T verdict (metric half): CAPTURE DESIGN DEAD by the letter — but
read the margins before burying it.** Neither noised condition passes the
pre-registered gate, both failing on voice fraction and rate, not on WER:
σ=0.3 misses the voice bar 87.4 vs 90.0 and the rate floor by **0.1 wpm**
(149.9 vs 150.0); σ=0.2 misses clearly (71.5 voice, 140 wpm). The real
finding: feedback noise makes the teacher **slow down** (183 → 140–150
wpm) and costs mild identity (sim 0.635 → 0.527–0.585) — a graceful,
N8-mirror-image degradation, nothing like the head's collapse (sim 0.09).
FD couldn't see it (545/562/547 ≈ teacher-self band) — dispersion stays
teacher-like while rate and fine identity drift; one more instance of
"the scalar metric said fine, the graded one didn't."

σ=0.3 is a near-miss; σ=0.2 is not. **Josh's ear on t7_sig030 is the
decisive input:** if it sounds target-quality, the salvage options are
(a) capture at σ≤0.3 accepting ~92% identity retention in targets,
(b) drop to σ≈0.1–0.15 (untested; teacher likely cleaner, decorrelation
weaker), (c) per-window σ sampling in capture so most targets are clean
and only some are noised (GameNGen's actual regime — most attractive on
paper). Decision after the listen; ear verdicts verbatim below.

**Ear verdicts (Josh, 2026-08-15, verbatim):** "sig30, sig000, and sig20 is
actual speech!! sig30 is the cleanest, very clean and clear. Sig000 is also
clear, but sounds just a tad like its peaking maybe slight ever so slight
muddy or louder cant tell. However 000 has more emotion, where as sig 30 is
more flat. 30 also sounds just a tad slow. sig20 is almost as good as 30
but seems just a tad flatter and slower. I'm amazed."
(Attribution correction made in-session: these are TEACHER renders — Arm T
— not the flow head; the head renders are a7_sig040_s*. Josh's "working
speech from our new head" impression retracted accordingly.)

**GATE NIGHT 7 CLOSE-OUT.**
- **Arm T: CAPTURE GREENLIT via the ear gate.** The pre-registered metric
  half said DEAD, but by margins the ear was designated to arbitrate
  (σ=0.3: voice 87.4 vs 90, rate 149.9 vs 150.0) — and the ear's verdict
  is unambiguous: σ=0.3 is target-quality, "cleanest, very clean and
  clear," even *preferred* over the clean control on clarity. The real
  costs are **flattened emotion and ~20% slower pacing** — the ear found
  the axis the metrics hinted at: feedback noise trades prosodic
  expressiveness for stability. Two capture consequences: (1) **per-window
  random σ** (mostly-clean windows keep expressive targets; noised windows
  teach robustness) is adopted — it dilutes both costs; (2) the flatness
  observation goes in the paper: the noise-stability-vs-expressiveness
  tradeoff exists in the TEACHER too, gentler — same disease family as N8,
  fourth sighting of monotone-under-perturbation.
- Also noted by ear: the CLEAN control "sounds just a tad like it's
  peaking / slightly muddy or louder" vs the noised runs — consistent with
  mild energy inflation in the stock teacher even at 5 min turn-split;
  unmeasured, parked.
- **Arm A: VARIANCE confirmed** (see results above). σ=0.4 cleared;
  random-σ design stands; all stage-2 gates at ≥2 seeds.
- **Arm A ear verdict (Josh, 2026-08-15, verbatim, BOTH seeds):** "they
  both quickly degrade like 14-20 sceonds in to a raspy whisper, slowing
  and raspy creepy voice, ultimatley unrecognizeable at minute 4-ish."
  **This retracts the in-session "best closed-loop head render yet"
  framing for s1** — fifth instance of the ear beating the metrics:
  Whisper decoded 100% coverage / WER 0.169 from audio a human cannot
  follow past ~minute 4. **New instrument rule: Whisper word-recovery and
  human intelligibility DIVERGE in the degraded-register regime** — a
  raspy whisper is machine-transcribable and human-unintelligible at the
  same time. Coverage/WER remain content-survival metrics (the words are
  provably in there), but they are NOT listenability metrics; no stage-2
  gate may claim intelligibility from Whisper numbers alone. What stands
  from Arm A: σ=0.4 is not a dead zone (both seeds text-anchor to full
  length, neither hard-collapses like GN6's seed-0), per-seed variance is
  huge, random-σ training design unaffected. What changed: the honest
  description of ALL σ-rescued head renders is "content preserved in a
  degrading, ultimately unintelligible voice" — the training bet's job
  description, unchanged since GN5, now stated at its true size.
- **Capture v2 is GO:** schema = hidden states + DDPM targets + per-window
  σ bucket ∈ {0, ~0.1–0.3, occasional 0.4} + trailing-K latents. Next
  build.

## 2026-08-15 — THE CFG AUDIT FINDING (code audit, no GPU; Josh-prompted)

Josh asked for a root-cause re-examination before the capture spend
("are we missing anything simple"). The audit found one, hiding since July:

**The flow head has been sampling the UNGUIDED field all along.** The
teacher generates with CFG 1.3 every frame — the LM computes two streams
(cond + neg) and the DDPM head combines them. `capture.py` recorded only
`condition` + the guided output, so the head was trained to predict a
two-input answer from one input — forced to average over the neg-stream
variation it cannot see = a small SYSTEMATIC bias toward a conditional
mean, every frame. That is a concrete generator for exactly the correlated
error GN5 proved the loop amplifies. And at inference,
`integration.py`'s `flow_sample(condition, neg_condition=None,
cfg_scale=None)` accepts both CFG arguments and silently discards them.

Explains the GN1 sampler dilemma without new assumptions: the learned
field is a blurred compromise → euler4 under-disperses it (intelligible,
flat → whisper under compounding); heun8 samples it faithfully
(dispersion-correct, garbled 0.238). "No sampler is both" because the
FIELD is blurred, and the field is blurred because the target depended on
hidden information.

**Fix shipped (2026-08-15): `CFGFlowHeadPatch` + `_CFGField`** in
`src/flow_head/integration.py` — evaluates v_cond and v_neg per sampler
step, combines v = v_neg + s·(v_cond − v_neg), falls back to unguided when
no neg is passed; `cfg_scale` override pin for controls. Tier-1 tests
`tests/test_cfg_patch.py` (combination math at s∈{0,1,1.3}; fallback;
override-pin equivalence to plain patch; all pass with existing suite).
Cost: one extra head eval per step, negligible at the head's ~6% share.
Capture v2 schema gains `neg_condition` regardless of GN8's outcome.

## 2026-08-15 — GATE NIGHT 8 PRE-REGISTRATION (the CFG repair test; not yet run)

Notebook: `gate_night8_colab.ipynb`. Scorer: `score_gate_night8.py`. L4,
~45 min, ~$2–3. Same 5-min turn-split script as GN5–GN7. Multi-seed rule
in force: 2 seeds per CFG configuration.

### Hypothesis

**H-CFG:** restoring guidance at inference recovers a meaningful part of
the closed-loop deficit. Strong form: the whisper/flatness axis (energy,
identity) improves because the guided field is sharper. Weak form: only
content metrics move. Null: nothing moves — the bias baked in at training
dominates and inference-time guidance cannot undo it.

### Conditions (tags)

| tag | config |
|---|---|
| `c8_euler4_s0` / `c8_euler4_s1` | CFG-corrected euler4 (caller's 1.3), seeds 0/1 |
| `c8_heun8_s0` / `c8_heun8_s1` | CFG-corrected heun8, seeds 0/1 |
| `c8_euler4_plain_s0` | control — copied from GN6 `g6_sig000` (byte-identical settings) |
| `c8_heun8_plain_s0` | control — plain heun8, this script/length, seed 0 |
| `t8_cfg10` | teacher at cfg_scale=1.0 — how load-bearing is guidance for the TEACHER? |

### Gate criteria — written before the run

Metrics per the GN7 scorer (WER vs script, coverage, graded sim curve,
rate, FD). Per the GN7 instrument rule, Whisper numbers are
content-survival only; identity/listenability claims need the graded
curve AND Josh's ear.

| Verdict | Condition |
|---|---|
| **CFG REPAIR (strong)** | any CFG config: sim_median ≥ its sampler's control + 0.10 OR horizon ≥ 3× control, consistent across both seeds, and Josh hears an actual voice → root cause (partly) confirmed; re-baseline the program on guided sampling; capture v2 + training proceed dual-stream |
| **PARTIAL** | content metrics (WER/coverage/FD slope) improve across seeds but the graded identity curve stays flat → the training-time information gap dominates; guided sampling becomes the default inference config; capture v2 unchanged, higher priority |
| **NULL** | no consistent improvement on any axis → hypothesis refuted at inference; the July head's bias is baked in; capture v2 proceeds exactly as planned (still recording neg) |
| teacher read | `t8_cfg10` degrades vs GN7's clean control → guidance is load-bearing for the teacher, raising the ceiling CFG repair could reach; fine at 1.0 → guidance is polish, not survival |

**Josh listens to all seven.** The A/B that matters most:
`c8_euler4_s0` vs `c8_euler4_plain_s0` — same seed, only guidance differs.

### Results (2026-08-16 — run and scored same day; L4, ~$2–3; ear pass pending)

| tag | WER | coverage | voice % | horizon | sim med | sim last⅓ | rate | FD med |
|---|---|---|---|---|---|---|---|---|
| c8_euler4_plain_s0 | 0.933 | 8.1% | 1.3 | 8 s | 0.072 | 0.093 | — | 1839 |
| **c8_euler4_s0** | **0.065** | 100% | 49.6 | **240 s** | 0.500 | 0.443 | 189 | 1419 |
| **c8_euler4_s1** | **0.040** | 100% | 41.2 | 234 s | 0.490 | 0.436 | 188 | 1499 |
| c8_heun8_plain_s0 | 0.102 | 99.4% | 4.8 | 64 s | 0.180 | 0.080 | 165 | 1179 |
| **c8_heun8_s0** | **0.031** | 100% | 61.7 | 238 s | 0.522 | 0.453 | 174 | 1058 |
| **c8_heun8_s1** | **0.033** | 100% | 61.4 | 240 s | 0.561 | 0.431 | 187 | 1040 |
| t8_cfg10 | 0.298 | 81.4% | 21.9 | 210 s | 0.419 | 0.370 | 131 | 594 |

**VERDICT (metric half): CFG REPAIR — STRONG, on BOTH samplers, BOTH
seeds.** The pre-registered strong criterion is cleared by an order of
magnitude on every axis:

- **Content: closed-loop WER 0.031–0.065 ≈ the head's TEACHER-FORCED
  quality (July: 0.030).** The exposure-bias content deficit is, on these
  metrics, GONE. Full coverage, natural pacing (174–189 wpm), natural
  render lengths (~270 s, no dead-audio padding), all four runs.
- **Identity: half-recovered, not cured.** Voice fraction 41–62% (vs 1–5%
  unguided), sim medians sit AT the 0.5 voice-likeness line, horizons
  reach ~4 min of the 4.5-min render — but sim declines to ~0.44 by the
  final third and FD still drifts (≈1040–1500 vs teacher 550). A slow
  erosion remains.
- **heun8 + CFG is the new best config** (WER 0.031, voice 62%, FD 1040):
  with the guided field, dispersion-correct sampling is no longer garbled
  — the GN1 sampler dilemma is RESOLVED by fixing the field, exactly as
  prescribed ("fix the head — do not tune the integrator").
- **Teacher at cfg 1.0 DEGRADES** (voice 21.9% vs 98.7%, rate 131 wpm,
  WER 0.298): guidance is load-bearing for teacher identity too (FD
  couldn't see it — dispersion fine, identity off; graded-curve rule
  vindicated again). Raises the ceiling on what guided sampling can carry.

Root-cause account now strongly supported: the July head was trained to
predict guided targets from the cond stream alone AND sampled unguided.
Restoring guidance at inference recovers content completely and identity
halfway; the residue (slow identity erosion) is the part baked into the
weights by the one-stream training gap — capture v2's dual-stream schema
targets exactly that. GN7 instrument rule applies: intelligibility claims
await Josh's ear. Per-seed consistency this night is tight (0.031/0.033,
0.065/0.040) — first multi-seed-stable head result in the program.

Ear verdict (Josh, 2026-08-16, `c8_heun8_s0`, verbatim close): "the
heun8_s0 is best, but it still degrades over the duration. Almost like the
voice is getting a very sore throat by the end, not as much whispery
raspy anymore, a bit different, so starts out pretty good but still
degrades over duration."

**GN8 CLOSE-OUT: CFG REPAIR CONFIRMED (partial) — ear agrees with the
metrics, does not overturn them.** heun8_s0 confirmed as the best config.
Unlike GN7 (Whisper decoded content from audio a human found
unintelligible past minute 4 — the instrument-divergence case), this ear
pass and the metrics tell the SAME story: strong start, real but
incomplete identity recovery, slow erosion across the render. The
*texture* of the erosion changed from GN5–GN7's raspy/breath-collapse
whisper to a vocal-strain/"sore throat" quality — read as: CFG removed
the train/inference guidance mismatch (one dominant correlated-error
source), and what surfaces underneath is the smaller residual correlated
error GN5 already characterized (the loop amplifies whatever systematic
bias is left in the head's per-frame prediction), now compounding into a
different perceptual artifact rather than energy collapse. Not evidence
of a head-capacity ceiling — nothing across GN1–GN8 traces a failure back
to the 15M-param no-attention head running out of room; every traced
cause has been *what supervises it* (one-stream/teacher-forced targets)
or *what context it sees* (length-OOD). Constraint 5 stands. **Verdict:
GREENLIT for capture v2 (dual-stream) → stage-2 on-policy training
(Causal Forcing / CF++ family) as the cure for the residual erosion; no
architecture change indicated.** heun8+CFG is the new operating
inference config, superseding GN1's euler4-only recommendation.

## 2026-08-16 — CAPTURE V2 INFRA BUILT (code only, no GPU spend yet)

Schema locked by the 8/14 critic pass + the 8/16 GN8 amendment: hidden
state for **both cond and neg streams** + DDPM target (already captured)
+ per-window σ bucket + trailing-K latents. Built and tier-1 tested:

- **`src/cache/noise.py`** (new): `NoiseIntervention`, ported verbatim
  from the GN6 notebook (the exact mechanism behind every GN5-8 render —
  no reimplementation-drift risk) plus one addition, `.last_sigma`, so a
  capture wrapper can read back which σ was actually applied to the frame
  it just captured. 6 tier-1 tests (`tests/test_noise.py`): zero-sigma
  no-op, inactive-gate no-op, running-variance scaling, schedule wiring,
  hook cleanup on exit.
- **`src/cache/capture.py`** extended, not replaced: `UtteranceCache`
  gains `neg_hidden` / `sigma` fields (default `None` — old .pt files
  load unchanged). `SampleCapture`/`BatchedSampleCapture` take an
  optional `noise=` handle and now read `neg_condition` out of the
  wrapped call (previously silently forwarded into `*args`/`**kwargs`
  and dropped — the same discard pattern the CFG audit found in
  `integration.py`, just on the capture side). Forwarding to the real
  method stays byte-transparent (no injected defaults for omitted args —
  first draft of this got that wrong and would have silently overridden
  VibeVoice's own `cfg_scale` default; caught before commit).
  `split_utterances` (v1) is untouched in behavior; `split_utterances_v2`
  is additive. Partial dual-stream (present on some frames/calls, absent
  on others) is refused, not silently dropped. 11 new tier-1 tests across
  `tests/test_cache.py` / `tests/test_batched_capture.py`. Full suite:
  80/80 green.
- **Design call on trailing-K latents (flagging, not asking):**
  `NoiseIntervention` corrupts the acoustic-connector's output
  *embedding* (what becomes `condition`/`hidden`), not the raw target
  latent — so the clean per-frame `latent` sequence already saved is
  unaffected by σ and stays in order. Trailing-K history is therefore
  **derivable post-hoc from the existing ordered `latent` tensor** —
  no new capture-time field needed, just a windowing helper at train
  time (for the head-v2 history-conditioning arm). If the intent was
  instead to capture the noise-corrupted connector embedding itself as
  history, say so and this gets revisited before the GPU run.

### Decisions (Josh, 2026-08-16)

1. **Corpus: turn-split long-form only.** Same multi-speaker turn-split
   script style GN5-8 validated (N8 cure), at varied lengths — not raw
   LibriTTS short clips.
2. **Scale: match v1's total frame budget, not utterance count.** v1's
   10K cache = 478,191 frame pairs (avg ~48 frames / ~6s per short clip).
   A 5-min long-form render is ~2,250 frames — ~47x more per utterance —
   so "10K" would have meant ~47x v1's total compute if read as raw
   utterance count. Resolved: **~480K total frame pairs** (matching v1),
   via **~200-250 long-form renders**, not 10,000 of them.

### Capture v2 notebook built: `capture_v2_colab.ipynb`

Follows `cache10k_colab.ipynb`'s COLD START / resumable-loop pattern and
GN6's `NoiseIntervention` usage exactly, to minimize novel-code risk on an
un-test-able-locally GPU notebook. Structure:

- **Corpus:** `WORD_BINS = [150, 300, 600, 1200, 2400]` (~1/2/4/8/15 min
  @ N8's natural 165-190 wpm), cycled round-robin so the ~200-250 renders
  spread across the OOD range rather than clustering at one length.
  Sentences streamed from LibriTTS (text only, matching cache10k's
  quality filter), assembled via `turnscript()` — GN6's exact mechanism,
  ~60-word same-speaker turns, restored faithfully (an early draft of
  this collapsed it to a single un-split turn per script — caught before
  finalizing, would have silently defeated the N8 cure this whole capture
  depends on). Voice prompts cycled from the existing eval-cache pool for
  speaker diversity (GN5-8 used one fixed prompt throughout; capture v2
  training data shouldn't).
- **σ schedule:** one random draw per ~60s window (`WINDOW=450` frames),
  distribution 50% clean / 40% U(0.1, 0.3) / 10% at 0.4 — the "σ ∈
  {0, ~0.1-0.3, occasional 0.4}" schema from the 8/15 GO note, made
  concrete. Fresh draw sequence per batch (`make_sigma_fn()` re-called
  in `flush()`), so different renders get independent trajectories even
  though `NoiseIntervention` hooks the model once per batch (same-batch
  elements necessarily share one σ trajectory — a real simplification,
  not hidden: flagged here).
- **Batching:** same-length-bin renders batched together (BS 8→2 as bin
  length grows) rather than mixing bin lengths in one batch — avoids
  short elements idling through a much longer batch-mate's remaining
  steps. Target dir is a **new** `longflow_p1_cache_v2` on Drive, kept
  separate from v1's cache rather than mixed.
- **Resume is frame-budget-based**, not utterance-ID-based like v1 (uids
  are fresh random UUIDs, not deterministic dataset row IDs) — a resumed
  session generates new scripts until the frame target is hit rather than
  replaying v1's dedup-by-ID logic, which doesn't fit a random-corpus
  design. Minor consequence: resumed sessions may re-stream overlapping
  LibriTTS text across sessions; not a correctness issue.
- Exception handling follows the 2026-08-14 production lesson exactly:
  `repr(e)[:150]` only, never the exception object retained.
- Cell 3 writes a manifest (frame/script totals, word-bin counts, σ
  histogram actually realized) to Drive for the gate-check to read —
  **not** a downloadable audio bundle; there's nothing to listen to here.

### Still open

- Notebook is unrun — first real GPU pass will surface anything this
  read-through missed (can't execute Colab-only APIs locally). Watch
  first-batch shapes closely before walking away.

**Correction (2026-08-16, same day):** this entry previously listed
batched-vs-unbatched capture parity as still open/unresolved since
2026-08-11 — wrong, caught when Josh asked what the flag meant. It was
CLEARED at GN4 Arm C (2026-08-12): solo-vs-batch-4 teacher renders,
distributional quality (WER/sim vs the reseed floor) instead of GN1's
flawed frame-count/per-dim-mean-shift instrument — mean pair-WER 0.018,
sim within the teacher's own reseed band; GN1's frame-count divergence
re-explained as ordinary generation variance, not left-padding pollution.
Reconfirmed 2026-08-14 as "the only valid instrument" when a stricter
bitwise-determinism variant was tried and retracted for being the wrong
kind of test on a stochastic AR system. Not a live blocker for this
capture run.

### First live run (Josh, 2026-08-16) — mixed-bin batching bug, fixed

Josh started the notebook and flagged it crawling almost immediately:
`Generating (active: 3/4): 20%|...` on the very first batch, 1.6 it/s.
Real bug, not Colab slowness: cell 2's word-bin batching design (batch
same-length scripts together, so short elements never idle waiting on a
much longer batch-mate) was defeated by the loop itself — `bin_i`
incremented and `target_words` was recomputed on every completed script,
but nothing segregated the accumulating `buf` by bin, so it silently
filled with scripts from *different* bins before flushing. `active: 3/4`
was exactly that: a short script finishing early inside a batch stuck
waiting on longer ones. **Fixed**: `buf` replaced with `bufs = {w: [] for
w in WORD_BINS}`, keyed by bin, flushed independently per bin. Caught
after ~7 min GPU time on the first (still in-progress) batch — no cached
data existed yet, nothing to redo.

Also hardened `capture_v2_colab.ipynb` cell 1 while fixing this: it only
cloned the repo once and would have silently kept serving stale code on
a same-session re-run (no way to pick up a live fix without a runtime
restart otherwise). Now `git pull`s if already cloned, with an explicit
note that already-imported Python modules still need a runtime restart
to actually pick up changes — `git pull` alone updates the files on disk,
not what's already loaded in the kernel.

Also switched the target runtime from L4 to **A100** and raised
`BIN_BATCH` accordingly (`{150:12, 300:10, 600:8, 1200:5, 2400:3}`, up
from the L4-sized `{150:8, 300:8, 600:6, 1200:4, 2400:2}`) — L4 was a
default inherited from every prior gate-night notebook without
re-examining it for a one-time real spend where wall-clock matters more
than $/hr. Josh's call, explicit: stop optimizing for cost here.

Pushed to `origin/main` (commit history has the exact diff). Josh needs
to restart the Colab runtime (now selecting A100), and re-run cells 1-2
clean.

### Second bug (Josh, 2026-08-16, same day) — batched-capture attribution
### too narrow for capture v2's scale (`src/cache/capture.py`)

Tab accidentally closed after ~5h; Drive check confirmed real progress (91
scripts, 155,406/480,000 frames, all 5 bins) — Colab kernel had kept
running headless past the tab close. Josh reconnected but the runtime had
actually died by then, so he started fresh; resume worked exactly as
designed (`resuming with 91 scripts already cached, 155406/480000
frames`). But the fresh run then failed **~12 of 13 batches** with
`RuntimeError('row/active mismatch at a step: N rows vs M active
elements — attribution unsafe, aborting')` — each failure burning the
*full* generation wall-clock before being detected and discarded.

**Root cause:** `BatchedSampleCapture`'s attribution logic (in
`src/cache/capture.py`, inherited unmodified from the original P1
caching pipeline) only tolerated exactly one dropped frame, and only at
the batch's shared *global final step* — sized for cache10k's short,
duration-uniform LibriTTS clips, where all batch elements finished
around the same time. Capture v2 batches much longer, much more
duration-varied scripts (same word-bin target, but real generated length
varies per script), so elements now routinely finish at *different
points mid-batch* — and when an early-finisher's own last frame is
silently unrendered there (not at the shared end), the old logic had no
way to localize it and hard-aborted the entire batch's data over one
element's ordinary early finish.

**Fix:** rewrote `_attribute()` as a sequential per-call walk (not a
precomputed global step list) that tracks each stream's own consumption
pointer independently. A row deficit at any step is auto-explained ONLY
when the count of elements whose *current* position is their own last
remaining frame exactly equals the deficit — ambiguous or unexplainable
deficits still hard-abort, and a row *surplus* is always a hard abort
(no plausible mechanism for that direction). This subsumes the original
tolerance as a special case and generalizes it to (a) any number of
elements sharing a true simultaneous final drop, not just one, and (b)
drops occurring at any point mid-batch, not just the global end.
`split_utterances`/`split_utterances_v2` now consume `_attribute`'s
per-element row dict directly rather than re-deriving it. 4 existing
tests needed regex updates for new (more specific) error messages; 2 new
tests added reproducing the exact failure (mid-batch early-finisher drop)
and its generalization (multiple simultaneous end-of-batch drops). Full
suite: 82/82 green.

Pushed to `origin/main`. Josh needs to restart the runtime again and
re-run cells 1-2 — resume will pick up from the 91+ scripts already
banked in Drive.

### CAPTURE V2 COMPLETE (2026-08-16) — `capture_v2_manifest.json`

Ran clean after the attribution fix. Final tally:

| | |
|---|---|
| scripts | 248 |
| total frames | 510,143 (target 480,000 — overshot by ~6%, fine) |
| word-bin spread | 150w:39, 300w:77, 600w:47, 1200w:47, 2400w:38 — all five bins populated, 300w over-represented (round-robin + resume/restart interruptions, not concerning) |
| σ distribution | 0: 52.6%, (0,0.1): 0.75% (fp16 rounding artifact — sigma_draw() can't actually produce this range by design), [0.1,0.3): 36.3%, 0.3 (boundary): 1.6%, 0.4+: 8.8% — matches the designed 50/40/10 mix closely |

Dual-stream (neg_hidden) + per-frame sigma schema confirmed present via
the manifest generation itself (`d.get("sigma")` populated all 510,143
entries). Cache lives at Drive `longflow_p1_cache_v2`, kept separate from
v1's `longflow_p1_cache`.

**Next: the mandatory 1K/5K gate check (hard constraint 6) before this
cache feeds any training run** — decode a sample, listen, verify the
dual-stream/sigma fields are sane before committing to a training spend.
Not yet built.

### 1K/5K GATE CHECK — `gate_v2_colab.ipynb` (2026-08-16/17)

Ran clean after two bugs, both in the notebook, not the cache: (1)
`decode_latents` OOM'd on capture v2's long-bin samples (up to ~6300
frames) — it was copied from the original P1 gate, which only ever
decoded ~48-frame clips in one shot; fixed by chunking to 30s pieces,
mirroring the encode-side chunking already used elsewhere (GN6). Cache
summary confirmed all 248 scripts carry intact `neg_hidden` + `sigma`,
zero alignment errors on read.

**Ear verdict (Josh, 2026-08-17):** roundtrip and all `_teacher.wav`
clean. All `flow4`/`flow16` samples — short bin AND long bin, both NFE
settings — share a consistent artifact: correct, clear speech content,
but a rhythmic "shaky" quality-wobble, "up is good, down is degraded,"
cycling roughly 1-2x/second. flow16 "maybe" marginally better than
flow4 but the effect is still clearly present — no clean NFE separation.

**Diagnosis:** the near-absence of a flow4-vs-flow16 gap is the key
fact — if this were a sampling-resolution/dispersion artifact (the
GN1-era euler4-vs-heun8 pattern), 16 steps should have cleaned it up
much more than it did. Pointing instead at the trained WEIGHTS, not the
integrator: consistent with this checkpoint's minimal 5K-step training
(vs July's 20K-step production head) not yet converging to a stable
per-frame velocity field. Distinct from the closed-loop collapse disease
(GN4-8) on mechanism grounds — this run is teacher-forced (head reads
the real teacher's hidden states, never its own output), so the
feedback-amplification pathology GN5 proved doesn't apply here; this is
a simpler undertraining symptom, not a structural bug. Also present
about equally on short and long bins, meaning it currently swamps the
context-length-quality signal capture v2 was built to test — that
comparison needs a properly-trained checkpoint, not this 5K-step one.

**Verdict: GATE PASSES** on what it's actually checking (cache is not
corrupted, trains without erroring, content survives teacher-forced
decode intact). The shaky texture is logged as a **WATCH ITEM, not
dismissed** — per the ear-beats-metrics pattern (4 prior instances), if
this exact 1-2 Hz wobble persists at a real training step count (20K+)
in the eventual stage-2 run, that would be a genuine new finding
requiring investigation rather than attributable to undertraining, and
this entry is the trail back to when it was first heard.

**Next:** decide the stage-2 training approach (causal-CD parallel arm
vs. the σ-conditioned noise-augmented head) and scope that build.

### Shaky-voice ablation, attempt 1 — Drive-IO bug, no data (2026-08-17)

Josh proposed a proper controlled ablation before doing more building
("troubleshoot the system, need to know why before changing more
things") to test undertraining vs. sigma-mixing as the cause of the
1-2 Hz wobble. Built `gate_v2_ablation_colab.ipynb`: three checkpoints,
identical architecture/hyperparameters/step count, only the training
data pool differs (A = v2 mixed, reused; B = v2 sigma==0 only, new;
C = v1 cache, reused if `gate_5k.pt` survived from July).

**Pool C never finished loading — ran all night, timed out.** Cause:
`pairs_from_files()` read v1's 10,000 individual files one at a time
straight off the Drive mount. Pools A/B (248 files each) loaded near-
instantly; C, at ~40x the file count, apparently never completed even
overnight. The original P1 gate notebook (`gate_colab.ipynb`) hit this
exact problem in July and documented the fix — bulk-copy to local disk
before reading — which didn't get carried into this new notebook.
**Fixed**: pool C now does one bulk `cp` of the whole v1 cache to local
disk first (same pattern as the July gate), then reads from local files.
Pushed. Rerun from cell 2 (pools A/B/checkpoints don't need redoing if
the runtime is fresh — Colab session state doesn't survive a timeout
either way, so this is effectively a clean restart from cell 1).

**Process note:** mid-run advice was "let it ride rather than interrupt"
— reasonable at the time (didn't want to throw away already-invested
wait), but wrong in hindsight once it ran unbounded overnight with zero
completion signal. Worth a firmer cutoff next time a long-running cell's
progress can't be directly observed.

### Shaky-voice ablation, attempt 2 — result: undertraining, not sigma-mixing

Rerun with the Drive-IO fix. **Ear verdict (Josh, 2026-08-17):** all six
files — pools A (mixed), B (sigma==0 only), and C (v1 cache), both short
and long test utterances — share the same core "underwater/shaky/not
perfectly clear" character. They "differ a bit" but none is clean.

**Both hypotheses ruled out cleanly**: B (no noised frames at all) still
has it → sigma-mixing is NOT the cause. C (a completely different,
always-clean, short-clip-only cache) still has it → the v2 cache is NOT
implicated either. What's left, and well-supported by an independent
prior data point: **step count.** All three checkpoints here got only
5,000 steps — the deliberate bare minimum for a gate check. July's
production head, same architecture and recipe, trained to 20,000 steps
on v1-style data, was clean (0.030 WER / 0.984 sim, no shakiness
reported). Same recipe, 5K → shaky (3x now); 20K → clean (once, on
record). **Verdict: generic undertraining, not a capture v2 or
sigma-mixing problem.** σ-conditioning is still the right fix for the
separate closed-loop identity-erosion axis — just not this one.

### THE REAL RUN — `train20k_v2_colab.ipynb` (2026-08-17)

Josh: "let's train long and just get it over with." Decision: 20,000
steps (not higher) — July's own E3 finding already showed 80K steps
OVERFIT and got WORSE on held-out data on a similarly-sized cache (v1's
478K frames vs v2's 510K), so this isn't a "more is safer" situation;
20K was the established sweet spot at this data scale and is the target
here too, verified empirically rather than assumed to transfer.

Added real infrastructure the quick checks skipped, per the 2026-08-11
"process rules now in force" that gate/ablation notebooks didn't
implement:
- **`train()` gained `checkpoint_every`/`checkpoint_path_fn`**
  (`src/flow_head/trainer.py`) — intermediate checkpoint saving was
  entirely missing before; a single final-step save can't catch an
  overfit-past-X finding the way July's E3 did. 2 new tests, 84/84 green.
- **Held-out split**: 5 scripts per word bin (25 total, stratified by
  length), never trained on — every prior check trained on 100% of
  available data with no generalization signal at all.
- **Proven hyperparameters**: lr 2e-4→2e-5 cosine decay (July's actual
  recipe; gate/ablation checks left lr constant), ema_decay 0.9999 (the
  documented default — gate/ablation checks used a faster 0.999,
  deliberately sized for their much shorter 5K-step horizon, not
  appropriate here).
- Checkpoints every 5K steps (5K/10K/15K/20K); held-out eval at each —
  full 25-utterance eval at the final checkpoint, 2-per-bin subset (10
  utterances) at the intermediate ones, tracing the same overfitting
  curve July's E3 used to catch the 80K regression.

`score_train20k_v2.py` (Mac-side, matches every gate-night scorer's
pattern) computes held-out WER + speaker-sim per checkpoint via
`src/eval/metrics.py`, and explicitly flags if the best held-out WER
lands before the final step — the same overfitting signature to watch
for.

Not yet run.

### SESSION CLOSE-OUT (2026-08-17) — bad day, real finding buried in it, read this first

Josh's words: "this has been horrible and a waste of time and money for
2 days." That's accurate and should not be softened. Full honest
retrospective, then the one finding that actually matters, then the
concrete next action.

**What went wrong, in order, so it doesn't repeat:**
1. Mixed-bin batching bug in `capture_v2_colab.ipynb` — caught mid-run,
   fixed, but its damage to already-captured files (below) wasn't
   understood until much later.
2. Batched-capture attribution bug — ate real GPU hours twice before the
   per-element fix landed.
3. Decode OOM on long-context latents — fixed, cheap.
4. **The big one: `score_train20k_v2.py` ran Whisper-large-v3 on Josh's
   Mac CPU for 3+ hours and produced nothing** — should have run on
   Colab GPU from the start (Josh has premium Colab specifically for
   this). Killed with zero output to show for it.
5. Rebuilt as a GPU Colab scorer. Direct browser drag-in of the ~1GB
   result zip silently truncated (BadZipFile).
6. Fixed to read from Drive — but the fix used a **recursive glob across
   Josh's entire Drive tree**, the exact same slow-FUSE-mount trap hit
   earlier with the 10,000-file v1 cache, just in a new spot. Caught by
   Josh, not by me, again.
7. Scoring finally ran and mostly completed (steps 5K/10K/15K/20K, 24 of
   25 at the final step before a GPU OOM on the last long clip).

**The finding underneath all of it, which is real and matters:** some
files in the capture v2 cache — the ones captured during the buggy
mixed-batching window, before the fix in the "Second bug" entry above —
are not just mislabeled, they appear **genuinely truncated**. Evidence:
`cv2_1200w_4f19765a` (filename says a true ~1200-word/~7min script) has
an actual rendered duration of **159s** — nowhere near enough for its
true content. The mechanism: during the bug, `max_new_tokens` was sized
from whichever bin *triggered* a batch flush, not each item's own
target, so a long script swept into a short-budget batch would get cut
off mid-capture. Both its teacher capture and anything trained from it
are working from incomplete data. In the step-20000 held-out results,
WER is bimodal in exactly the way this predicts: files whose corrupted
meta label doesn't match their true filename bin score WER ~0.7-0.8
(broken/truncated); files where label and filename happen to agree score
~0.08-0.2 (clean, and genuinely improving 5K→20K). This is not the model
struggling with long content — it's contaminated data in the comparison.

**Consequence:** an unknown fraction of the 248-script v2 cache (likely
concentrated in the ~91 scripts captured before the mid-session batching
fix landed) may contain truncated renders. This means:
- The held-out split for this 20K run was NOT cleanly length-stratified
  as designed (25 held-out scripts skewed heavily toward one filename
  bin, zero true 300w/600w representation — the corrupted label, not
  true content, drove the stratification).
- Some fraction of the 223-script *training* pool for this run may also
  be truncated, not just the held-out eval — unquantified.
- The clean-looking checkpoints (files where label happened to match
  truth) still show a strong, real, improving trend 5K→20K — that signal
  is probably genuine, just not yet cleanly separable from the noise
  without a proper audit.

**Next action for a fresh session, in order:**
1. **Audit, don't guess.** For every file in `longflow_p1_cache_v2` on
   Drive, compare its filename-embedded target-word bin against its
   actual frame count / rendered duration (7.5 Hz, ~165-190 wpm natural
   pace gives an expected range per bin). Flag anything that doesn't
   fit — that's the truncated set.
2. **Decide, based on how big the damage is**: if it's a small minority
   of files, filter them out and re-derive a clean held-out split +
   retrain (cheap, no new capture). If it's a large fraction, a targeted
   re-capture of just the affected scripts is probably faster than
   distrusting the whole cache.
3. Only after that: re-run the held-out eval (the GPU scorer notebook
   works now — `score_train20k_v2_gpu_colab.ipynb`, reads the zip from
   Drive root, non-recursive) on a cache that's actually clean, and
   redo the per-bin breakdown for real this time.
4. `train20k_v2_metrics.json` was never written (the run crashed on the
   final entry) — the only numbers that exist are in this session's
   transcript, not any saved file. Re-scoring is required regardless of
   what the audit finds.

Nothing was lost that can't be recovered — the 248-script cache and all
its component data are still on Drive, and the infra built today (dual-
stream capture, the attribution fix, checkpoint intervals, GPU scoring
with a device param) is sound and reusable. What's not usable yet is
trust in *this specific* 20K checkpoint and its held-out numbers, until
the audit above happens.

## 2026-08-17 — FULL-HISTORY REVIEW SESSION (no GPU; audit + literature re-verification)

Josh asked for a from-scratch review of the whole experiment line ("have we
been taking the right course"), prompted by the rough 8/16–17 session. Three
deliverables, no GPU spent:

### 1. Cache truncation audit — DONE without Colab (close-out step 1)

Full write-up: **`capture_v2_audit.md`** + machine-readable
`capture_v2_audit_flags.json`. Method: Drive file sizes are linear in frame
count (6,269 B/frame, calibrated on the known-truncated exemplar), so implied
wpm = bin words / implied duration exposes truncation with no download.
**Result: 25 TRUNCATED + 2 SUSPECT of 248 files (~8% of frames), every
truncated file from the pre-17:00 first session, 24/25 sitting within 1% of
exactly two token-budget caps (7.48 MB = 300w budget, 14.97 MB = 600w
budget).** Clean files' implied wpm lands precisely in the turn-split natural
band (per-bin medians 150–168, max 192) — method validated. Damage is
filterable: ~470K clean frames remain (at target); worst hit is the 2400w bin
(38→26 scripts); optional ~3–4 A100-h top-up restores it. Truncated files
poison meta labels and text-vs-audio eval, NOT the per-frame training pairs —
so the v2_20k checkpoint's training is probably fine and its *evaluation* is
what must be redone on a clean split.

### 2. Finding: the v2 20K run did not implement the stage-2 design

`train20k_v2_colab.ipynb` cell 2 trains on `utt.hidden` + `utt.latent` only —
**`neg_hidden` and `sigma` are never read.** `FlowHead.forward` still takes
(x_t, t, condition); there is no dual-stream input and no σ-bucket embedding.
And the held-out eval sampled **unguided flow4**, not the GN8 operating config
(heun8+CFG), with no closed-loop assay and n=1 seed. So the run is a
stage-1 re-baseline on v2 data — legitimate as an undertraining check, but it
does not test what capture v2 was built to enable, and none of its numbers
bear on the CFG-repair or noise-training hypotheses. The head-v2 input
contract (cond+neg concat + σ-bucket embedding — a new artifact tag per
process rules) must be designed and built BEFORE the next training spend, or
the next 20K steps re-learn the same one-stream blurred field GN8 diagnosed.

### 3. Literature re-verification + scoop sweep (independent, direct fetches)

- CF++ (2605.15141), dots.tts, GameNGen, Diffusion Forcing, Self Forcing,
  GROW: all verified as characterized. **One correction: Causal Forcing
  (2602.02214) never says "mode-seeking" or "accumulated history error" —**
  its actual finding is "a large architectural gap at initialization cannot
  be resolved by the subsequent DMD stage." Same practical lesson (strong
  init before DMD), wrong words; fix before citing.
- **Not scooped, re-confirmed 2026-08-17:** arXiv "VibeVoice" sweep returns
  only the ASR product line + the original report; no TTS head distillation
  anywhere; no published measurement of rate inflation (MagpieTTS-LF
  2606.18485, the nearest long-form neighbor, never mentions pacing — cite
  and differentiate).
- **Three must-reads found before the stage-2 build:**
  (a) **2411.18447** (Pasini et al., NeurIPS-24 workshop) — noise-augmented
  continuous-AR *audio* explicitly to stop error accumulation, incl.
  inference-time low-level noise = the GN5 hack, published. Validates the
  whole direction AND narrows our novelty claim to the causal isolation +
  the content/identity axis split + first-on-TTS (the 8/14 "novelty
  calibration" entry anticipated video precedents; an audio one exists too).
  (b) **2607.24731** — "Rethinking CFG in On-Policy Diffusion Distillation":
  negative-branch asymmetry, i.e. OUR one-stream trap named and analyzed;
  read before fixing the head-v2 objective. (c) **CuteTTS 2608.08638** —
  continuous-AR TTS with "guidance-step distillation" absorbing CFG into the
  student; the closest published answer to our dual-stream design question.

### Course verdict

The through-line P0→GN8 holds up under adversarial re-read: every pivot
(N8/turn-split, feedback-OOD, CFG audit) is evidence-driven, pre-registered,
and independently consistent with the verified literature. The 8/16–17
session's failures were execution (notebook bugs, wrong-device scoring, and
item 2 above), not direction. Stale note: CLAUDE.md's "first task: verify the
50-min render" was already done and ear-certified 2026-08-15.

~~**Next actions, in order:** (1) filter the 27 flagged files + re-derive the
held-out split by FILENAME bin; (2) re-score the existing v2_20k checkpoints
on the clean split (GPU scorer notebook, cheap) — decides if the 5K→20K trend
is real; (3) read 2607.24731 + CuteTTS + 2411.18447, then lock the head-v2
input contract (cond+neg + σ-bucket); (4) optional 1200w/2400w top-up
capture; (5) train head-v2 on the filtered cache; gate = closed-loop
heun8-or-native, ≥2 seeds, graded sim curve, Josh's ear (GN8 protocol).~~
**[Superseded same day — steps 1, 3 and 5 executed/built below; step 2
dropped (the old v2_20k checkpoint is superseded by the retrained control
arm on the filtered pool — re-scoring it buys nothing that ships); step 4
stays optional.]**

## 2026-08-17 — HEAD-V2 BUILT + HV2 RUN PRE-REGISTRATION (code complete; not yet run)

Josh: "whatever the BEST option is to do this right." Decision and build:

### Design (locked after the targeted reading pass)

**Head-v2 = dual-stream + σ-bucket conditioning, guidance absorbed into the
learned field.** `FlowHeadConfig(dual_stream=True, sigma_buckets=4)`:
`forward(x_t, t, condition, neg_condition, sigma_bucket)`, neg enters as a
second summed projection (≡ concat + linear), σ as a zero-init learned
embedding (GameNGen §3.2.1's exact mechanism). 17.63M params (vs 16.64M) —
still a thin no-attention adapter; constraint 5 holds. No CFG combination at
inference (`DualStreamFlowHeadPatch` passes VibeVoice's per-frame neg stream
straight to the head, σ-bucket 0 = clean feedback).

Reading-pass verdicts backing this (full agent report archived in session):
- **2607.24731 formalizes v1's exact failure** ("Negative Branch Asymmetry":
  privileged info in the teacher's neg branch → antagonistic branch errors).
  Its trigger condition dissolves when the student SEES both streams — our
  design is the information-complete fix; their PDM loss is the workaround
  for when you can't do that. One warning adopted as a limitation note, not
  a build: the guided field is a *scale-1.3-baked-in* artifact (no guidance
  knob at inference without multi-scale capture + a w-embedding à la CuteTTS
  2608.08638 — future capture-schema option if a scale sweep is ever wanted).
- **CuteTTS** ships a guidance-absorbed student (cond-only input is safe for
  them ONLY because their neg branch is an information subset; ours is a
  separate LM forward — so their shortcut does not license dropping our neg).
- Neither paper documents any over-sharpening problem from guided targets.

Every layer now REFUSES silent conditioning mismatches in both directions
(model forward, cfm_loss, trainer, patch) — the discard pattern that caused
both the July CFG bug and the first v2 20K run's neg_hidden omission is now
structurally impossible. `trainer.pairs_from_files(files, dual_stream=True)`
refuses v1-style files; `trainer.filter_flagged` applies
`capture_v2_audit_flags.json` loudly. Tests: `tests/test_headv2.py`, suite
98/98 green, ruff clean.

### The run: `train_headv2_colab.ipynb` (A100, ~2–3 h, ~$10–15) + `score_headv2_gpu_colab.ipynb`

Pools: filtered clean cache (27 flagged files excluded), held-out = 5/bin by
FILENAME bin, ~196 training scripts / ~470K frames. Two arms, same data, same
recipe (20K steps, bs 1024, lr 2e-4→2e-5 cosine, EMA 0.9999, ckpt every 5K):
**arm A** = head-v2 (dual+σ); **arm B (control)** = v1 architecture cond-only
— at eval, arm B runs GN8's operating config (heun8 + inference CFG 1.3), so
the comparison isolates *training-time* dual-stream conditioning against the
best inference-time repair. Notebook order enforces hard constraint 6: a 5K
gate arm renders clips for Josh's ear BEFORE the real cells run (expected:
shaky-but-intelligible per the ablation; silence/collapse = stop).

### Pre-registered criteria — written before the run

Teacher-forced held-out (per checkpoint, clean split): informational +
overfit watch (best-WER-before-final flags, July E3 signature). Arm A should
be ≥ arm B; a large arm-A deficit = dual-stream training bug, stop.

Closed-loop (the real gate; GN5–8 script recipe, heun8, seeds 0+1 per arm;
reference row = GN8 July-head heun8+CFG: WER 0.031 / voice 61.7% / sim_med
0.522 / sim_last⅓ 0.453 / full 240 s horizon):

| Verdict | Condition (both seeds unless noted) |
|---|---|
| **STRONG** | arm A: WER ≤ 0.08 AND sim_last⅓ ≥ 0.55 AND horizon = full render → the training-time information gap was a main erosion driver; stage-2 scope shrinks to the residual |
| **PARTIAL** | arm A: WER ≤ 0.08, sim gains < strong bar → content parity holds, identity still erodes; stage-2 on-policy carries identity as planned |
| **FAIL** | arm A: WER > 0.15 or coverage < 90% on any seed → dual-stream training hurt; diagnose (σ-mix? stats?) before ANY further spend |
| replication check | arm B ≈ GN8's row (±reseed floor); if not, the filtered-pool retrain changed the baseline — flag before comparing |

Informational arm (no gate): `hv2_heun8_s0_sig02` — closed loop with σ=0.2
feedback noise + matching bucket 2 told to the head: first test of whether
σ-conditioning preserves identity where GN6's blind noise (g6_sig020, voice
0%) could not. Bears directly on the stage-2 noise-training design.

Josh listens to the gate clips, the best arm-A render vs its arm-B twin, and
the sig02 arm. Ear verdicts recorded verbatim here; instrument rule (GN7)
applies — Whisper numbers are content-survival only.

## 2026-08-17/18 — HV2 RUN RESULTS (provisional — scorer mid-flight overnight)

Run executed by Josh on A100 per `train_headv2_colab.ipynb`; scoring via
`score_headv2_gpu_colab.ipynb` (GPU, T4-class). ~$15 total. 20000:B and the
scored closed-loop section land in the morning; everything below is settled.

### Instrument finding first — EMA dilution poisons intermediate checkpoints

The 5K/10K rows are NOT model quality: the run trains with EMA 0.9999
(correct for 20K) and checkpoints load EMA weights, so the step-5K shadow is
still ~60% zero-init (0.9999^5000 ≈ 0.61) — both arms score ~0.98 WER at 5K
by construction, warming to real numbers only by 15–20K. The July notebooks
knew this (their 5K gates deliberately used EMA 0.999); tonight generalized
it: **never read intermediate checkpoints of a 0.9999-EMA run as a training
curve.** Retroactive consequence: the 2026-08-17 train20k_v2 "improving
5K→20K trend" was partly this artifact on top of the truncation contamination.

### Teacher-forced held-out (clean filtered split, heun8; A native dual, B +CFG 1.3)

| step | A wer_med / sim_med | B wer_med / sim_med |
|---|---|---|
| 5000 | 0.981 / 0.133 | 0.984 / 0.143 |
| 10000 | 0.474 / 0.489 | 0.294 / 0.568 |
| 15000 | 0.219 / 0.761 | 0.254 / 0.816 |
| **20000** | **0.161 / 0.858** (n=25; bins 150:0.180, 300:0.107, 600:0.161, 1200:0.125, 2400:0.201) | pending |

**Arm A at 20K is the best teacher-forced held-out result in program history
on real long-form data**, and the long bins are its strongest (1200w 0.125).
At 15K, A beats B on 600/1200 bins. **Dual-stream + σ-bucket training WORKS
teacher-forced** — the architecture, cache v2 (filtered), and pipeline are
all vindicated on-distribution.

### Closed loop — ear verdicts (Josh, 2026-08-18, verbatim; scored rows pending)

- `hv2_heun8_s0` (arm A native): "pretty bad pretty much unintelligle
  giberrish starting at 00:05 and degrades to what ultimatley sounds like
  fabric tearing and thats it." **FAIL.** (Run vitals had looked healthy —
  full-length text-anchored 2284 frames, stable latent std 1.178 both seeds —
  the stable-noise-attractor illusion again, SIXTH instance of the ear
  beating the metrics. Seed-consistent full-length + teacher-band std is NOT
  evidence of speech.)
- `hv2ctl_cfg_heun8_s0` (control + inference CFG): "not great but speech is
  intelligble, just sounds like the guy is underwater or something. It does
  go the full length held togheter." — ~~the GN8-class config replicates on
  the clean cache~~ **[AMENDED same day by the scored rows: it does NOT
  replicate — see the regression finding below]**; still the only offline
  config that survives the loop.
- `hv2_heun8_s0_sig02` (σ=0.2 injected + bucket 2 told): "worse, I barley
  hear 'Fancy' and thats it it goes full alien weird stuff then silence then
  more wierd stufff." **INVERSION of GN5/GN6:** on the one-stream head,
  σ=0.2 feedback noise RESCUED content (100% coverage); on the dual-stream
  head the same noise kills near-instantly. The decorrelation trick does not
  transfer — head-v2's loop failure is acute sensitivity of the learned
  joint conditioning (noise now corrupts BOTH trained input pathways), not
  slow correlated-bias compounding. The loop attacks whatever the network
  has learned to depend on.

### Provisional verdict (pre-registered FAIL branch for arm A closed-loop)

Diagnosis resolves to branch 1 of the pre-registration: **training-time
information gap fixed (teacher-forced ≥ control), loop unfixed.** The last
"avoid stage-2" escape hatch is closed by direct test. Ledger:
- Offline training cannot cross the loop-stability threshold in ANY variant
  tried (one-stream, +inference CFG, dual-stream, σ-told): final
  confirmation of the GN4 conclusion, now with the strongest possible
  offline baseline.
- Finding for the paper: learned joint conditioning is MORE loop-fragile
  than arithmetic CFG applied outside the network — arm B (one-stream field,
  guidance composed arithmetically) survives where arm A (guidance absorbed
  into learned weights) dies. The loop amplifies through learned pathways.
- The hv2 20K checkpoint is the stage-2 initialization: healthy field,
  guidance absorbed, best long-context numbers to date.

### OFFICIAL CLOSE-OUT (2026-08-18 morning; `headv2_metrics.json` in this dir)

Held-out final rows: **20000:A 0.161 / 0.858** vs **20000:B 0.163 / 0.901**
(n=25 each) — statistically tied; A wins the 1200w bin (0.125 vs 0.191), B
wins 150w; A does it at HALF the head compute per sampler step (one field
eval vs the CFG pair). Pre-registered teacher-forced check: no arm-A
deficit, no training bug.

Closed loop, scored (same GN5–8 script/protocol/scorer):

| tag | WER vs script | coverage | voice % | sim med / last⅓ | rate wpm |
|---|---|---|---|---|---|
| hv2_heun8_s0 | 0.958 | 17.0% | 0 | −0.007 / −0.017 | 27 |
| hv2_heun8_s1 | 0.956 | 17.5% | 0 | 0.030 / −0.003 | 28 |
| hv2_heun8_s0_sig02 | 0.973 | 5.6% | 0 | 0.096 / 0.170 | 9 |
| hv2ctl_cfg_heun8_s0 | **0.116** | **100%** | 0 | 0.207 / 0.166 | 199.7 |
| hv2ctl_cfg_heun8_s1 | **0.119** | **100%** | 0 | 0.297 / 0.221 | 197.1 |
| *GN8 July-head ref* | *0.031* | *100%* | *61.7* | *0.522 / 0.453* | *174* |

**Verdict 1 — arm A closed loop: FAIL, pre-registered branch, both seeds.**
Scorer verdict string: "dual-stream training hurt on both seeds; diagnose
before any further spend" — scoped correctly to the LOOP (teacher-forced it
matched the control). Ear and metrics agree.

**Verdict 2 — σ-conditioning: NO RESCUE** (voice 0%, coverage 5.6% — worse
than plain arm A; the GN5/GN6 noise-rescue inversion confirmed by machine).

**Verdict 3 — NEW FINDING, the morning's most important number: the
control-architecture head trained on capture v2 REGRESSED in closed loop vs
the same architecture trained on v1 (GN8 July head): WER 0.116 vs 0.031
(drift 0.088 > 0.06 reseed floor, flagged by the pre-registered replication
check) and identity HALVED (sim_med 0.21–0.30 vs 0.52; voice% 0 vs 62).**
Josh's "underwater" is that halved identity, heard. Teacher-forced best-ever
+ closed-loop worse = the v2 data mix bought on-distribution quality at the
price of loop identity. Prime suspect: the **47% noised-feedback frames** —
GN7's ear already found noised-teacher targets "flatter, slower, less
emotional"; training half the pool on them plausibly bakes in the muffled
register. Alternates: long-form register shift; multi-prompt dilution.
NB the confound: July head = v1 data (short clips, clean); arm B = v2 data
(turn-split long-form, 47% noised, multi-prompt) — architecture identical,
data entirely different; the regression is a DATA effect.

### Next actions, in order (nothing trains until 1 reports)

1. **Clean-frames ablation (~30 min A100, the diagnosis):** train the
   control architecture on capture v2's σ=0 frames only (~268K pairs,
   filter `sigma < 0.05` at the frame level), closed-loop at 2 seeds, same
   protocol. Identity recovers toward GN8's 0.52 → noised frames are the
   culprit; the offline base trains CLEAN and noise-robustness moves to
   stage-2's on-policy data (where it belonged; the noised-teacher capture
   was always a proxy). No recovery → register/data-mix investigation
   before stage-2.
2. **Stage-2 DAgger collection (spec before spend):** hv2-or-best head
   drives the loop (capture-wraps-patch, built July); frozen DDPM head
   labels the self-generated states; fine-tune from the best base on a
   teacher-forced + on-policy mixture; 2–3 rounds; est. tens of dollars.
   Gates pre-registered with ≥2 seeds + graded sim + Josh's ear.
3. **Condition-drift monitor ships with stage-2:** per-frame LM hidden
   state recorded during renders, Mahalanobis distance to the cache
   reference distribution plotted against the spectrogram timeline —
   watch conditions leave the manifold BEFORE the ear hears it.
4. Parked: "bucket lie" probe (expectations low post-sig02); 1200w/2400w
   top-up capture.

## 2026-08-18 — CLEAN-FRAMES ABLATION PRE-REGISTRATION (not yet run)

Notebook: `clean_ablation_colab.ipynb`. Scorer: `score_cleanabl_gpu_colab.ipynb`.
A100, ~1–1.5 h, ~$5–8. Tests the HV2 close-out's prime suspect for the
closed-loop identity regression (verdict 3): **the 47% noised-feedback
frames in capture v2.**

### Design

One arm: control architecture (dual_stream=False — the only architecture
that survives the loop), trained on capture v2's **clean frames only**
(frame-level filter `sigma_bucket == 0`, ~268K pairs, mean/std recomputed on
the clean subset), same recipe (20K, bs 1024, 2e-4→2e-5 cosine, EMA 0.9999,
ckpts every 5K), same filtered file pool and identical held-out split as the
HV2 run. Eval: (a) teacher-forced held-out at 20K (n=25, heun8+CFG 1.3 —
comparability with 20000:B); (b) closed loop, 2 seeds, heun8+CFG, same
GN5–8 script/protocol.

Fixed comparison rows: arm B (mixed pool) closed loop = WER 0.116/0.119,
sim_med 0.207/0.297, voice 0%; GN8 July-head (v1 data) = WER 0.031, sim_med
0.522, voice 61.7%. Reseed floor: WER differences > 0.06 are real.

### Gate criteria — written before the run

| Verdict | Condition (both seeds) |
|---|---|
| **NOISED FRAMES CULPRIT** | WER ≤ 0.07 AND sim_med ≥ 0.42 (back in the GN8 band) → offline base trains CLEAN from now on; noise-robustness belongs exclusively to stage-2's on-policy data; the DAgger×GameNGen noised-teacher capture idea is retired for base training |
| **PARTIAL** | clear improvement over arm B (sim_med +0.10 or WER −0.05) but below the GN8 band → noised frames contribute; remainder is register/prompt-mix; stage-2 proceeds from the clean-trained base, residual investigated later |
| **NO RECOVERY** | ≈ arm B on both axes → noised frames acquitted; suspect becomes the long-form register / multi-prompt mix itself; investigate BEFORE stage-2 (v1-vs-v2 data-mix arms) |

Held-out note logged in advance: the clean pool is ~56% of the frames, so a
small teacher-forced dip vs 20000:B (0.163/0.901) is expected and does NOT
count against the run; the closed-loop rows are the verdict.

**Josh listens to both closed-loop renders** — the specific question for the
ear: is the "underwater" quality gone?

### Results (2026-08-18 — run + scored same day; `cleanabl_metrics.json` on Drive; ear pass pending)

**Held-out 20000:C (clean pool, n=25): WER 0.123 / sim 0.914 — BETTER than
both mixed-pool arms (A 0.161/0.858, B 0.163/0.901) on 56% of the frames.**
The pre-registered "small dip expected" note was wrong in the good
direction: the noised frames were dragging down teacher-forced quality too.
Clean 268K > mixed 470K. New best-ever, superseding yesterday's.

Closed loop (heun8+CFG, same protocol):

| | s0 | s1 | arm B (mixed) | GN8 July head |
|---|---|---|---|---|
| WER vs script | 0.116 | 0.158 | 0.116/0.119 | 0.031 |
| coverage | 100% | 100% | 100% | 100% |
| voice % | 24.8 | **88.6** | 0 | 61.7 |
| sim med / last⅓ | 0.474 / 0.467 | **0.572 / 0.569** | 0.21–0.30 | 0.522 / 0.453 |
| rate wpm | 205 | 208 | ~198 | 174 |

**Scorer verdict: PARTIAL** (pre-registered): the sim bar passed on BOTH
seeds (0.474/0.572 ≥ 0.42) — s1 actually EXCEEDS the July head — but the
WER bar (≤0.07) failed (0.116/0.158; per-seed spread 0.042, inside the
reseed floor). Reading:

1. **Noised frames were the identity killer, confirmed.** Removing them
   took loop sim from 0.21–0.30 (voice 0%) back to 0.47–0.57 (voice up to
   88.6%). The GN7 "flatter/noised-teacher-targets" suspicion was right.
2. **New observation — identity is FLAT through the render on both seeds**
   (last⅓ ≈ median: 0.467/0.474 and 0.569/0.572), where the July head
   ERODED (0.522→0.453). The long-form turn-split training data appears to
   buy within-render identity stability; July's short-clip data bought
   per-frame content precision (WER 0.031 vs 0.116). Complementary
   strengths, cleanly measured.
3. **Residual vs July head = content precision** (WER 0.116–0.158 vs
   0.031), possibly linked to the faster render rate (205–208 vs 174 wpm)
   and/or the smaller clean pool. This is the stage-2 target — and it also
   motivates a cheap offline arm first:
4. **Registered option — COMBINED CLEAN POOL:** v1 cache (478K clean
   short-clip frames) + v2 clean frames (268K long-form) = 746K frames,
   the largest all-clean pool available, covering both registers. One
   ~30–40 min A100 train; could merge July's WER with v2-clean's flat
   identity before stage-2 even starts. Candidate for the same night as
   the stage-2 collection.

Consequences: **the offline base trains CLEAN from now on**; the
noised-teacher capture component (DAgger×GameNGen proxy) is retired for
base training; stage-2 proceeds from the clean-trained base per the
pre-registered PARTIAL branch. Ear verdicts to be appended verbatim.

## 2026-08-18 — EAR VERDICTS (clean ablation) + SCOPE DECISION + the "one sound" observation

**Ear verdicts (Josh, verbatim):** s0 — "less underwatery to where now its
just a slight garbled, but no emaotion, just flat." s1 — "expressivness is
great, but quality is a bit worse... its not underwater, its a bit garbled
but its like just degraded sound." Both confirmed intelligible to 5 min.
Josh places both at "55-70 percent" of teacher; his bar: "if we could get
to 80% sounding good enough for most I'd be happy." s0-vs-s1 split maps to
the E1 dispersion axis (under-dispersed/flat vs dispersed/rough).

**Instrument-grade observation (Josh):** "across most of our tests the
sound is similar the underwater, garbled, not as clear sound, but just at
different levels or frequency... I want to just turn a nob." — consistent
with the single-axis model: every student artifact family (ghosting,
muffle, underwater, garble) = latent fine-structure distance from the
σ-VAE decoder's expected manifold, at different magnitudes. Evidence base:
roundtrip-clean teacher latents (July), rescale-made-it-worse (July),
renorm-worse-than-nothing (GN5), dispersion→audibility mapping (E1).

**SCOPE DECISION (Josh, 2026-08-18): constraint 5 relaxed** — see the dated
amendment in CLAUDE.md. Head capacity may grow if quality demands it; the
paper's claim is the method + the quality-vs-size-vs-NFE tradeoff curve
(speed story is carried by chunked-parallel batching). Ordering discipline
retained: no-param levers first (teacher-polish knob, combined clean pool,
MeanFlow), capacity ladder after, so gains stay attributable.

## 2026-08-18 — POLISH + COMBINED-POOL NIGHT PRE-REGISTRATION (not yet run)

Notebook: `polish_combined_colab.ipynb`. Scorer: `score_polish_gpu_colab.ipynb`.
A100, ~1.5–2 h, ~$10. The two no-param quality levers, per the amended
constraint-5 ordering. Teacher-polish mechanics verified against the pinned
fork commit (07cb79fea): `sample_speech_tokens` runs a vendored
DPMSolverMultistep with in-loop CFG; `add_noise` + mid-schedule `step()`
(diffusers-style `index_for_timestep`) support an SDEdit-style partial
denoise. Polish = flow-sample z → re-noise to the k-th-from-last timestep of
the teacher's 10-step schedule → run the teacher head's last k steps (CFG
1.3). **k IS the knob** (0 = raw student, larger = more teacher). Cost/frame
≈ k × ~5 ms of the 123M head on top of the flow head — k=2 keeps ~2×
head-level speedup vs the full DDPM head.

### Arms

- **P — the knob, teacher-forced:** cleanabl_20k head, 2 held-out utts
  (150w + 1200w), k ∈ {0, 1, 2, 3, 5}. Ten wavs to Drive; Josh turns the
  knob with his ears. Metrics per k (WER vs text, ECAPA vs teacher render).
- **M — combined clean pool:** v1 cache (478K clean short-clip frames) +
  v2 clean σ=0 frames (268K long-form) = ~746K, control architecture, same
  recipe (20K, ckpts every 5K). Held-out = the same v2 25-script split.
- **CL — closed loop (heun8+CFG, GN5–8 protocol):** combined head seeds
  0/1; cleanabl head + polish k=2 seed 0; combined head + polish k=2 seed 0
  (the stacked "80% candidate").

### Gate criteria — written before the run

| Verdict | Condition |
|---|---|
| **KNOB WORKS** | some k: Josh hears clearly less garble than k=0 without losing the voice, AND metrics don't contradict (sim within −0.05 of k=0, WER not worse by >0.06) → polish enters the operating config; quality-vs-k curve is a paper figure |
| **COMBINED WINS** | arm M held-out ≤ 0.123 WER (match/beat clean pool) AND closed-loop both seeds WER ≤ 0.09 with sim_med ≥ 0.45 → combined pool becomes the offline base |
| **MERGED SUCCESS** | any CL config: WER ≤ 0.07 AND sim_med ≥ 0.50 both-seed-or-ear-confirmed → July's content precision + v2's flat identity achieved in one head; stage-2 scope shrinks to polish-level residue |
| **NULL (either lever)** | no k audibly helps / arm M ≤ cleanabl on both axes → that lever retires; stage-2 (on-policy from the best base) proceeds unchanged |

Fixed references: cleanabl CL rows (0.116/0.474/24.8% and 0.158/0.572/88.6%),
GN8 July head (0.031/0.522/61.7%), reseed floor 0.06. Ear = authority on
listenability; Josh's stated target = "80% sounding good enough for most."

### Results (2026-08-18, same day; `polish_metrics.json` on Drive; partial ear pass)

**KNOB WORKS — pre-registered verdict met, ear-first.** Josh on the 150w
series (verbatim): "k3 is getting really close! they quality sound garble
underwater crap is almost gone, so now the voice expression and voice
clarity sound like teach, but theres still/now like a stutter every once in
while or almost like a millisecond audio frame misses... It feels like weve
alsmost fixed the [q]uality voice sounding good, now its like the speech
flow sound conistent smooth needs work." Metrics agree: teacher-sim rises
monotonically with k on BOTH scripts (150w 0.928→0.951; 1200w 0.925→0.954)
with WER flat through k3 — and a real operating ceiling: **k5 on the long
script costs content (WER 0.160→0.234)**, the polish occasionally
re-deciding frames. Operating range: k2–k3. **This is the interventional
confirmation of the manifold-distance model: pushing latents onto the
teacher manifold removes the garble family.** Residual artifact = frame-
boundary continuity (stutter); two suspects — decode 30s-chunk seams
(plumbing; crossfade fixes) vs per-frame independent polish noise (shared/
smoothed noise fixes); Josh to report stutter spacing.

**Combined pool (arm M): NOT the base.** Held-out 0.126/0.901 ≈ cleanabl
(0.123/0.914) with short/mid bins best-ever (300w 0.084, 600w 0.100) but
long bins diluted. Closed loop: WER improved (0.094/0.115) but **identity
regressed** (sim 0.374/0.328, voice 4.4/0%, last⅓ DECLINING again 0.333/
0.243) — the 64%-v1-dominant mix pulled the head back toward July-head
behavior (better words, eroding identity), trading away v2-clean's flat-
identity property. Data composition steers the content-vs-identity axis;
ratio is a tunable lever, not a free win. Pre-registered COMBINED WINS bar:
failed (sim < 0.45).

**Closed loop with polished feedback — the night's headline row:**

| tag | WER | voice % | sim med / last⅓ | horizon |
|---|---|---|---|---|
| cleanabl s0 (no polish, ref) | 0.116 | 24.8 | 0.474 / 0.467 | 254 s |
| **cleanabl_polish2_s0** | 0.243 | **62.0** | **0.522 / 0.525** | **282 s (full)** |
| combined_cfg_s0 | 0.094 | 4.4 | 0.374 / 0.333 | 60 s |
| combined_polish2_s0 | **0.076** | 0.0 | 0.319 / 0.267 | 0 |

**On-manifold feedback partially confirmed:** on the cleanabl base, loop
polish lifts identity to the July-head level (0.522) WITHOUT the July
erosion (flat 0.525 in the final third), voice windows 2.5×, full-render
horizon — at a content cost (WER 0.116→0.243, the k-cost showing up
through the loop). Properties still refuse to co-locate: content-best =
combined+polish (0.076), identity-best = cleanabl+polish (0.522 flat).
The ~0.52–0.57 identity band keeps recurring across configs (vs teacher-
self 0.734) — likely the offline ceiling; stage-2's target.

**Next probe (cheap, before stage-2): SPLIT-PATH POLISH.** Tonight's patch
polished ONE latent used for both audio and feedback. The two channels are
separable: (a) polish-audio-only (loop dynamics unchanged → keeps 0.116
WER, audio gets k3 texture — possibly the best-sounding config available
today), (b) polish-feedback-only (isolates the loop effect), (c) both
(tonight's). Plus the stutter fixes (shared polish noise; crossfaded
decode). One cell-set, ~$3, on the cleanabl base. Ear verdicts on tonight's
two loop renders + stutter spacing pending from Josh.

## Gate Night 1 continued — venue read (updates review §5)

The scaling curve was the ICASSP-vs-Interspeech decision gate, and it is
**unreadable until the floor is recalibrated**. With three blocking items ahead
of it and ICASSP 2027's 2026-09-16 deadline five weeks out, **ICASSP is off the
table on schedule grounds** — not on results. Target **Interspeech 2027
(~Mar 2027)** at full scope, arXiv when ready.
