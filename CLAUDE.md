# CLAUDE.md — LongFlow

Project context for Claude Code. Read this fully before writing any code.

## What this project is

LongFlow makes VibeVoice — the only open 90-minute, 4-speaker TTS system — fast, emotion-controllable, and drift-free **without training the backbone or tokenizers**. Four contributions, each a bolt-on to a frozen VibeVoice:

- **C1** — Replace the 10-step DDPM diffusion head with a ~15M-param flow-matching head (OT-CFM 4-NFE baseline → MeanFlow 1–2 NFE), trained on cached hidden states. ~~Target 15–20× end-to-end speedup.~~ **[2026-08-10]** Measured: head swap = **1.47× e2e** at euler4 (190→129ms/frame; Amdahl-capped, head is 35% of step cost). ~~But euler4 is dispersion-incorrect — **at the dispersion-correct heun8 sampler it is 2.2× head-level / ~1.24× e2e, and those are the numbers to quote.**~~ **[RETRACTED 2026-08-11, Gate Night 1 cell 5: heun8 WER 0.238 vs euler4 0.079 — not shippable. euler4 is intelligible but under-dispersed. No sampler is both; shipped config undecided, fix is the head. P2 MeanFlow is load-bearing for CORRECTNESS, not just speed.]** Claim = head-level speedup + first-packet latency + bottleneck map, not RTF. See README thesis note and `experiments/p1_flow_head/NOTES.md` Gate Night 1 entry.
- **C2** — ~~Training-free per-speaker emotion control~~ **DESCOPED 2026-07-07 (P0 verdict: PARTIAL)** — steering localizes cleanly but the backbone's affect ceiling caps perception; now a paper-appendix negative result (finding N7). The turn-localized injection machinery (`src/steering/`) is retained for possible reuse (e.g. anti-jingle suppression).
- **C3** — Inference-time speaker anchoring: blend immutable reference embedding with a running buffer of the speaker's generated turns.
- **C4** — Long-horizon consistency benchmark: speaker-drift curves, durational WER, windowed UTMOS at 30s → 90min.

Full scope, positioning, data plan, and phases: `README.md`. ~~Paper target: Interspeech 2026 / arXiv.~~ **[2026-08-10]** Paper target: ICASSP 2027 (deadline 2026-09-16, reduced scope) or Interspeech 2027 (full scope) — decision gated on this week's data-scaling curve; arXiv regardless. See `docs/review-2026-08-10.md`.

## Current state (July 2026) — see August addendum below

- Repo contains README, `docs/` (architecture, negative results incl. **N7**, **`resources.md` — pinned repos/weights/datasets/eval stack, verified 2026-07-05, read it before touching any external dependency**), the package scaffold (uv-managed Python 3.11 env, ruff + pre-commit), working `src/steering/` (capture + extraction + injection, retained post-descope), `src/eval/metrics.py` (WER/ECAPA/prosody, exercised on real sweeps), and `src/cache/alignment.py`.
- **P0 complete (2026-07-07, verdict PARTIAL, C2 descoped — see `experiments/p0_steering/NOTES.md` and finding N7).** Paper scope is now C1+C3+C4.
- ~~**Next task is P1: the flow-head baseline** — caching pipeline (`src/cache/`, reuse the positive-stream logic from `src/steering/contrast_pairs.py`), then OT-CFM 4-NFE head with the mandatory 1K/5K gate check before the full 75K run.~~ **[2026-08-10]** P1 is mid-flight, not pending: caching pipeline built (batched, 4.9×), gate PASSED, 10K scaling PASSED, E3 steps-scaling FAILED. See the August addendum for the actual next action.
- ~~Phase order: P1 flow-head baseline → P2 MeanFlow 1–2 NFE → P3 anchoring + benchmark → P4 encoder distillation (stretch) → P5 paper.~~ **[2026-08-10]** See revised ordering in the August addendum.

## Current state (2026-08-14 — READ THIS FIRST; supersedes the addenda below)

Five gate nights (GN1–GN5, 2026-08-11→14, ~$15 total) are fully logged in
`experiments/p1_flow_head/NOTES.md` — read the Gate Night entries before
proposing anything. Where things stand:

- **Teacher operating mode SOLVED — finding N8 + the turn-split cure**
  (`docs/negative-results.md` N8): VibeVoice inflates speaking rate 1.5× on
  monolithic long scripts (1.5B AND 7B, ratio 1.31) and drifts identity after
  ~8 min. **Both defects share one cause and one cure: same-speaker
  turn-splitting (~60-word turns) within ONE generate call** → natural rate
  (165–177 wpm, 3 replications) AND flat identity through 19+ min. Chunking
  between calls does NOT cure pacing; prompt time-stretching does not
  transfer. Never generate monolithic long-form.
- **Product path VALIDATED end to end** (stock teacher, no training):
  turn-split × parallel chunk batching (audio-level parity PASSED, ~2.9× at
  batch-4) × 0.25 s crossfade seams (ears-pass) × detect-and-reroll for the
  rare transient-babble glitch (ECAPA+rate filter is the detector). 90-min /
  4-speaker renders in ~20–30 min today; head + CFG later → single digits.
- **The 20K flow head: fast and correct teacher-forced, DIES in closed loop**
  (GN4: euler4 collapses at 0:14, heun8 degrades to 53% coverage). Cause
  isolated by GN5 ablations: **feedback-OOD — the loop amplifies the
  CORRELATED component of the head's error.** Feedback removal = instant
  death (channel is load-bearing); per-frame statistical renorm = worse than
  nothing; **σ=0.5 feedback noise = content survives the FULL render (14 s →
  5 min, WER 0.296 vs teacher rendition) at the cost of voice identity
  (raspy whisper).** Register-OOD and turn-reset hypotheses: refuted.
- **Consequences:** (a) stage-2 on-policy training (Self Forcing family) is
  the only road to shippable closed-loop quality — read Causal Forcing
  2602.02214 + CF++ 2605.15141 (verified real; CF++ = frame-wise 1–2 step,
  no cached trajectories) before choosing the method; (b) GN6 candidate: the
  feedback-noise σ sweep (0.1–0.5) hunting a listenable sweet spot; (c) the
  75K offline scale-up stays DEFERRED until the stage-2 method is chosen.
- **Instruments:** collapse metric = per-window latent Fréchet distance
  (scalar std lied twice — never use it as a collapse metric); reseed floor
  = median 0.000, IQR [0, 0.062], n=20 (differences >0.06 WER are real;
  exclude proper-noun-heavy texts); no bitwise batch-determinism tests
  (stochastic AR — distributional parity is the only valid instrument);
  quality filter = per-window ECAPA + speaking rate. **Josh listens to
  everything; his ear has beaten the metrics four times.**
- **Process:** notebooks carry a `NOTEBOOK_VERSION` banner (verify before
  running); every artifact mirrors to Drive per-run; reruns skip completed
  work via Drive; experimental revisions get NEW artifact tags (f_→f2_→f3_).
- **Venue: Interspeech 2027 (~Mar). ICASSP is dead (Josh's call, twice).**

## Current state (2026-08-10 addendum)

Written after the 2026-08-10 project review (`docs/review-2026-08-10.md`). The July 7–9 sprint is fully logged in `experiments/p1_flow_head/NOTES.md`; headlines:

- **P1 gate PASS** (roundtrip clean; flow4 0.030 WER / 0.984 sim vs teacher on train set) → **10K scaling PASS** (held-out 0.105 WER / 0.824 sim, beats 800-utt 5/5) → **E3 steps-scaling FAIL** (80K ckpt overfits 10K cache: held-out WER 0.209 vs 20K's 0.088; endurance collapses frame-zero on long-prefix OOD). **20K ckpt is the operating checkpoint. Data, not steps, is the binding constraint.**
- **Closed-loop fade CURED** (E1/E1b): sampler was the thief — heun8 matches teacher dispersion; clean termination. Residual "digital cold" texture = underfit polish problem.
- **Dominant failure axis = context-length OOD** (text prefix ≫ training samples → hidden states off-distribution from frame one). Consequence: windowed/anchored context (StreamingLLM-style sink) is promoted from P3 side-arm toward core contribution; C3 embedding-blend anchoring demoted to ablation arm.
- **Recipe alignment (2026-08-10):** our roadmap independently reconstructed the 2025–26 video causal-distillation recipe — offline teacher-forcing head distillation (=our P1; cf. CausVid/TMD 2601.09881) → on-policy self-forcing/DAgger stage (Huang et al. 2025; Causal-rCM 2606.25473) → attention-sink context management for long rollouts (DySink 2605.21028, TetherCache 2606.13035). **Nobody has instantiated this in speech.** Import the recipes; cite the lineage; see `docs/review-2026-08-10.md` §4.
- **Revised phase order:** P1 finish (blocking gates → scaling curve → cache sized by the curve → ≥50K-step train w/ val machinery, intermediate ckpts every 5–10K) → P2 MeanFlow 1–2 NFE (now load-bearing for the speed story) → P3 windowed context + on-policy distillation (recipe stages 2–3) + C4 benchmark → P4 encoder distillation (stretch) → P5 paper.
- ~~**Immediate next action (unchanged since 2026-07-09):** run `experiments/p1_flow_head/gate_night1_colab.ipynb` (stats equality, teacher determinism, reseed floor, heun8-vs-euler4 A/B) + the data-scaling curve on the existing 10K cache (~5 GPU-h total). The curve decides ICASSP-vs-Interspeech scope.~~ **[DONE 2026-08-11 — Gate Night 1 ran; read the Gate Night 1 entry in `experiments/p1_flow_head/NOTES.md` before doing anything else.]** Outcome: cells 2 and 3 PASS (75K paired noise→latent schema viable); cells 5 and 6 BLOCKING; cell 4 unusable at n=6; cell 7 produced finding **N8** (teacher inflates speaking rate 1.5× on long scripts). The scaling curve was **not** run — no valid noise band yet. **ICASSP 2027 is off on schedule grounds; target Interspeech 2027.**
- **Next actions, in order (2026-08-11) — nothing touches 75K until 1–3 clear:** (1) reseed floor at n=30–50, report median+IQR not mean; (2) resolve batched-vs-unbatched condition parity, or capture unbatched and eat the 4.9×; (3) switch the cache quality filter from latent std to per-window ECAPA + speaking rate (latent std missed the largest audible defect in the endurance run); (4) decide rate correction for long-context captures (N8); then (5) the data-scaling curve, which is still the scope gate.
- **Open question, do not paper over:** no sampler is both intelligible and dispersion-correct (heun8 0.238 WER / euler4 0.079 but under-dispersed). The shipped configuration is undecided. Fix the head — do not tune the integrator.
- **Second open question (Gate Night 2, 2026-08-11):** N8's rate inflation is NOT cured by text chunking at practical sizes (237 wpm at 320-word chunks vs 165 natural) — the fast register triggers at low hundreds of words and ramps over the first ~1–2 min. The community folk remedy (also Microsoft's official tip and the ComfyUI wrapper's silent default) only partially mitigates; nobody had measured it before. Do not propose chunking as the pacing fix. Windowed context is justified by drift, not pacing.
- **Process rules now in force:** commit every training invocation (steps/lr/ema/seed/bundle commit) per run; save intermediate checkpoints in ALL runs; held-out sets stratified by speaker AND context length.

## Hard constraints — never violate these

These come from documented negative results (`docs/negative-results.md`). They are not preferences.

1. **Never train or fine-tune the Qwen2.5 backbone, the tokenizers, or the σ-VAE decoder.** Everything upstream and downstream of the flow head stays frozen. If a proposed fix involves unfreezing the backbone, it's the wrong fix.
2. **Never condition a generative head on hidden states from an MSE/regression-pretrained backbone** (finding N1/N2). LongFlow conditions only on VibeVoice's own hidden states, captured from *full inference runs including the acoustic/semantic feedback loop* — never standalone text-conditioned forward passes.
3. **Verify frame alignment before wiring anything.** Cached hidden states `[B, T, d_model]` and acoustic latents `[B, T, d_latent]` must share the same T. Read `d_model` from the checkpoint config and `d_latent` from the σ-VAE config at runtime, and assert both in the caching code — never hardcode widths in code or docs (a stale Qwen3-era 2048 survived three months of doc propagation before being caught). A T-misalignment reproduces the April 7 failure signature (sharp but unintelligible audio).
4. **Stay in VibeVoice's native latent space end-to-end** (finding N3/N4). No cross-tokenizer prediction, no codec swaps.
5. **Keep adapters thin** (finding N5). The flow head is ~15M params, no attention. Don't add capacity to fix a conditioning problem.
6. **Gate check before every full run.** 1K samples / 5K steps / decode / listen. If a gate fails, diagnose before scaling — 75× more data has already been proven not to fix representation problems.

## Compute conventions

- **Iteration and gate checks:** Vast.ai RTX 3090 (~$0.20–0.35/hr). Container startup on Modal hurts iteration speed — save Modal for long unattended training runs.
- **Full training runs:** A100 40GB (Vast.ai or Modal).
- **Modal DataLoader settings (mandatory, from the April 8 crash):** `pin_memory=False`, `num_workers=2`, `persistent_workers=False`.
- Cached `.pt` pairs are the training substrate; caching runs are one-time costs — checkpoint them aggressively.

## Baseline model + references

- Backbone system: VibeVoice community fork — https://github.com/vibevoice-community/VibeVoice (Microsoft pulled the original repo; the community fork is the maintained one — verify current state before pinning). VibeVoice-1.5B for iteration; technical report arXiv:2508.19205.
- Flow head lineage: LatentLM arXiv:2412.08635; ZipVoice distillation arXiv:2506.13053.
- MeanFlow-objective references for P2: MeanFlow (Geng et al. 2025), DSFlow arXiv:2602.09041 (AdaLN vs token-conditioning at few NFE), RealUID (ICLR 2026).
- Steering reference for C2: EmoSteer-TTS arXiv:2508.03543 (single-utterance; our novelty is per-speaker, turn-localized, VAD-continuous).
- Eval: Whisper-large-v3 (WER), WavLM + ECAPA-TDNN (speaker sim), UTMOS (sliding windows).

## Code conventions

- Python 3.11, PyTorch. `ruff` + `pre-commit` per Josh's standard project-starter setup.
- Repo layout (already specified in README): `src/cache/`, `src/flow_head/`, `src/steering/`, `src/anchoring/`, `src/eval/`, `configs/`, `experiments/`, `tests/`.
- Every experiment gets a directory under `experiments/` with a `NOTES.md` copied from `experiments/NOTES-template.md` — hypothesis and gate criteria filled in *before* the run, results and verdict after, *even when it fails*. Negative results are first-class citizens in this project.
- Testing is three-tier: **tier 1** fast unit tests in `tests/` (synthetic tensors, shapes/masks/NaN, T-alignment, loss values on known inputs, JVP-tangent vs finite differences — run by default, keep them fast); **tier 2** `@pytest.mark.slow`/`gpu` integration tests (1-clip overfit gate — opt-in via `pytest -m slow`); **tier 3** gate checks are *experiments with a human listening step*, documented via the NOTES template, never automated into assertions.
- Audio artifacts for listening checks go to `experiments/<name>/audio/` — never committed to git except curated demo samples in `samples/` (git-lfs).
- Tag a git release at each phase gate (`p0-pass`, `p1-baseline`, ...) so paper numbers map to commits.

## Things Claude Code should push back on

- Any suggestion to "just fine-tune the backbone a little" — see constraint 1.
- Scaling data to fix a failing gate check — diagnose the representation/alignment issue instead.
- Adding attention or depth to the flow head before the MLP version has been fully evaluated.
- Skipping the listening step. Metrics missed the April 7 failure mode until a human listened.
- Streaming inference, 5+ speakers, or multilingual — explicitly out of scope for this paper (see README non-goals in "Rejected alternatives").
