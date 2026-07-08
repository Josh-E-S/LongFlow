# Pre-75K review round 2 (2026-07-08, post-E3) — three clean-context reviewers

Commissioned by Josh after the E3 overfit verdict, before the big-cache spend:
(A) code audit for small polluting bugs, (B) experiment-design gaps neither
prior review caught, (C) data-strategy research (2025–26 literature + datasets).
Distilled from the full reports; decisions merged into the execution plan at
the bottom.

## A. Code audit — verdict: no confirmed bug pollutes existing results

The overfit verdict and "data is the binding constraint" conclusion stand.
Traced clean: CFM loss/samplers/standardization round-trip, fp16 cache storage
(lossless for bf16-range values; overflow loudly refused by alignment asserts),
EMA warm-start confound (decays as decay^n → e−30 over 60K steps; negligible),
batched row attribution (cannot break hidden↔latent pairing by construction —
only utt metadata, which training never uses), decode-path consistency, voice-
prompt symmetry, drift-curve units.

Actioned same day (commit with this file):
1. **Shape guards** — `FlowHead.forward` rejects non-2-D inputs (a `[B,1,d]`
   condition would silently broadcast through AdaLN); `to_utterance` rejects
   multi-row capture calls (a fork update stacking pos+neg streams would
   silently flatten to `[1, 2d]` and poison an entire cache).
2. **`FlowHeadPatch` nesting restoration** — patch exit now restores an outer
   `SampleCapture` wrapper instead of deleting it (would have fired on the
   DAgger probe).
3. **Test hardening** — synthetic cache now has non-trivial latent mean/std
   (5.0/3.0) and non-N(0,1) hiddens (4.0/2.0), so any standardization
   asymmetry (hidden standardized too, mean/std swap, wrong-ckpt stats) fails
   the e2e test instead of hiding behind identity stats; plus explicit
   hidden-not-standardized assert, 3-D-rejection tests, nesting test.

Open items (not code):
- **Stats-equality check (2 min, next Colab):** assert
  `ckpt20["latent_mean"] == ckpt80["latent_mean"]` (and std). Expected equal
  (same frozen cache, deterministic sorted glob) — confirms the 20K-vs-80K
  dispersion table was apples-to-apples. Also: future dispersion evals should
  sample frames across all speakers, not `[:512]`/`[:24]` of the sorted files.
- **Process rule:** training-run invocations (steps, lr, ema_decay, seed,
  use_ema at load, bundle commit) must be committed per run — the 80K run is
  currently not reproducible from the repo.

## B. Experiment-design review — BLOCKING gates before the 35 GPU-h (~7–9 GPU-h total)

1. **Teacher determinism audit + teacher-vs-teacher floor (~1 GPU-h, no
   training).** (a) The paired-map schema assumes `sample_speech_tokens` is a
   deterministic function of (initial noise, condition) — if the teacher's
   solver injects intermediate noise, captured initial noise determines
   nothing and the paired objective silently degrades to independent coupling.
   Fix noise, run same condition twice, assert near-identical latents. (b) Run
   the TEACHER twice (different seeds) on held-out conditions; WER/sim between
   the two draws = the metric floor legitimate sampling variance imposes. If
   teacher-vs-teacher ≈ 0.07–0.08 WER, the 20K head is already near the
   independent-coupling ceiling and the big run's value rides entirely on the
   schema change. Fold in a 15-min heun8-vs-euler4 held-out WER A/B (NOTES
   never recorded which sampler produced 0.088).
2. **1K gate on the NEW recipe (~2–3 GPU-h).** The 75K run is a new schema +
   new objective + new data mix + forced dataloader rewrite — CLAUDE.md
   constraint 6 applies to the recipe, not just the data. Capture ~1K with
   full new schema; verify noise→latent replay; train paired-map AND
   independent-CFM on the same 1K; held-out eval; listen. Capacity probe
   (width-960) waits for this pilot cache — the paired objective plausibly
   changes capacity demand, so an old-objective probe could mis-rank widths.
3. **Validation machinery + retro-validation (~0.5 GPU-h + code).** train()
   gets: val split BY UTTERANCE (frame-level splits leak), standardization
   stats from train split only, low-variance val estimator (fixed stratified
   t-grid + fixed x0 per val frame), EMA-weight eval every 1–2K steps,
   checkpoints every 5K, patience stop. Implement multi-t (~4 draws/condition)
   now, debut in item 4. RETRO-CHECK: compute this val loss on the saved
   20K/80K ckpts — if it separates them, val CFM loss is a trustworthy
   early-stop proxy for the big run; if not, the run needs periodic decoded
   WER instead. EMA decay scales with run length (0.9995@20K → ~0.9999@50K+).
4. **Data-scaling curve on the EXISTING cache (~3–4 GPU-h).** Train on 2.5K
   and 5K subsets (zero capture cost), early-stopped via item 3, same held-out
   set → best-val WER/sim vs data size (800/2.5K/5K/10K, log axis).
   Falsification: if 5K ≈ 10K within the teacher-reseed noise band (item 1b
   defines it), volume is NOT the constraint → shrink the cache to ~25K and
   reallocate GPU-h to long-context/dialogue coverage.
5. **E4 rebuilt as context-length OOD spec (~1–2 GPU-h).** E3's frame-zero
   collapse shows the dominant OOD axis is text-prefix length, not feedback.
   Teacher-driven captures at ~50/200/400/800/1600/4000 words (concatenated
   same-speaker chapter text); measure per-frame condition distance to the
   10K cache distribution (kNN in PCA-50 + diagonal Mahalanobis) vs frame
   index; then teacher-force those long-context conditions through the 20K
   head → WER/dispersion per length bucket. Deliverable: a context-length CDF
   requirement for the 75K capture. Bonus: these captures are reusable
   training data.
6. **PARALLEL — batched-capture condition parity (~0.5 GPU-h).** The 75K cache
   uses batch-8 LEFT-PADDED capture; inference is unbatched. Nobody has
   checked the hidden states (training INPUTS) match between batched/unbatched
   capture of the same utterance. Capture 10–20 utts both ways, compare
   per-frame. If offset: length-bucket or capture long-context unbatched.
7. **PARALLEL — mid-length eval ladder (30s/90s/3min on the 20K head,
   ~1 GPU-h + listen).** The 1–3 min band (where drift onset lives and where
   the long-context data must show improvement) has zero baseline
   measurements; without it the post-75K gain in the paper's core regime is
   unattributable.

Sequence: 1 → 2 (needs 1a's harness) → 3+6 parallel → 4 → 5 → 7 → 75K capture.

## C. Data strategy — recommended capture mix (budget by FRAMES, not utterances)

~35 GPU-h ≈ 3.5–4.5M pairs. Long captures are the cheapest pairs/GPU-h AND the
only source of the long-context conditions that collapsed. Mix:

| Share | Bucket | Source |
|---|---|---|
| 30% | Short single-speaker read (status quo, regression control) | LibriTTS-R train.clean.360 |
| 30% | Long-form monologue, 2–20 min, log-uniform lengths, frames from ALL positions | Gutenberg chapter text; prompts from LibriTTS-R + `nvidia/hifitts-2` (36.7k h / 5,000 speakers, CC BY 4.0 — 5× speaker diversity) |
| 25% | Multi-speaker dialogue, 2–4 speakers, 2–20 min | Scripts: `allenai/soda` (pinned in resources.md, never used!) chained + LLM-generated podcast scripts (MOSS-TTSD precedent); distinct prompt speakers per slot. Include rapid short exchanges — turn boundaries are where conditions move fastest (ZipVoice-Dialog arXiv:2507.09318) |
| 15% | Short conversational-register text | SODA turns / podcast-register sentences |

Field prior: FireRedTTS-2, SoulX-Podcast, MOSS-TTSD all land near 20–25%
dialogue by hours. Reserve a few 45–90-min runs as eval probes, not bulk.
Interim gate at ~25K-equivalents before burning all 35 GPU-h.

Key literature: **teacher-generated distillation data is not just standard but
state-of-the-art** — "Flow Map Distillation Without Data" (arXiv:2511.19428)
names Teacher-Data Mismatch as the failure mode and E3's long-script collapse
is a textbook case on the conditioning side; the input distribution must match
DEPLOYMENT (long multi-speaker), not a convenience corpus. VibeVoice itself
was curriculum-trained 4K→65K tokens (arXiv:2508.19205) — deep-position hidden
states are in-distribution for the BACKBONE, only OOD for OUR CACHE; fixable
purely by capture distribution. Distillation scaling laws (arXiv:2502.08606):
small students saturate early — spend increments on coverage, not volume.
MagpieTTS-LF (arXiv:2606.18485) validates the P3 windowed-context arm.

Flags against the current plan:
1. **resources.md §4 "sample 75K from train.clean.360" as written reproduces
   the OOD failure** — 7.5× more of the same short-read distribution.
2. **cfg_scale covariate: cache captured at 1.3; VibeVoice's shipped default
   is 3.0.** Decide the deployment cfg and capture consistently (or capture
   the scale per the schema and ablate).
3. Listen to a teacher-generated 20-min clip BEFORE harvesting long captures
   (tier 3) — distilling frames from a region where the teacher itself
   degrades teaches the student the degradation.

Datasets verified live on HF 2026-07-08: `nvidia/hifitts-2`, `allenai/soda`,
`speechcolab/gigaspeech` (conversational text), `amphion/Emilia-Dataset` (use
CC BY YODAS portion only). Skip: `nvidia/Granary` (25 EU languages, ASR/AST),
MLS (redundant), People's Speech (noisy, machine transcripts). SPoRC
(`blitt/SPoRC`, 1.1M diarized podcast transcripts) = best real dialogue
structure but murky license — teacher-input-only if used.
