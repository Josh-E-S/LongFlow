# Adversarial review (reviewer 2, clean context) — received 2026-07-08

Verbatim findings; supersedes parts of review-pre75k.md (see merged plan there).

## Verdict

75K-first attacks the wrong variable. Quantitatively, from our own logs: a
mathematically perfect velocity field sampled with our exact settings (Euler,
NFE 4, sway -1.0) produces conditional-spread shrinkage to ~0.55-0.63 of
target. Infinite data cannot fix that.

## CFG hypothesis: mostly acquitted

- Caches were captured at cfg_scale=1.3 (not 3.0) - mild guidance.
- CFG-unaware distillation is the STANDARD recipe: Meng & Salimans
  (arXiv:2210.03142) distill combined guided output into a cond-only student;
  dots.tts fuses CFG into the student (guidance_scale inert at inference);
  ZipVoice-Distill eliminates the dual pass. None treat baked-in CFG as a
  dispersion problem.
- Decisive: the student is under-dispersed relative to ITS OWN TRAINING
  TARGETS (the post-CFG latents ARE the std~1.0 distribution). Baked-in
  guidance cannot make a student undershoot its own targets.
- Legitimate residue: neg_condition is discarded at capture; information in it
  not recoverable from the positive state = extra irreducible conditional
  entropy -> mean-regression pressure. Judged second-order; testable (E5).

## What actually explains std 0.65

(a) Converged loss ~0.965 implies conditional sigma <= ~0.6 in standardized
space (Gaussian toy: optimal CFM loss 0.965 at sigma_cond=0.61): ~1/3 of
per-frame latent variance is noise-seeded by the teacher's DPM-Solver sampling
and NOT determined by the hidden state. Irreducibly stochastic under the
current independent-coupling objective.

(b) Perfect-field simulation, Euler + sway_grid(4,-1.0) = t {0,.076,.293,.617,1}
(final step 0.383 sits in the re-expansion phase near t=1):

| sigma_cond | NFE4 sway-1 | NFE4 sway0 | NFE16 sway-1 | NFE64 |
|---|---|---|---|---|
| 0.5 | 0.55 | 0.68 | 0.87 | 0.96 |
| 0.61 | 0.59 | 0.70 | 0.88 | 0.97 |

Sway -1 is an F5-TTS utterance-level heuristic; for a per-frame head with
tight conditionals it is plausibly the WRONG SIGN.

(c) Sampler bias alone gives marginal ~0.87-0.91, not 0.65 -> conditional
MEANS must also be attenuated (~0.7x). Loop-specific candidates: (i)
off-distribution conditions -> AdaLN-zero modulation weakens -> velocity
biased toward dataset mean -> muted latents -> feedback drift (also explains
progressive fade + broken termination); (ii) seed hazard: the closed-loop
patch was UNVERSIONED; sample_latents(seed=0) default would feed identical x0
to every frame if used naively per-frame (invisible in all batched offline
evals). Audit + version the patch before trusting any conclusion.

(d) The x1.79 global rescale scaled the conditional MEANS (the component that
was right) -> off-manifold garbage under EVERY hypothesis; "fine structure not
amplitude" is currently UNPROVEN. Valid test: residual inflation
x <- mean + k*(x - mean) around the head's own conditional mean.

(e) Dismissed: fp16 storage, standardization path, EMA lag at 20K (tau=2K,
10tau elapsed), uniform-t (standard), sway as train/test mismatch (it's just a
grid; but see (b) for direction).

## Training recipe gaps

- 20K steps = 43 epochs with loss still declining: UNDERtrained, not
  data-starved; distinct diagnoses, the plan conflated them.
- resources.md's own multi-t trick (LatentLM: ~4 t-draws per condition per
  step) is unimplemented - ~4x gradient signal, free.
- Constant lr to the end; add cosine tail to ~2e-5.
- Capacity: 16.6M x 4 NFE x 1 stream vs teacher 123M x ~20 x 2 streams =
  ~300x less compute per frame; ffn_ratio=2.0 thinner than paramcount
  suggests. Width-960 (~40M) ablation mandatory before "more data".
- n=5 single-speaker held-out set gating a 35h decision: re-split multi-speaker.
- load_pairs materializes all pairs on-device: 75K = ~22GB fp32 hidden; the
  75K plan silently includes a data-loader rewrite.

## Ranked experiments (<=4 GPU-h total, existing assets)

- E1 (~15 min, DECISIVE): teacher-forced dispersion audit - marginal std +
  per-condition spread (K=16 seeds/frame) at {NFE4/sway-1, NFE4/sway0, NFE16,
  NFE64, Heun NFE8}. std -> ~0.9 at NFE64/Heun = free sampler fix; flat 0.65
  teacher-forced = field collapsed (capacity/steps); healthy teacher-forced
  but 0.65 in loop = loop problem (E2/E4). NOBODY HAS MEASURED TEACHER-FORCED
  STD YET.
- E2 (~30 min): version FlowHeadPatch in src/, guarantee fresh per-frame x0,
  fix sample_latents seed default; rerun one closed-loop gen with best E1
  sampler.
- E3 (~1-2 h): steps-only continuation 20K->60K with lr tail + multi-t;
  dispersion-vs-steps curve. Then width-960 if plateaued.
- E4 (~30 min): condition-drift probe - hidden-state stats per frame vs cache
  distribution in a closed-loop run ("loop reacts to muted latents" is
  asserted, never measured).
- E5 (~50 min): capture neg_condition on ~100 utts; test residual
  predictability beyond condition. Kills or confirms CFG info-gap on record.

## If a big run happens: schema first

Capture (teacher initial noise, teacher sample) pairs -> converts training
from independent-coupling CFM (irreducible variance -> mean regression +
sampler bias) to PAIRED near-deterministic map distillation (ReFlow/
consistency-family; the mechanism behind honest 1-4 NFE in dots.tts/ZipVoice).
Also store neg_condition + cfg_scale. Near-zero capture cost; discovering the
need post-run means paying twice.

Sources: arXiv:2210.03142, arXiv:2606.07080 (+dots.tts-mf card), arXiv:2506.13053.
