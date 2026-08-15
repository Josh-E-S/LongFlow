# Stage-2 reading pass — 2026-08-14

Pre-registered in NOTES ("Reading of dots.tts + CF++ happens off-GPU in
parallel; stage-2 method decision follows it"). Three parallel deep-reads:
(1) Causal Forcing + CF++, (2) dots.tts, (3) freshness sweep + citation
verification. This file is the digest + the resulting method decision.
Full agent reports archived in the session; key claims below were verified
by direct paper fetches, not search snippets.

## Headline: NOT scooped, and the plan improves

- **No published work does on-policy / self-forcing / noise-augmented
  training of a head-only student under a frozen AR TTS backbone.** Nothing
  touches VibeVoice's diffusion head (no distillation, replacement, or
  acceleration in print; community work is quant/engineering only).
- **dots.tts (2606.07080v2, rednote+SJTU, revised 2026-08-10) is adjacent,
  not a scoop.** Its "SOAR" self-corrective post-training updates only the
  acoustic DiT with everything frozen — superficially our regime — but the
  rollout is a single detached Euler step *within one patch* (a solver-
  discretization fix), conditioned on **ground-truth prefixes** throughout.
  Zero occurrences of "exposure bias", "on-policy", "scheduled sampling",
  or history noise in the paper; the Self Forcing / Diffusion Forcing
  lineage is entirely uncited. Their own tables show SOAR buys ~0.4 SIM and
  slightly *worsens* hard-set WER. It never faces closed-loop drift.
- **Option 3 (noise-augmented feedback training) is unpublished for
  speech/TTS**, as is our inference-time σ-noise finding. GameNGen's
  mechanism has no speech descendant (multiple query phrasings, verified
  negative).
- **N8 remains unmeasured anywhere.** Only anecdotal corroboration:
  microsoft/VibeVoice issue #85 "Generated voice talking too fast"
  (workaround = repeated speaker labels ≈ our turn-split). Cite it as
  independent sighting. The render-record claim is uncontested (best public
  timing: RTF 0.5 complaint in issue #268; exl3 fork ~1×).

## What each read contributed

### Causal Forcing (2602.02214, ICML 2026) + CF++ (2605.15141) — video

Verified real by direct fetch (the sweep agent's search couldn't surface a
"Causal Forcing++" title; the direct fetch of 2605.15141 returns "Causal
Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation…", Zhao,
Zhu, Zheng et al., v2 2026-06-01, no confirmed venue). Three-stage recipe:
AR teacher (teacher-forced) → distillation init → **asymmetric DMD with
self-rollout** (student conditions on its own prefix; frozen base model as
real score, online-trained fake score).

**The transferable findings:**

1. **CF++'s central warning maps onto GN4/GN5 exactly:** causal **DMD is
   mode-seeking → "more sensitive to accumulated history errors"**, i.e. it
   amplifies the correlated-error channel we isolated in GN5. Running the
   heavyweight distribution-matching stage on a weak initialization is the
   one move both papers say not to make. Their prediction: CD-initialized
   students degrade gracefully under history drift; DMD on weak init
   collapses fast. That is euler4-dies-at-0:14 in their vocabulary.
2. **Causal consistency distillation (causal CD) is a cheap stage-1.5 for
   us:** their CD stage is *teacher-forced* — our existing cached .pt pairs
   are exactly the right substrate, no rollout collection needed. Our frozen
   600M DDPM head plays "AR teacher": one online DDPM step between adjacent
   noise levels per update, self-consistency loss + EMA target on the 15M
   head. Nearly a loss-function swap on the current pipeline; est.
   single-digit-to-~20 A100-h at our scale (extrapolated, not from paper).
   Theory says this buys **mode coverage** → collapse becomes drift.
3. **DMD is implementable in our regime if ever needed:** frozen DDPM head
   = s_real; second 15M flow head trained online = s_fake; rollout cost
   dominated by the frozen LM forward. Est. 20–80 A100-h. Hard parts:
   closed-loop collection wired into training, and ε-pred↔velocity
   conversion so the DDPM head serves as a score at our parameterization.
4. **Structural caveat both papers share:** they train the *whole*
   generator; conditioning flows through trainable causal attention. No
   frozen-backbone-plus-tiny-head configuration exists in print — gradients
   never see how head errors reshape future LM hidden states; the on-policy
   *data distribution* is the only mechanism we get. Untested at a 15M/1.5B
   trainable-to-frozen ratio. This risk applies to every candidate equally.
5. Causal Forcing's Prop. 3.4 argues *against* noisy-history conditioning
   for their setting — but their inference context is clean; ours is the
   student's own error-bearing output. Their negative does not transfer to
   our regime (GameNGen's positive, below, is the matching regime).

### dots.tts (2606.07080v2) — closest architecture in print

- **The NOTES claim is CONFIRMED and strengthened:** their LM "sees only
  this semantic summary, not the raw VAE latent… necessary to keep
  continuous-AR rollouts stable." The semantic encoder is born from
  WavLM-aligned VAE training, transplanted, then trained end-to-end — it
  defines the LM's audio input vocabulary. Retrofitting semantic-only
  feedback onto frozen VibeVoice = LM retrain. Definitively off the table.
  (Caveat: they offer zero ablation evidence that semantic beats acoustic
  feedback; there is no ablation table anywhere in the paper.)
- **The acoustic loop isn't deleted — it's moved INTO the head.** Their
  AR-FM head gets **full clean-acoustic-prefix conditioning** (all prior
  patches interleaved with all LLM hidden states); the paper notes the head
  "is on its own a complete text-conditioned speech generator". **Our 15M
  head sees only the current hidden state — it has no acoustic memory of
  its own.** This is the architectural lever for the identity axis: a head
  with direct access to (even a window of) real prior latents has a timbre
  anchor the fragile feedback loop can't erode. Tension with constraint 5
  (thin head, no attention) noted — but constraint 5 forbids capacity as a
  *substitute for fixing conditioning*; history conditioning IS a
  conditioning fix. A windowed-MLP or tiny-cross-attn variant is a design
  decision for the head-v2 gate, not a violation by default.
- No long-form anything: max duration never stated, no drift/pacing
  measurements, single-speaker cloning only. Our C4 territory untouched.

### Freshness sweep — the rest of the field (Mar–Aug 2026)

- **OmniForcing** (2603.11647, Mar 2026): first self-forcing distillation
  including an AR audio stream (joint audio-video). Weakens any "first
  self-forcing for audio" phrasing; strengthens feasibility. Not TTS, not
  head-only.
- **Mutual Forcing** (2604.25819): self-forcing family without a
  bidirectional teacher — relevant to our design space (we lack one too).
- **GROW** (2608.03215, Aug 4 2026): on-policy RL on an AR-diffusion TTS
  (DiTAR), rewards = intelligibility + speaker sim — literally our two
  failure axes. Short utterances, external rewards, not aimed at
  closed-loop collapse. Registered as **fourth stage-2 candidate**, low
  priority; its reward pair is a ready-made eval harness idea.
- **BAgger** (2512.12080, Dec 2025): the published DAgger-for-AR-video-
  diffusion template — cite if option 2 ever runs. No speech DAgger exists.
- **Prior art to cite for "self-generated context in TTS training":**
  2509.17021 (Sept 2025) — hybrid teacher-forcing/free-running with
  self-generated *discrete* tokens. Discrete, not latent-continuous; does
  not cover our option 3.
- **Citations verified for the paper:** Diffusion Forcing = 2407.01392
  (NeurIPS 2024, MIT; per-token independent noise on history — accurate as
  characterized). GameNGen = 2408.14837 (ICLR 2025, Google; §3.2.1
  "Mitigating Auto-Regressive Drift Using Noise Augmentation" — Gaussian
  corruption of encoded context frames, **noise level fed to the model via
  a learned bucket embedding**, "critical for preserving visual stability").
- VibeVoice tech report accepted **ICLR 2026 Oral** (OpenReview
  FihSkzyxdv) — reviews unread, worth a skim for long-form discussion.

## The GameNGen detail that upgrades option 3 → the plan

GameNGen doesn't just noise the context — **it tells the model how much
noise it added** (per-bucket learned embedding). Our σ=0.5 inference hack
injects noise the head cannot discount, which is plausibly exactly why
identity erodes to a raspy whisper while content survives. The training
version fixes the mechanism GN5 proved matters AND the deficiency GN5's
hack suffered from. Direct design import: sample σ per frame/window during
training, condition the head on the σ bucket.

## DECISION (stage-2 method) — AMENDED same day by the critic pass, see below

~~Original sequence kept for the record; superseded by the amended sequence
at the end of this file.~~

1. **GN6 — σ sweep (unchanged, ~$3):** inference-time sweet-spot hunt +
   the noise schedule for step 3. Runs first regardless.
2. **Stage 1.5 — causal-CD loss swap on the existing 20K cached pairs
   (NEW, from CF++):** cheapest training move available (~10–20 A100-h),
   zero new data plumbing, theory predicts collapse→graceful-drift. Gate:
   closed-loop horizon vs GN4's 14 s / 53 % coverage baselines. 1K/5K gate
   rule applies.
3. **Stage 2 proper — GameNGen-style noise-augmented feedback training
   WITH σ-bucket conditioning** (the GN5-validated mechanism + the missing
   ingredient). Rollout-context collection can piggyback on the
   capture-wraps-patch nesting built for this. Optionally folded into the
   same run as step 2 (CD loss × noised history) if the gate design stays
   clean.
4. **Head-v2 design question, gated separately:** windowed acoustic-history
   conditioning for the head (the dots.tts lever) as the identity-axis fix.
   Decide after 2–3 report; changes the head's input contract, so it is a
   new artifact tag, not a patch.
5. **Reserve — short asymmetric-DMD polish** (frozen DDPM head as real
   score, 15M online fake score) ONLY on top of a healthy init, per CF++'s
   mode-seeking warning. Est. 20–80 A100-h.
6. **Parked:** DAgger (BAgger template exists, video-only), GROW-style RL
   (fourth candidate; steal its reward pair for eval, not its method).

75K scale-up stays DEFERRED until 2–3 report (unchanged).

## Open items from the pass

- Resolve CF++ venue status before citing as "ICML" anything (only CF is
  ICML 2026; CF++ unconfirmed as of v2 2026-06-01).
- Skim VibeVoice ICLR reviews (OpenReview FihSkzyxdv) for long-form/drift
  discussion.
- Spot-check the snippet-only reads if cited: Mutual Forcing, BAgger,
  FlashTTS, RAVEN.
- Self Forcing (2506.08009) rollout/KV/gradient-truncation mechanics were
  inherited by reference in both CF papers — fetch before implementing
  step 5.
- arXiv indexing lag: a same-week scoop is not ruled out; re-run the sweep
  before submission.

## CRITIC PASS (2026-08-14, post-GN6) — corrections to the decision above

Requested by Josh ("double check our history and our current approach again
as a critic"). Four findings; the strategy survives, the sequence changes.

1. **Design flaw in step 3 as originally written: the head has no context
   input to noise.** Verified `src/flow_head/model.py:95` — the head takes
   (x_t, t, current hidden state), per-frame. GameNGen noises context the
   model RECEIVES; our head receives none. Noise-augmented training
   therefore requires a **new capture**: teacher runs with σ-noise injected
   into the acoustic feedback, caching the *perturbed* hidden states paired
   with the frozen DDPM head's outputs from those states — expert labels on
   corrupted states (a DAgger×GameNGen hybrid; stronger than either, but a
   capture spend, not a loss tweak). **Untested prerequisite: teacher
   health under feedback noise** — GN5/GN6 noised only student runs. → GN7.
2. **Sequencing error: causal-CD must not gate the main bet.** CF++'s
   mode-seeking warning applies to DMD (the reserve), not to supervised
   training on noised captures. CD targets the dispersion axis, not the
   GN5-proven correlated-error axis. Run it as a **parallel arm** on the
   existing cache, never as a serial prerequisite.
3. **Evidence-strength gap: every GN6 σ point is n=1 seed** and the σ=0.4
   anomaly shows the per-seed variance is enormous. Random-σ training is
   robust to this by design, but all stage-2 gates must run the collapse
   assay at 2–3 seeds, and the binary voice≥0.5 window metric has zero
   discrimination left (0% everywhere in GN6) — grade stage-2 identity on
   the continuous similarity curve instead.
4. **Capture schema must be designed for reuse:** record per-frame σ
   (bucket) AND a trailing window of K prior latents, so ONE capture serves
   the noise-trained head, the σ-conditioned head, and the head-v2
   history-conditioning experiment (the dots.tts identity lever). A
   re-capture is the expensive mistake to design away now.

**AMENDED SEQUENCE (operative):**
1. **GN7** (~$3–5): (a) teacher robustness under feedback noise at
   σ=0.2/0.3 — greenlights or kills the capture design; (b) σ=0.4 reroll
   at 2 new seeds — settles anomaly-vs-dead-zone.
2. **Capture v2** (the real spend, batched 4.9×): teacher runs, per-window
   random σ ∈ {0, 0.1–0.3}, schema = hidden state + DDPM target + σ bucket
   + trailing K latents.
3. **Training arms off the one cache:** (a) causal-CD on the clean subset;
   (b) noise-augmented σ-conditioned head; (c) optional combined. Gate:
   closed-loop collapse assay at 2–3 seeds, graded identity curve, Josh's
   ear.
4. **Reserve unchanged:** short asymmetric DMD only on a healthy init.
5. What survives the critique untouched: frozen-backbone constraints, N8 +
   product path, 75K deferral, and GN5's causal mechanism proof — still
   the strongest evidence in the program.
