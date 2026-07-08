# Pre-75K fresh-eyes review — findings + experiment plan

Two clean-context reviewers were launched 2026-07-08 to attack the plan before
the ~35 GPU-h spend. Reviewer 1 (cheap-experiments hunter) reported; its full
findings are distilled below. Reviewer 2 (adversarial diagnosis review +
CFG-distillation literature) was still running at session end — if its report
is lost, re-run: an adversarial review of the drift diagnosis focusing on
CFG-unaware distillation (DMD / dots.tts "CFG-aware" / ZipVoice-Distill /
DSFlow), alternative under-dispersion explanations, and the training recipe.

## HARD GATE (do not violate)

**Do NOT start the 75K cache until the CFG-dispersion check (Exp 2) resolves.**
If CFG extrapolation is the source of the missing dispersion, the cache schema
must add `neg_condition` (one-line edit to `wrapped` in src/cache/capture.py —
it's already an argument) and the head needs a second conditioning input.
Getting this wrong wastes the entire 35 GPU-h run.

## Status of reviewer-1 findings

- **Exp 0 (seed bug audit): CLEARED.** The e2e FlowHeadPatch seeds a fresh
  generator per frame (manual_seed(patch.calls), incrementing); offline renders
  draw per-frame noise inside euler_sample. Not the cause.

## The cheap-experiment day (≈6 GPU-h total, one L4 session), in order

1. **Conditional-spread decomposition** (~30 min, no training): ~20 cached
   conditions; sample flow head 64× per condition (different seeds) AND teacher
   head 64× on the same conditions. Compare intra-condition (noise-driven) vs
   inter-condition spread, standardized space.
   Decision: head conditional spread ≈ teacher's but inter-condition muted →
   plain underfit → scaling plan OK. Head conditional spread ≈ 0 with teacher's
   large → mean-collapse/missing input → design change first.
2. **CFG dispersion measurement** (~45 min, no training) — THE GATE: capture
   ~20 utts at cfg_scale ∈ {1.0, 1.3, 3.0}; compare per-dim latent std; listen
   to one clip each. If std(cfg=1.0)/std(cfg=3.0) ≈ 0.6–0.7 (matching our
   head's 0.63 deficit) → smoking gun → add neg_condition to cache schema +
   head conditioning; then a 300–500-utt (pos,neg,latent) capture and a 5K-step
   pos-only vs concat A/B before the big run.
3. **Closed-loop inference-knob sweep** (~45 min, no training): NFE {4,8,32} ×
   sway {0,−1}; x0 temperature {0.9,1.0,1.2,1.4} (NOTE: we're UNDER-dispersed —
   τ>1 is the interesting direction); stochastic-churn Euler (x += dt·v +
   σ√dt·ε mid-steps); OU-correlated x0 across frames (ρ≈0.8–0.95; feedback
   encoder may need temporal coherence); EMA vs raw weights
   (load_checkpoint(use_ema=False)). Measure fade-onset time + latent-std-over-
   time (this curve IS the C4 machinery — build once).
4. **Train-longer curve** (~1–1.5 h): resume 10K ckpt → 80K steps, ckpt every
   5–10K; plot dispersion-ratio vs steps (log-x). Trend → ≥0.9 in-budget = go;
   plateau ≤0.75 by 60–80K = steps alone insufficient.
5. **Capacity probe** (~45 min): width-1024 (~40M) head, 10–20K steps, same
   data. If it reaches width-640's dispersion in ≤half the steps → big run uses
   the wider head (runtime cost negligible: head is 8ms/6% of step).
6. **Objective tweaks** (~40 min): logit-normal t shifted high; loss weight
   w(t)=1+2t. ≥5-point dispersion gain at matched steps → fold into recipe.
7. **On-policy DAgger probe** (~2 h, only if 1–2 don't explain): flow head
   drives the loop while the ORIGINAL sample_speech_tokens computes teacher
   targets per frame on the same conditions (neg stream still runs, so real
   cfg targets available); ~500 utts; fine-tune 2–3K steps on 80/20
   off/on-policy mix; re-run closed loop. Signal → budget an on-policy phase
   into the big run.

## Context for whoever picks this up

- Full state: experiments/p1_flow_head/NOTES.md (integration findings section).
- Closed-loop fade + under-dispersion (std 0.65 vs 1.0, constant over time);
  ×1.79 linear rescale falsified (fine structure, not amplitude).
- Profile: our head 8ms/frame (6%); parent head 67ms (35%); e2e head-swap
  speedup 1.47× (Amdahl); cfg_scale=1.0 does NOT disable the negative stream.
- Strategic decision (Josh, 2026-07-08): stay on VibeVoice; hedge by making the
  C4 benchmark multi-system (VibeVoice + MOSS-TTSD + SoulX-Podcast) — pending
  a capability-verification pass on those two systems. "Compose a new system
  from pieces" rejected (= TransplantTTS graveyard, N1–N6).
