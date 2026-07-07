# Negative Results — TransplantTTS (March–April 2026)

These findings come from the TransplantTTS project (archived April 2026), the direct predecessor of LongFlow. They are recorded here because (a) each one shaped a LongFlow design decision, (b) several are independently publishable, and (c) the field wastes enormous compute re-discovering exactly these failure modes. Intended for the LongFlow paper appendix and related-work sections.

**Context.** TransplantTTS attempted to transplant VibeVoice's frozen 7.5 Hz continuous tokenizers onto a smaller Qwen3 backbone (0.6B → 1.7B) via SpeechConnector bridging, exploring both continuous (MSE regression) and discrete (codebook classification) output heads.

---

## N1 — MSE pre-training locks backbone representations into regression-shaped features

**Finding.** A backbone pre-trained with autoregressive MSE loss on continuous acoustic latents develops internal representations optimized for mean prediction. These representations are then incompatible with downstream objectives that need distributional structure — discrete classification heads and flow-matching heads both fail on top of them.

**Evidence.**
- Extended MSE pre-training followed by discrete codebook classification plateaued at ~32% cb0 accuracy — reproduced at both 1K and 75K sample scale. 75× more data could not overcome the representation mismatch.
- A short MSE curriculum (5–10 epochs) before switching objectives moved cb0 from 5% → 34%, confirming curriculum *duration* — not the curriculum itself — is the critical hyperparameter.
- MSE training itself worked as regression: loss 23.7 with 92% high-frequency preservation on memorized samples; pitch correlation 0.86–0.90 on unseen speakers. The structure was right; the distribution was collapsed (speech-band energy 0.45 on unseen data — the classic muffled mean-prediction signature).

**Implication for LongFlow.** Never condition a generative head on hidden states shaped by a regression objective. LongFlow conditions only on VibeVoice's own Qwen2.5 hidden states, which are pre-validated: every cached state already produced intelligible speech through the original diffusion head.

---

## N2 — Flow-matching head on MSE-shaped hidden states: architecture correct, conditioning fatal

**Finding.** A 15M-param DiT-style flow head (AdaLN MLP, no attention, OT-CFM loss) trained on frozen MSE-pretrained Qwen3 hidden states produced output statistically indistinguishable from the MSE baseline. The head architecture was not the problem; the conditioning signal was.

**Evidence (April 7–8, 2026 experiment).**
- Epoch 47 checkpoint, 10 Euler steps: RMS −16.0 dBFS (vs −15.6 baseline), zero-crossing rate ~1248/sec vs ~1274/sec baseline — both far below the 3,000–8,000/sec speech range.
- Acoustic texture was sharper than the MSE baseline (the flow head *was* generating), but intelligibility was worse — sharpness without correct phoneme content.
- Front-heavy energy distribution (6221 → 3910 first→second half) suggested partial but unsustained temporal structure.

**A cautionary note on self-persuasion.** Mid-experiment, we argued ourselves out of the hidden-state-quality concern ("the training objective of the backbone does not determine the quality of its internal representations, only what it was optimized to predict") and formally retracted the concern. The epoch-47 result refuted the retraction. The training objective *does* shape the representations, at least enough to starve a conditional generator. Lesson: a plausible architectural analogy (TADA/VibeVoice both use LM-states → head) does not transfer if the LM was trained under a different objective than the analogy assumes.

**Implication for LongFlow.** This experiment is LongFlow's existence proof-by-contradiction: identical head, identical recipe, different conditioning source. It also serves as the pre-registered baseline — if the same head succeeds on VibeVoice states, the conditioning hypothesis is confirmed.

---

## N3 — Cross-tokenizer translation has a hard ceiling

**Finding.** Predicting one codec's discrete tokens from a backbone trained in another codec's latent space ceilings at ~9% cb0 accuracy regardless of head architecture.

**Evidence.** Multiple head variants over Qwen3-TTS 12Hz 16-codebook targets from VibeVoice-latent-trained backbones; none escaped single-digit cb0.

**Implication.** Tokenizer spaces are not interchangeable views of the same audio; the representational geometry is codec-specific. LongFlow keeps input and output in VibeVoice's native latent space end-to-end.

---

## N4 — A matched codec pair does not rescue discrete prediction from an MSE backbone

**Finding.** Switching to DualCodec (so encoder and prediction targets share a codec family) still yielded 0.7% cb0 accuracy at 16,384 vocab — worse than the mismatched setup.

**Implication.** Confirms N1 is the binding constraint, not codec mismatch alone. Fixing the tokenizer pairing without fixing the conditioning representation fixes nothing.

---

## N5 — SpeechConnector capacity: 2-layer SiLU is the optimum, more is worse

**Finding.** A 2-layer SiLU SpeechConnector bridged the text-LM ↔ acoustic modality gap best; a 3-layer variant degraded results.

**Implication.** The modality bridge should be a thin adapter, not a learner. Extra capacity in the connector invites it to absorb (and distort) work that belongs to the backbone or head. LongFlow's flow head follows the same philosophy: ~15M params, no attention, minimal moving parts.

---

## N6 — What *did* transplant successfully (positive control)

For completeness — the frozen-tokenizer-transplant thesis itself was partially validated, which is why it isn't the thing LongFlow abandons:

- VibeVoice's frozen 7.5 Hz tokenizers transplanted to a new backbone with pitch correlation 0.86–0.90 on unseen speakers.
- A 4× smaller backbone (Qwen3-1.7B vs VibeVoice-7B) learned speech patterns from VibeVoice latents.
- The failure was never the tokenizers or the transplant — it was every attempt to train a *new* backbone's representations to condition generation. LongFlow's answer: don't train the backbone at all.

---

## Infrastructure footnote

The April 8 training crash (epoch 16, `_pin_memory_loop` file-descriptor failure on Modal, likely `/dev/shm` exhaustion) was unrelated to the scientific result but cost a run. Standing mitigations for all LongFlow training: `pin_memory=False`, `num_workers=2`, `persistent_workers=False` on Modal; prefer Vast.ai for iteration loops.

---

# Negative Results — LongFlow (July 2026 onward)

## N7 — Activation steering on VibeVoice: localization clean, perception capped by the backbone's affect ceiling

**Finding (P0, July 2026).** Contrast-pair activation steering (EmoSteer-style) on frozen VibeVoice-1.5B produces mechanically clean, turn-localized control — the unsteered speaker is untouched by ears and ECAPA, WER 0.000 and speaker-sim 0.97–1.06× inside the usable window — but the perceptual affect shift saturates at "subtle" and does not survive long-form listening as an effective control.

**Evidence.** LOSO-AUC 0.70–0.825 at layers 17/18 (real transferable internal signal); natural prompted-affect contrast only ~6% of the residual-stream norm at L17; steering at α=1 (the natural gap) audible-but-subtle, α≥2 degrades timbre (ECAPA 0.63–0.83×) and can catastrophically derail content (a Whisper-hallucinated clip) before affect strengthens. Critically, VibeVoice's own *prompted* contrast pairs — the ceiling for any extraction-based method — were themselves only subtly different.

**Implication.** Steering cannot exceed the expressive dynamic range the backbone exposes; VibeVoice is stability-first with little affect range. Full record: `experiments/p0_steering/NOTES.md`. Retained assets: turn-localized injection machinery (positive-stream targeting through the CFG double-stream, segment gating) and the eval pipeline. C2 dropped from the paper's claims; LongFlow proceeds on C1+C3+C4 per the pre-registered P0 plan.

---

## Rejected alternative architectures

See README § "Rejected alternatives." Summary: Mamba-Flow-TTS (from-scratch AR-MSE backbone + flow head = N1/N2 recipe with a Mamba swap; no dialogue data; unsupervised VAD controller), Endurance v2 (sound, but single-speaker by design — retained as follow-on work).
