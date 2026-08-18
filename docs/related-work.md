# Related work + novelty positioning

Populated from the 2026-07-07 novelty sweep. **Re-run the sweep at every phase
gate and immediately before paper submission** — this field moves in weeks.

## C1 (flow-head replacement) — the crowded-but-clear lane

**Nobody distills, accelerates, or replaces VibeVoice's diffusion head.** Community
activity is quantization + step-count knobs only. But the neighborhood:

- **dots.tts** (arXiv 2606.07080, rednote, Jun 2026) — **closest prior art, cite and
  position head-on.** MeanFlow-distilled flow head on their own 2B continuous-AR TTS,
  85/54ms first-packet. Differences that preserve our novelty: they distill their own
  end-to-end-trained head; we retrofit a **frozen third-party** system via cached
  pairs from the teacher's own inference, 16M vs 123M (7.7×), plus the mechanistic
  states-provenance finding they have nothing like. **Timeline pressure: they hold
  the "MeanFlow head for AR TTS" headline — move briskly on P2.**
- **FAR** (2504.18391) — replaces MAR's diffusion head with a lightweight shortcut
  head, in images. The conceptual antecedent of the exact move; cite.
- **TMD** (NVIDIA, 2601.09881) — flow head grafted onto a frozen video diffusion
  backbone, distilled from backbone representations. Closest "frozen backbone +
  small flow head" precedent, wrong modality; cite.
- Secondary: DiSA (2505.20297), IntMeanFlow (2510.07979), LatentLM (2412.08635 —
  the natural citation for the regression-pretrained-backbone contrast), VoxCPM2,
  ZipVoice (2506.13053), DMDSpeech (2410.11097).

**Our C1 claim, phrased defensibly:** first offline head-distillation of a frozen
third-party long-form TTS system from its own cached inference states, with a
documented failure/success contrast on conditioning-state provenance (N1/N2 vs P1
gate). NOT "first few-step flow head for TTS."

## C4 (long-horizon benchmark) — neighbors moved in during 2026

- **SwanBench-Speech** (2605.28618, May 2026) — closest in intent: 1,101 samples,
  17 scenarios, 7 metrics incl. consistency. Differentiate on **duration horizon
  (30s–90min) and time-resolved drift curves**; read the full paper before P3
  (max durations UNVERIFIED from abstract).
- **SpeechSSM / LibriSpeech-Long** (2412.18603, Google) — quality-vs-time curves
  but single-speaker, ~16 min max.
- **MagpieTTS-LF** (2606.18485) — method paper whose eval protocol (per-chunk
  speaker-sim, UTMOSv2 stability) overlaps ours; cite the protocol.
- **MOSS-TTSD / TTSD-eval** (2603.19739) — 60-min multi-party synthesis with
  speaker-attribution eval; adjacent, narrower eval.
- Term collision: "speaker drift" used at utterance scale by 2604.06327.

**Our C4 claim:** the 90-minute multi-speaker horizon with drift *curves*
(degradation as a function of position) is unclaimed. Frame explicitly against
SwanBench-Speech.

## P0 negative result — engagement obligations

- **EmoSteer-TTS** (2508.03543) — engage head-on: our result contradicts its
  optimism in a different architecture class (AR next-token diffusion vs parallel
  flow-DiT). The class contrast IS the framing. Also: TADA (2602.11910), DUET
  (2606.00066).
- **PsiPi's community method** (HF `microsoft/VibeVoice-1.5B` discussion #12, Aug
  2025; credited in the fork README) — reference-audio trick: same voice recorded
  at 4 emotion intensities loaded into the 4 speaker slots, script annotated with
  pseudo-speaker labels. Requires multiple recordings per voice; not activation-
  level; unpublished. Footnote as the community's working alternative — consistent
  with our finding that VibeVoice's internals don't expose steerable affect range.

## Inworld TTS-1 (2507.21138, Jul 2025) — read 2026-08-17 (Josh flagged it)

Discrete-token AR TTS: X-codec2 (single codebook 65,536, 50 tok/s) on
LLaMA-3.2-1B / 3.1-8B, 48 kHz decoder, 11 languages, MIT **training code**
(SFT + multi-node vLLM RL) — but **no weights released anywhere** (GitHub +
HF both checked). Relevance to us, in order:

1. **Stage-2 evidence + reward design.** Their GRPO alignment stage uses a
   composite reward that is literally our two closed-loop failure axes plus
   quality: `R = exp(−2.5·WER)` (Whisper-large-v3) + WavLM speaker-sim +
   DNSMOS, equal weights, 8 rollouts/prompt on a 1K-hour subset. Second
   sighting of on-policy RL for TTS (with GROW 2608.03215) and the earlier
   one; their open-sourced RL pipeline is the infra reference. Caveat: their
   policy is a discrete-token LM — gradients through token logits; our
   student is a continuous flow head, so the recipe transfers as *reward
   design + evidence*, not drop-in code (GROW's advantage-weighted flow
   objective remains the mechanically-matching variant).
2. **Long-form positioning sentence.** Max training sequence 2,048 tokens ≈
   40 s of audio; they note longer generations degrade and offer nothing
   beyond streaming chunking. A 2025 top-rated commercial-grade system caps
   at ~40 s — the 90-min territory (N8, drift curves, chunked-parallel
   renders) stays unclaimed. Cite in C4 framing.
3. **Emotion markups = the trained-affect contrast for N7.** 8 styles + 7
   non-verbals via LoRA (r16, α32) on ~180 h of paired neutral/stylized data
   (style tag as transcript delimiter, 0.5–1.5 s silence join). This is what
   affect control costs when the backbone must LEARN it — supports N7's
   conclusion that training-free steering can't exceed the backbone's affect
   range. In-paper: one contrast sentence. (A VibeVoice-LoRA emotion port
   using their exact data recipe is a clean SEPARATE project — it violates
   LongFlow constraint 1 by design, so it lives outside this repo if ever.)
4. Not relevant to C1: discrete-codec architecture (constraint 4 divide);
   no diffusion/flow component to distill.

## Ecosystem notes

- VibeVoice-ASR (2601.18184, MSR, Mar 2026) exists — the backbone is a live
  research platform; good for the "system people care about" motivation.
- Community: Q8/4-bit quants, ComfyUI integration — engineering, not research.
