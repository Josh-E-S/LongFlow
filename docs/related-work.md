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

## Ecosystem notes

- VibeVoice-ASR (2601.18184, MSR, Mar 2026) exists — the backbone is a live
  research platform; good for the "system people care about" motivation.
- Community: Q8/4-bit quants, ComfyUI integration — engineering, not research.
