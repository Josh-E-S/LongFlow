# CLAUDE.md — LongFlow

Project context for Claude Code. Read this fully before writing any code.

## What this project is

LongFlow makes VibeVoice — the only open 90-minute, 4-speaker TTS system — fast, emotion-controllable, and drift-free **without training the backbone or tokenizers**. Four contributions, each a bolt-on to a frozen VibeVoice:

- **C1** — Replace the 10-step DDPM diffusion head with a ~15M-param flow-matching head (OT-CFM 4-NFE baseline → MeanFlow 1–2 NFE), trained on cached hidden states. Target 15–20× end-to-end speedup.
- **C2** — Training-free per-speaker emotion control: activation steering vectors along continuous valence/arousal axes, injected into the frozen Qwen2.5 backbone, localized to a speaker's turns.
- **C3** — Inference-time speaker anchoring: blend immutable reference embedding with a running buffer of the speaker's generated turns.
- **C4** — Long-horizon consistency benchmark: speaker-drift curves, durational WER, windowed UTMOS at 30s → 90min.

Full scope, positioning, data plan, and phases: `README.md`. Paper target: Interspeech 2026 / arXiv.

## Current state (July 2026)

- Repo contains README, `docs/` (architecture, negative results, **`resources.md` — pinned repos/weights/datasets/eval stack, verified 2026-07-05, read it before touching any external dependency**), the package scaffold (`src/`, `tests/`, uv-managed Python 3.11 env, ruff + pre-commit), and the frame-alignment guard `src/cache/alignment.py`. No model/experiment code yet.
- Next task is **P0: the steering gate check** — spec in `docs/experiments/p0-steering.md`. It is deliberately first: cheapest experiment, validates the highest-risk contribution (C2), no training required.
- Phase order after P0: P1 flow-head baseline → P2 MeanFlow 1–2 NFE → P3 anchoring + benchmark → P4 encoder distillation (stretch) → P5 paper.

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
- Every experiment gets a directory under `experiments/` with a `NOTES.md` — record hypothesis, config, result, and verdict *even when it fails*. Negative results are first-class citizens in this project.
- Shape-and-gradient tests in `tests/` before any training script runs (synthetic tensors, NaN/Inf checks, T-alignment assertion).
- Audio artifacts for listening checks go to `experiments/<name>/audio/` — never committed to git except curated demo samples in `samples/` (git-lfs).
- Tag a git release at each phase gate (`p0-pass`, `p1-baseline`, ...) so paper numbers map to commits.

## Things Claude Code should push back on

- Any suggestion to "just fine-tune the backbone a little" — see constraint 1.
- Scaling data to fix a failing gate check — diagnose the representation/alignment issue instead.
- Adding attention or depth to the flow head before the MLP version has been fully evaluated.
- Skipping the listening step. Metrics missed the April 7 failure mode until a human listened.
- Streaming inference, 5+ speakers, or multilingual — explicitly out of scope for this paper (see README non-goals in "Rejected alternatives").
