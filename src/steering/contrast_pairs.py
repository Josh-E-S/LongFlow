"""P0 Stage 1 — contrast-pair activation capture.

Records per-layer hidden states from the frozen Qwen2.5 backbone while VibeVoice
generates affect-contrasted renditions of identical neutral scripts, then
mean-pools them into per-(script, axis, pole, layer) steering-source vectors.

Deliberately duck-typed: nothing here imports vibevoice, so the math and hook
mechanics are unit-testable on synthetic modules (tests/test_steering.py) and the
VibeVoice-specific wiring lives in the experiment notebook.

Two assumptions MUST be calibrated in-session before trusting output
(see the calibration cell in experiments/p0_steering/stage1_colab.ipynb):

1. `calls_per_step` — VibeVoice's inference keeps a parallel negative KV cache
   for CFG, which may drive a second forward pass through the decoder layers
   each AR step. If so, hooks fire twice per generated token: set
   calls_per_step=2 and keep_call to whichever index is the positive pass.
2. Frame attribution — the affect lead-in sentence is spoken too. If the emitted
   token stream exposes per-turn speech_start/speech_end boundaries, mask to the
   target turn's frames; otherwise fall back to dropping the first
   `drop_first_fraction` of captured frames (lead-in is deliberately short
   relative to the target text).
"""

from dataclasses import dataclass

import torch


@dataclass
class PairRecord:
    script_id: str
    axis: str  # "arousal" | "valence"
    pole: str  # "pos" | "neg"
    sample_idx: int
    layer_vectors: torch.Tensor  # [num_layers, d_model], mean over kept frames
    num_frames_kept: int
    num_calls_total: int  # raw hook calls, for calibration sanity checks


class LayerActivationRecorder:
    """Forward hooks on a list of decoder layers, buffering per-call hidden states.

    Call pattern per layer during HF generate with KV cache:
      call 0: prefill (seq_len == prompt length) -- always skipped,
      calls 1..N: one per AR step (seq_len == 1), x calls_per_step if CFG
      runs a parallel negative pass through the same modules.

    Layers may return tuples (Qwen2DecoderLayer does); the hidden state is
    element 0. States are detached, moved to CPU, and stored fp32.
    """

    def __init__(self, layers, calls_per_step: int = 1, keep_call: int = 0):
        if calls_per_step < 1 or not (0 <= keep_call < calls_per_step):
            raise ValueError(f"bad calibration: {calls_per_step=}, {keep_call=}")
        self.layers = list(layers)
        self.calls_per_step = calls_per_step
        self.keep_call = keep_call
        self._buffers: list[list[torch.Tensor]] = [[] for _ in self.layers]
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._buffers[layer_idx].append(hidden.detach().to("cpu", torch.float32))

        return hook

    def __enter__(self):
        self._handles = [
            layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    @property
    def num_calls(self) -> int:
        return len(self._buffers[0]) if self._buffers else 0

    def step_states(self) -> torch.Tensor:
        """Per-AR-step hidden states, calibration applied.

        Returns [num_layers, num_steps, d_model]: prefill call dropped, then one
        kept call per step (keep_call within each calls_per_step group), squeezed
        from seq_len 1. Raises if the call pattern contradicts the calibration.
        """
        counts = {len(b) for b in self._buffers}
        if len(counts) != 1:
            raise RuntimeError(f"layers saw different call counts: {sorted(counts)}")
        per_layer = []
        for buf in self._buffers:
            ar_calls = [t for t in buf[1:]]  # drop prefill
            if len(ar_calls) % self.calls_per_step != 0:
                raise RuntimeError(
                    f"{len(ar_calls)} AR calls not divisible by calls_per_step="
                    f"{self.calls_per_step} — recalibrate before trusting output"
                )
            kept = ar_calls[self.keep_call :: self.calls_per_step]
            for t in kept:
                if t.shape[1] != 1:
                    raise RuntimeError(f"expected seq_len 1 AR call, got shape {tuple(t.shape)}")
            per_layer.append(torch.cat(kept, dim=1)[0])  # [num_steps, d]
        return torch.stack(per_layer)  # [L, num_steps, d]


def select_frames(
    step_states: torch.Tensor,
    speech_frame_mask: torch.Tensor | None = None,
    drop_first_fraction: float = 0.0,
) -> torch.Tensor:
    """Keep speech frames belonging to the target text.

    speech_frame_mask: bool [num_steps], True where the emitted token was a
    speech frame inside the target region (preferred, from token-stream parsing).
    drop_first_fraction: fallback lead-in exclusion when no mask is available.
    """
    L, n, d = step_states.shape
    if speech_frame_mask is not None:
        if speech_frame_mask.shape != (n,):
            raise ValueError(f"mask shape {tuple(speech_frame_mask.shape)} != ({n},)")
        kept = step_states[:, speech_frame_mask, :]
    else:
        start = int(n * drop_first_fraction)
        kept = step_states[:, start:, :]
    if kept.shape[1] == 0:
        raise ValueError("frame selection kept 0 frames")
    return kept


def pool_record(
    step_states: torch.Tensor,
    *,
    script_id: str,
    axis: str,
    pole: str,
    sample_idx: int,
    num_calls_total: int,
    speech_frame_mask: torch.Tensor | None = None,
    drop_first_fraction: float = 0.0,
) -> PairRecord:
    kept = select_frames(step_states, speech_frame_mask, drop_first_fraction)
    return PairRecord(
        script_id=script_id,
        axis=axis,
        pole=pole,
        sample_idx=sample_idx,
        layer_vectors=kept.mean(dim=1),  # [L, d]
        num_frames_kept=kept.shape[1],
        num_calls_total=num_calls_total,
    )


def save_records(records: list[PairRecord], path) -> None:
    torch.save([vars(r) for r in records], path)


def load_records(path) -> list[PairRecord]:
    return [PairRecord(**d) for d in torch.load(path, weights_only=True)]
