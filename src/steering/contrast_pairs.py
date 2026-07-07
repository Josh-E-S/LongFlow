"""P0 Stage 1 — contrast-pair activation capture.

Records per-layer hidden states from the frozen Qwen2.5 backbone while VibeVoice
generates affect-contrasted renditions of identical neutral scripts, then
mean-pools them into per-(script, axis, pole, layer) steering-source vectors.

Deliberately duck-typed: nothing here imports vibevoice, so the math and hook
mechanics are unit-testable on synthetic modules (tests/test_steering.py) and the
VibeVoice-specific wiring lives in the experiment notebook.

Calibration finding (Colab L4, 2026-07-06): VibeVoice's inference runs a parallel
negative (CFG) stream through the same decoder layers, and only on speech-frame
steps — hook calls per generated token are NON-uniform (measured: 116 calls =
1 prefill + 60 positive + 55 negative for 61 tokens). So no fixed calls-per-step
assumption can work. Instead each hook call records the layer's `cache_position`;
the positive AR stream is recovered as the unique chain of seq-len-1 calls whose
positions continue the prompt (prompt_len, prompt_len+1, ...). The negative
stream's positions restart near zero and stay ~prompt_len below the positive
chain forever (both advance at most 1 per step), so classification is unambiguous
at any generation length.
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
    num_calls_total: int  # raw hook calls, for sanity checks


class LayerActivationRecorder:
    """Position-aware forward hooks on a list of decoder layers.

    Layers may return tuples (Qwen2DecoderLayer does); the hidden state is
    element 0. States are detached, moved to CPU, and stored fp32. Requires the
    layers to receive `cache_position` as a kwarg (transformers 4.51.x does).
    """

    def __init__(self, layers):
        self.layers = list(layers)
        self._states: list[list[torch.Tensor]] = [[] for _ in self.layers]
        self._meta: list[tuple[int, int]] = []  # (last cache_position, seq_len); layer 0 only
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer_idx == 0:
                cp = kwargs.get("cache_position")
                if cp is None:
                    raise RuntimeError(
                        "decoder layer did not receive cache_position kwarg — "
                        "expected the transformers==4.51.x call signature"
                    )
                self._meta.append((int(cp[-1]), hidden.shape[1]))
            self._states[layer_idx].append(hidden.detach().to("cpu", torch.float32))

        return hook

    def __enter__(self):
        self._handles = [
            layer.register_forward_hook(self._make_hook(i), with_kwargs=True)
            for i, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    @property
    def num_calls(self) -> int:
        return len(self._states[0]) if self._states else 0

    def step_states(self, prompt_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover the positive AR stream.

        Returns (states, token_indices):
          states [num_layers, n, d_model] — hidden states of positive AR calls;
          token_indices [n] — for each kept call, the index into the
          generated-token sequence of the token that call PREDICTED (i.e. the
          state VibeVoice conditions that token's diffusion on). Generated token
          0 is conditioned by the prefill call and is deliberately dropped.
        """
        counts = {len(s) for s in self._states}
        if len(counts) != 1:
            raise RuntimeError(f"layers saw different call counts: {sorted(counts)}")
        if len(self._meta) != self.num_calls:
            raise RuntimeError("metadata/call-count mismatch — hooks were tampered with")
        kept, token_indices = [], []
        expected = prompt_len
        for ci, (pos, seq_len) in enumerate(self._meta):
            if seq_len != 1:
                continue  # a prefill (positive prompt, or negative CFG seed)
            if pos == expected:
                kept.append(ci)
                token_indices.append(pos - prompt_len + 1)
                expected += 1
        if not kept:
            raise RuntimeError(
                f"no positive-stream calls found for prompt_len={prompt_len} — wrong prompt_len?"
            )
        states = torch.stack(
            [
                torch.cat([self._states[li][ci] for ci in kept], dim=1)[0]
                for li in range(len(self.layers))
            ]
        )
        return states, torch.tensor(token_indices)


def target_frame_mask(
    gen_ids: list[int],
    token_indices: torch.Tensor,
    *,
    frame_id: int,
    end_id: int | None = None,
) -> torch.Tensor:
    """Bool mask over step_states columns: speech frames in the target region.

    With end_id set, frames at or before the FIRST speech_end are excluded —
    that first segment is the spoken affect lead-in, which must not contribute
    to the pooled vector. With end_id=None, all speech frames are kept.
    """
    first_end = None
    if end_id is not None:
        first_end = next((i for i, t in enumerate(gen_ids) if t == end_id), None)
    mask = torch.zeros(len(token_indices), dtype=torch.bool)
    for j, g in enumerate(token_indices.tolist()):
        is_target_frame = 0 <= g < len(gen_ids) and gen_ids[g] == frame_id
        if is_target_frame and (first_end is None or g > first_end):
            mask[j] = True
    return mask


def drop_leading_true(mask: torch.Tensor, fraction: float) -> torch.Tensor:
    """Fallback lead-in exclusion when no turn markers exist: clear the first
    `fraction` of True entries (earliest frames) from a frame mask."""
    out = mask.clone()
    true_idx = mask.nonzero(as_tuple=True)[0]
    out[true_idx[: int(len(true_idx) * fraction)]] = False
    return out


def select_frames(step_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    L, n, d = step_states.shape
    if mask.shape != (n,):
        raise ValueError(f"mask shape {tuple(mask.shape)} != ({n},)")
    kept = step_states[:, mask, :]
    if kept.shape[1] == 0:
        raise ValueError("frame selection kept 0 frames")
    return kept


def pool_record(
    step_states: torch.Tensor,
    mask: torch.Tensor,
    *,
    script_id: str,
    axis: str,
    pole: str,
    sample_idx: int,
    num_calls_total: int,
) -> PairRecord:
    kept = select_frames(step_states, mask)
    vectors = kept.mean(dim=1)  # [L, d]
    if not torch.isfinite(vectors).all():
        raise ValueError("non-finite pooled vectors — refusing to record")
    return PairRecord(
        script_id=script_id,
        axis=axis,
        pole=pole,
        sample_idx=sample_idx,
        layer_vectors=vectors,
        num_frames_kept=int(kept.shape[1]),
        num_calls_total=num_calls_total,
    )


def save_records(records: list[PairRecord], path) -> None:
    torch.save([vars(r) for r in records], path)


def load_records(path) -> list[PairRecord]:
    return [PairRecord(**d) for d in torch.load(path, weights_only=True)]
