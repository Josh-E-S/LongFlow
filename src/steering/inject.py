"""P0 Stage 3 — steering-vector injection (spec: p0-steering.md Stage 3).

Adds alpha * direction to the residual stream (decoder-layer output) during
generation, ONLY on the positive AR stream — the CFG negative stream and both
prefills pass through untouched. Positive-stream identification reuses the
cache_position-chain logic validated in contrast_pairs.LayerActivationRecorder.

Turn localization (the multi-speaker probe) is handled by an optional `gate`
callable evaluated per steered call — the experiment notebook wires it to a
segment counter fed by the generated token stream.

Duck-typed like the rest of src/steering: no vibevoice imports, testable on
synthetic modules.
"""

import torch


class SteeringInjector:
    """Context manager registering steering hooks on selected decoder layers.

    directions: {layer_idx: unit direction tensor [d_model]} — layer_idx into
      the `layers` list. Directions are cast to the hidden state's dtype/device
      at call time.
    alpha: scalar steering strength (spec sweep: 0.5, 1, 2, 4, 8).
    prompt_len: length of the positive prompt; positive AR calls are the chain
      of seq-len-1 calls with cache_position == prompt_len, prompt_len+1, ...
    gate: optional () -> bool; when given, steering applies only where it
      returns True (e.g. "only speaker B's turns"). Evaluated once per
      positive-stream call, per layer.
    """

    def __init__(
        self, layers, directions: dict[int, torch.Tensor], alpha: float, prompt_len: int, gate=None
    ):
        self.layers = list(layers)
        for li in directions:
            if not 0 <= li < len(self.layers):
                raise ValueError(f"layer index {li} out of range 0..{len(self.layers) - 1}")
        self.directions = {li: d.flatten() for li, d in directions.items()}
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)
        self.gate = gate
        self.steered_calls = 0  # positive-stream calls where steering was applied
        self._expected = {li: self.prompt_len for li in self.directions}
        self._handles = []

    def _make_hook(self, layer_idx: int):
        direction = self.directions[layer_idx]

        def hook(_module, _args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] != 1:
                return output  # prefill (positive or negative seed): never steer
            cp = kwargs.get("cache_position")
            if cp is None:
                raise RuntimeError(
                    "decoder layer did not receive cache_position kwarg — "
                    "expected the transformers==4.51.x call signature"
                )
            if int(cp[-1]) != self._expected[layer_idx]:
                return output  # negative CFG stream: never steer
            self._expected[layer_idx] += 1
            if self.gate is not None and not self.gate():
                return output
            if layer_idx == min(self.directions):
                self.steered_calls += 1  # count once per step, not per layer
            steered = hidden + self.alpha * direction.to(hidden.dtype).to(hidden.device)
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        return hook

    def __enter__(self):
        self._handles = [
            self.layers[li].register_forward_hook(self._make_hook(li), with_kwargs=True)
            for li in self.directions
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class SegmentGate:
    """Turn-localized gating for the multi-speaker probe.

    The notebook increments `segment` (via a logits processor or manual loop)
    every time a speech_end token is emitted; steering applies only while the
    current segment index is in `target_segments`.
    """

    def __init__(self, target_segments: set[int]):
        self.segment = 0
        self.target_segments = set(target_segments)

    def advance(self):
        self.segment += 1

    def __call__(self) -> bool:
        return self.segment in self.target_segments
