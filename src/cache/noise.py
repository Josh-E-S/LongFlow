"""Acoustic-connector feedback noise injection — promoted from the GN5/GN6
Colab notebooks to src for capture v2 (the sigma-noised teacher captures
need the exact same intervention the GN5-GN8 renders used, not a
reimplementation that could silently drift from it).

Hooks `model.model.acoustic_connector`'s forward output, corrupting it with
Gaussian noise scaled by a running-std EMA of the connector's own output
distribution. Untouched during prompt encoding (active_fn gate) -- noising
prompt encoding was the v1 mistake corrected in GN6. Byte-for-byte the
GN5 ConnectorIntervention / GN6 NoiseIntervention mechanism, plus one
addition for capture v2: `last_sigma`, so a capture wrapper can record which
sigma bucket was actually applied to the frame it just captured.
"""

import torch


class NoiseIntervention:
    """sigma_fn: callable(call_count) -> sigma, so schedules (ramps, per-window
    randomization) are one lambda away. calls_fn: callable() -> int, wired by
    the caller to whatever driving loop (FlowHeadPatch.calls, a manual
    counter, etc.) counts frames -- defaults to always-0 (constant sigma_fn
    only) so the class has no hidden dependency on a specific patch."""

    def __init__(self, module, sigma_fn, active_fn=None, calls_fn=None):
        self.module = module
        self.sigma_fn = sigma_fn
        self.active_fn = active_fn or (lambda: True)
        self.calls_fn = calls_fn or (lambda: 0)
        self.mu = None
        self.var = None
        self.h = None
        self.last_sigma = 0.0

    def __enter__(self):
        def hook(mod, args, out):
            if not self.active_fn():  # untouched during prompt processing
                self.last_sigma = 0.0
                return out
            t = out[0] if isinstance(out, tuple) else out
            with torch.no_grad():
                if self.mu is None:
                    self.mu = t.mean().detach()
                    self.var = t.var().detach()
                else:
                    self.mu = 0.99 * self.mu + 0.01 * t.mean().detach()
                    self.var = 0.99 * self.var + 0.01 * t.var().detach()
                sigma = self.sigma_fn(self.calls_fn())
                self.last_sigma = float(sigma)
                if sigma <= 0:
                    return out
                new = t + torch.randn_like(t) * (sigma * self.var.sqrt())
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        self.h = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.h:
            self.h.remove()
        self.h = None
        return False
