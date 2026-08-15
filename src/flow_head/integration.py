"""Closed-loop integration: VibeVoice generation with the flow head installed.

Versioned after review-adversarial.md flagged the notebook-only patch as
unauditable. Guarantees by construction: fresh independent x0 noise per frame
(generator=None -> global entropy), safe method restoration, and per-frame
timing/latent recording for drift analysis.
"""

import time

import torch

# sampler injected via constructor (default heun_sample per E1 audit)
from src.flow_head.model import FlowHead


class FlowHeadPatch:
    """Context manager replacing `model.sample_speech_tokens` with the flow head.

    latent_mean/latent_std: de-standardization stats from the training
    checkpoint. Records per-call wall time and every produced latent (CPU
    fp32) so closed-loop dispersion/drift can be analyzed after any run.
    """

    def __init__(
        self,
        model,
        head: FlowHead,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        nfe: int = 8,
        sway: float = 0.0,
        sampler=None,
    ):
        from src.flow_head.cfm import heun_sample

        self.model = model
        self.head = head
        device = next(head.parameters()).device
        self.mean = latent_mean.to(device)
        self.std = latent_std.to(device)
        self.nfe = nfe
        self.sway = sway
        # default heun_sample per the E1 audit: euler4/sway-1 under-disperses
        # even a perfect field; heun8 matches teacher dispersion exactly
        self.sampler = sampler or heun_sample
        self.calls = 0
        self.time_s = 0.0
        self.latents: list[torch.Tensor] = []

    def __enter__(self):
        # mirror SampleCapture's semantics so the two context managers nest
        # (the DAgger probe wraps capture around the patch — audit finding 4)
        self._was_instance_attr = "sample_speech_tokens" in vars(self.model)
        self._orig = getattr(self.model, "sample_speech_tokens", None)
        patch = self

        def flow_sample(condition, neg_condition=None, cfg_scale=None):
            t0 = time.time()
            patch.calls += 1
            z = patch.sampler(
                patch.head,
                condition.float(),
                patch.head.cfg.d_latent,
                nfe=patch.nfe,
                sway=patch.sway,
                generator=None,  # fresh independent x0 EVERY frame — never seeded
            )
            z = z * patch.std + patch.mean
            patch.time_s += time.time() - t0
            patch.latents.append(z.detach().float().cpu())
            return z.to(condition.dtype)

        self.model.sample_speech_tokens = flow_sample
        return self

    def __exit__(self, *exc):
        if self._was_instance_attr:
            self.model.sample_speech_tokens = self._orig  # restore outer wrapper
        elif "sample_speech_tokens" in vars(self.model):
            del self.model.sample_speech_tokens  # restore class-method lookup
        self._orig = None
        return False

    def latent_stats(self, segments: int = 4) -> dict:
        """Std over equal segments of the generated sequence — the drift curve.

        segments=8 matches the endurance-test analysis (NOTES 2026-07-08);
        keep it fixed across runs being compared."""
        if not self.latents:
            return {}
        zs = torch.cat(self.latents)
        q = max(len(zs) // segments, 1)
        return {
            "frames": len(zs),
            "std_overall": float(zs.std()),
            "std_segments": [float(zs[i * q : (i + 1) * q].std()) for i in range(segments)],
        }


class _CFGField:
    """Head adapter presenting the sampler's head(x_t, t, condition) interface
    while evaluating the CFG-combined velocity field:

        v = v_neg + cfg_scale * (v_cond - v_neg)

    This mirrors what VibeVoice's own DDPM head does with the two LM streams —
    the flow head had been silently discarding neg_condition/cfg_scale since
    July (2026-08-15 audit finding), sampling the unguided field instead.
    """

    def __init__(self, head: FlowHead, neg_condition: torch.Tensor, cfg_scale: float):
        self.head = head
        self.cfg = head.cfg  # FlowHeadPatch and samplers read .cfg.d_latent
        self.neg = neg_condition
        self.scale = cfg_scale

    def __call__(self, x_t, t, condition):
        v_cond = self.head(x_t, t, condition)
        v_neg = self.head(x_t, t, self.neg)
        return v_neg + self.scale * (v_cond - v_neg)


class CFGFlowHeadPatch(FlowHeadPatch):
    """FlowHeadPatch that honors neg_condition/cfg_scale instead of dropping
    them. Falls back to the unguided field when the caller passes no
    neg_condition (e.g. cfg disabled upstream). Costs one extra head eval per
    sampler step — negligible at the head's ~6% share of the loop."""

    def __init__(self, *args, cfg_scale: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # None -> trust the per-call cfg_scale VibeVoice passes; a float pins it
        self.cfg_scale_override = cfg_scale

    def __enter__(self):
        self._was_instance_attr = "sample_speech_tokens" in vars(self.model)
        self._orig = getattr(self.model, "sample_speech_tokens", None)
        patch = self

        def flow_sample_cfg(condition, neg_condition=None, cfg_scale=None):
            t0 = time.time()
            patch.calls += 1
            scale = patch.cfg_scale_override if patch.cfg_scale_override is not None else cfg_scale
            if neg_condition is not None and scale is not None and scale != 1.0:
                field = _CFGField(patch.head, neg_condition.float(), float(scale))
            else:
                field = patch.head
            z = patch.sampler(
                field,
                condition.float(),
                patch.head.cfg.d_latent,
                nfe=patch.nfe,
                sway=patch.sway,
                generator=None,  # fresh independent x0 EVERY frame — never seeded
            )
            z = z * patch.std + patch.mean
            patch.time_s += time.time() - t0
            patch.latents.append(z.detach().float().cpu())
            return z.to(condition.dtype)

        self.model.sample_speech_tokens = flow_sample_cfg
        return self
