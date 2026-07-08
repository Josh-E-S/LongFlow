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
