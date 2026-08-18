"""OT-CFM objective + Euler sampler (P1 baseline; hand-rolled per resources.md §2).

Loss (the F5-TTS/ZipVoice-lineage linear interpolant, independent coupling):
    t ~ U(0,1);  x0 ~ N(0,I);  x_t = (1-t)·x0 + t·x1;  target v = x1 - x0
    loss = MSE(head(x_t, t, cond), v)

Sampler: explicit Euler from x0 ~ N(0,I) over an increasing t-grid. Optional
sway warping (F5-TTS) concentrates steps where they matter at few-NFE.
"""

import math

import torch


def cfm_loss(
    head,
    x1: torch.Tensor,
    condition: torch.Tensor,
    generator=None,
    neg_condition: torch.Tensor | None = None,
    sigma_bucket: torch.Tensor | None = None,
) -> torch.Tensor:
    """x1: clean latents [B, d_latent]; condition: [B, d_model].

    neg_condition/sigma_bucket: head-v2 conditioning, forwarded verbatim; the
    head itself refuses a mismatch with its config in either direction."""
    if not torch.isfinite(x1).all() or not torch.isfinite(condition).all():
        raise ValueError("non-finite training batch — refusing to step")
    if neg_condition is not None and not torch.isfinite(neg_condition).all():
        raise ValueError("non-finite neg_condition batch — refusing to step")
    b = x1.shape[0]
    t = torch.rand(b, device=x1.device, generator=generator)
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    v_target = x1 - x0
    if neg_condition is None and sigma_bucket is None:
        v_pred = head(x_t, t, condition)  # v1 heads keep the 3-arg call
    else:
        v_pred = head(x_t, t, condition, neg_condition=neg_condition, sigma_bucket=sigma_bucket)
    return torch.nn.functional.mse_loss(v_pred, v_target)


def sway_grid(nfe: int, coef: float = -1.0, device="cpu") -> torch.Tensor:
    """t-grid of nfe+1 points in [0,1]; coef<0 front-loads steps (F5-TTS sway)."""
    u = torch.linspace(0, 1, nfe + 1, device=device)
    if coef == 0.0:
        return u
    return u + coef * (torch.cos(math.pi / 2 * u) - 1 + u)


@torch.no_grad()
def heun_sample(
    head,
    condition: torch.Tensor,
    d_latent: int,
    nfe: int = 8,
    sway: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """Heun (2nd-order) sampler: 2 function evals per step; better curvature
    tracking near t=1 where the distribution re-expands (review-adversarial.md
    §2b — Euler's linearization under-disperses there)."""
    b = condition.shape[0]
    device = condition.device
    grid = sway_grid(nfe, sway, device=device)
    x = torch.randn(b, d_latent, device=device, generator=generator)
    for i in range(nfe):
        dt = grid[i + 1] - grid[i]
        v1 = head(x, grid[i].expand(b), condition)
        v2 = head(x + dt * v1, grid[i + 1].expand(b), condition)
        x = x + dt * 0.5 * (v1 + v2)
    if not torch.isfinite(x).all():
        raise RuntimeError("non-finite sample — head diverged")
    return x


@torch.no_grad()
def euler_sample(
    head,
    condition: torch.Tensor,
    d_latent: int,
    nfe: int = 4,
    sway: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """condition [B, d_model] -> sampled latents [B, d_latent]."""
    b = condition.shape[0]
    device = condition.device
    grid = sway_grid(nfe, sway, device=device)
    x = torch.randn(b, d_latent, device=device, generator=generator)
    for i in range(nfe):
        t = grid[i].expand(b)
        v = head(x, t, condition)
        x = x + (grid[i + 1] - grid[i]) * v
    if not torch.isfinite(x).all():
        raise RuntimeError("non-finite sample — head diverged")
    return x
