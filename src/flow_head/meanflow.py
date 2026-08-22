"""P2 — MeanFlow objective + few-step samplers (pre-registered 2026-08-21).

The head learns the AVERAGE velocity u(x_t, t -> r) over an interval instead
of the instantaneous field, so 1-2-step sampling lands where the many-step
trajectory would — a tighter conditional per frame, targeting the ear-named
residue (frame-boundary sampling scatter). References: MeanFlow (Geng et al.
2025); DSFlow 2602.09041; IntMeanFlow 2510.07979.

Time convention matches cfm.py (OURS, not the paper's): x0 ~ N(0,I) at t=0,
data x1 at t=1, x_t = (1-t)x0 + t*x1, sampling moves t upward. For r >= t,
displacement is x_r = x_t + (r - t) * u(x_t, t, r). The MeanFlow identity in
this convention (derived by differentiating (r-t)*u = INT_t^r v dtau w.r.t.
the CURRENT time t along the trajectory; sanity-checked against a linear
field, see tests):

    u(x_t, t, r) = v(x_t, t) + (r - t) * d/dt u(x_t, t, r)
    d/dt = del_t + v . del_x        (total derivative along the trajectory)

Training regresses u_theta onto stopgrad(v + (r-t) * du/dt) where du/dt
comes from one forward-mode JVP with tangents (v, 1, 0) over (x_t, t, r).
r = t degenerates exactly to the CFM objective (target = v), so p_equal
controls the CFM-vs-interval mix per batch.
"""

import torch

from src.flow_head.model import FlowHead


def meanflow_loss(
    head: FlowHead,
    x1: torch.Tensor,
    condition: torch.Tensor,
    p_equal: float = 0.75,
    generator=None,
    adaptive_c: float | None = 1e-3,
) -> torch.Tensor:
    """x1: clean latents [B, d_latent]; condition: [B, d_model].

    p_equal: fraction of the batch trained with r = t (the instantaneous /
    CFM term — keeps the r=t field sharp for heun-compatible sampling and
    stabilizes training per the MeanFlow paper's r=t mixing).

    adaptive_c: the paper's adaptive per-sample weighting w = 1/(err+c),
    stop-graded — WITHOUT it the JVP-target bootstrap diverges (verified
    2026-08-21: plain MSE exploded 2 -> 461 on the synthetic task; adaptive
    weighting is load-bearing, not a refinement). None = plain MSE (kept
    only for the r=t-degenerates-to-CFM identity test)."""
    if not head.cfg.meanflow:
        raise ValueError("meanflow_loss requires a meanflow=True head config")
    if not torch.isfinite(x1).all() or not torch.isfinite(condition).all():
        raise ValueError("non-finite training batch — refusing to step")
    b = x1.shape[0]
    dev = x1.device
    a1 = torch.rand(b, device=dev, generator=generator)
    a2 = torch.rand(b, device=dev, generator=generator)
    t = torch.minimum(a1, a2)
    r = torch.maximum(a1, a2)
    eq = torch.rand(b, device=dev, generator=generator) < p_equal
    r = torch.where(eq, t, r)

    x0 = torch.randn(x1.shape, device=dev, dtype=x1.dtype, generator=generator)
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    v = x1 - x0

    def fn(x_in, t_in, r_in):
        return head(x_in, t_in, condition, r=r_in)

    u, dudt = torch.func.jvp(fn, (x_t, t, r), (v, torch.ones_like(t), torch.zeros_like(r)))
    target = (v + (r - t)[:, None] * dudt).detach()
    if adaptive_c is None:
        return torch.nn.functional.mse_loss(u, target)
    err = (u - target).pow(2).mean(dim=-1)
    w = 1.0 / (err.detach() + adaptive_c)
    return (w * err).mean()


@torch.no_grad()
def mf_sample(
    head: FlowHead,
    condition: torch.Tensor,
    d_latent: int,
    nfe: int = 2,
    generator=None,
) -> torch.Tensor:
    """condition [B, d_model] -> latents [B, d_latent] in nfe steps on a
    uniform grid: x_{t+dt} = x_t + dt * u(x_t, t, t+dt). nfe=1 is the
    single-jump x1 = x0 + u(x0, 0, 1)."""
    b = condition.shape[0]
    device = condition.device
    x = torch.randn(b, d_latent, device=device, generator=generator)
    grid = torch.linspace(0, 1, nfe + 1, device=device)
    for i in range(nfe):
        t = grid[i].expand(b)
        r = grid[i + 1].expand(b)
        x = x + (grid[i + 1] - grid[i]) * head(x, t, condition, r=r)
    if not torch.isfinite(x).all():
        raise RuntimeError("non-finite sample — head diverged")
    return x


@torch.no_grad()
def mf_cfg_sample(
    head: FlowHead,
    condition: torch.Tensor,
    neg_condition: torch.Tensor,
    d_latent: int,
    cfg_scale: float = 1.3,
    nfe: int = 2,
    generator=None,
) -> torch.Tensor:
    """mf_sample with per-step CFG combination of the average-velocity field
    (mirrors _CFGField / the teacher's own stream arithmetic):
    u = u_neg + s * (u_cond - u_neg)."""
    b = condition.shape[0]
    device = condition.device
    x = torch.randn(b, d_latent, device=device, generator=generator)
    grid = torch.linspace(0, 1, nfe + 1, device=device)
    for i in range(nfe):
        t = grid[i].expand(b)
        r = grid[i + 1].expand(b)
        u_c = head(x, t, condition, r=r)
        u_n = head(x, t, neg_condition, r=r)
        u = u_n + cfg_scale * (u_c - u_n)
        x = x + (grid[i + 1] - grid[i]) * u
    if not torch.isfinite(x).all():
        raise RuntimeError("non-finite sample — head diverged")
    return x


class RTEqualField:
    """Adapter presenting a meanflow head as an instantaneous field
    (r = t) so cfm.py's euler/heun samplers — and _CFGField — run on it
    unchanged. Used for NFE-8 heun comparability evals against CFM heads."""

    def __init__(self, head: FlowHead):
        self.head = head
        self.cfg = head.cfg  # samplers/patches read .cfg.d_latent

    def __call__(self, x_t, t, condition):
        return self.head(x_t, t, condition, r=t)


def interframe_scatter(z: torch.Tensor) -> dict:
    """The NEW instrument (pre-registered 2026-08-21): distribution of
    adjacent-frame latent jumps ||z_t - z_{t-1}|| over a [T, d] sequence —
    the direct metric for the ear-named residue ('tiny voice frames
    mismatched ever so slightly'). Compare student vs teacher on the SAME
    utterance; report the ratio of medians as 'excess scatter'."""
    if z.ndim != 2 or z.shape[0] < 3:
        raise ValueError(f"need [T>=3, d] latents, got {tuple(z.shape)}")
    d = torch.linalg.vector_norm(z[1:].float() - z[:-1].float(), dim=-1)
    q = torch.quantile(d, torch.tensor([0.25, 0.5, 0.75]))
    # second difference isolates the high-frequency (frame-to-frame jitter)
    # component from legitimate slow trajectory movement
    d2 = torch.linalg.vector_norm((z[2:].float() - 2 * z[1:-1].float() + z[:-2].float()), dim=-1)
    return {
        "scatter_median": float(q[1]),
        "scatter_iqr": float(q[2] - q[0]),
        "jerk_median": float(d2.median()),
        "n_frames": int(z.shape[0]),
    }
