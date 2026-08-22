"""Tier-1 tests for P2 MeanFlow (loss + samplers + the scatter instrument).

Includes the JVP-tangent-vs-finite-differences check the project's testing
plan (CLAUDE.md tier list) anticipated from day one, the r=t-degenerates-
to-CFM identity, and a synthetic learning test at NFE 1-2.
"""

import numpy as np
import pytest
import torch

from src.flow_head.cfm import heun_sample
from src.flow_head.meanflow import (
    RTEqualField,
    interframe_scatter,
    meanflow_loss,
    mf_cfg_sample,
    mf_sample,
)
from src.flow_head.model import FlowHead, FlowHeadConfig

DM, DL = 24, 8


def mf_head(width=32, layers=2):
    return FlowHead(
        FlowHeadConfig(d_model=DM, d_latent=DL, width=width, layers=layers, meanflow=True)
    )


def test_r_input_refused_both_directions():
    x, t, c = torch.randn(4, DL), torch.rand(4), torch.randn(4, DM)
    with pytest.raises(ValueError, match="r missing"):
        mf_head()(x, t, c)
    v1 = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=32, layers=2))
    with pytest.raises(ValueError, match="r given"):
        v1(x, t, c, r=t)


def test_jvp_matches_finite_differences():
    """The total derivative d/dt u (moving x along v, r fixed) from
    torch.func.jvp must match a central finite difference."""
    # float64 + tiny eps: the time embedding's 1000x frequency scaling makes
    # float32 finite differences meaningless (first attempt failed on exactly
    # that; the JVP itself was correct all along)
    head = mf_head().double()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    head.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, DL, generator=g).double()
    t = (torch.rand(6, generator=g) * 0.6 + 0.2).double()
    r = torch.clamp(t + torch.rand(6, generator=g).double() * 0.3, max=1.0)
    c = torch.randn(6, DM, generator=g).double()
    v = torch.randn(6, DL, generator=g).double()

    def fn(x_in, t_in, r_in):
        return head(x_in, t_in, c, r=r_in)

    _u, dudt = torch.func.jvp(fn, (x, t, r), (v, torch.ones_like(t), torch.zeros_like(r)))
    eps = 1e-6
    with torch.no_grad():
        u_plus = fn(x + eps * v, t + eps, r)
        u_minus = fn(x - eps * v, t - eps, r)
    fd = (u_plus - u_minus) / (2 * eps)
    assert torch.allclose(dudt, fd, atol=1e-4), f"max diff {(dudt - fd).abs().max():.2e}"


def test_r_equals_t_degenerates_to_cfm():
    """With p_equal=1.0 the MeanFlow target is exactly v — the loss must
    equal cfm_loss on the same head/data/noise draw."""
    head = mf_head()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x1 = torch.randn(16, DL)
    c = torch.randn(16, DM)
    g1 = torch.Generator().manual_seed(7)
    mf = meanflow_loss(head, x1, c, p_equal=1.0, generator=g1, adaptive_c=None)
    # replicate the mf loss's internal draws to feed cfm the same t/x0:
    # meanflow_loss draws a1, a2, eq-mask, then x0; with p_equal=1 t=min(a1,a2)
    g2 = torch.Generator().manual_seed(7)
    a1 = torch.rand(16, generator=g2)
    a2 = torch.rand(16, generator=g2)
    t = torch.minimum(a1, a2)
    _eq = torch.rand(16, generator=g2)
    x0 = torch.randn(16, DL, generator=g2)
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    v = x1 - x0
    pred = head(x_t, t, c, r=t)
    ref = torch.nn.functional.mse_loss(pred, v)
    assert torch.allclose(mf, ref, atol=1e-5), f"{float(mf):.6f} vs {float(ref):.6f}"


def test_loss_grads_reach_all_params():
    head = mf_head()
    loss = meanflow_loss(head, torch.randn(8, DL), torch.randn(8, DM), p_equal=0.5)
    loss.backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"
    assert any("r_mlp" in n for n, _ in head.named_parameters())


def test_samplers_shapes_and_finite():
    head = mf_head()
    c = torch.randn(5, DM)
    for nfe in (1, 2, 4):
        z = mf_sample(head, c, DL, nfe=nfe, generator=torch.Generator().manual_seed(0))
        assert z.shape == (5, DL) and torch.isfinite(z).all()
    z = mf_cfg_sample(
        head,
        c,
        torch.randn(5, DM),
        DL,
        cfg_scale=1.3,
        nfe=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert z.shape == (5, DL) and torch.isfinite(z).all()
    # r=t adapter drives the existing heun sampler unchanged
    z = heun_sample(RTEqualField(head), c, DL, nfe=4, generator=torch.Generator().manual_seed(0))
    assert z.shape == (5, DL) and torch.isfinite(z).all()


def test_synthetic_learning_one_and_two_step():
    """A tiny mf head must learn a linear condition->latent map well enough
    that 1- and 2-step samples track the targets (the CLAUDE.md synthetic
    learning gate, MeanFlow edition)."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(DM, DL, generator=g) / DM**0.5
    head = mf_head(width=64, layers=2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for step in range(1500):
        c = torch.randn(128, DM, generator=g)
        x1 = c @ w
        loss = meanflow_loss(
            head, x1, c, p_equal=0.5, generator=torch.Generator().manual_seed(step)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
    head.eval()
    c = torch.randn(64, DM, generator=g)
    target = c @ w
    bars = {1: 0.6, 2: 0.3}
    for nfe in (1, 2):
        z = mf_sample(head, c, DL, nfe=nfe, generator=torch.Generator().manual_seed(1))
        err = (z - target).pow(2).mean().sqrt() / target.pow(2).mean().sqrt()
        assert err < bars[nfe], f"nfe={nfe}: relative RMSE {err:.2f} — did not learn"


def test_interframe_scatter_orders_smooth_vs_jittery():
    g = torch.Generator().manual_seed(0)
    t = torch.linspace(0, 4 * np.pi, 200)
    smooth = torch.stack([torch.sin(t + k) for k in range(DL)], dim=-1)
    jitter = smooth + 0.3 * torch.randn(smooth.shape, generator=g)
    s1 = interframe_scatter(smooth)
    s2 = interframe_scatter(jitter)
    assert s2["scatter_median"] > s1["scatter_median"]
    assert s2["jerk_median"] > 3 * s1["jerk_median"]  # jerk isolates the jitter
    with pytest.raises(ValueError, match="T>=3"):
        interframe_scatter(torch.randn(2, DL))
