"""Tier-1 tests for the P1 flow head (model + CFM objective + sampler).

Includes the synthetic learning test (CLAUDE.md: shape-and-gradient tests before
any training script) — a tiny head must drive CFM loss down on a learnable
synthetic condition->latent mapping in seconds on CPU. The real 1-clip overfit
gate on cached VibeVoice pairs is tier-2 (@slow) and lives with the trainer.
"""

import pytest
import torch

from src.flow_head.cfm import cfm_loss, euler_sample, sway_grid
from src.flow_head.model import FlowHead, FlowHeadConfig

DM, DL = 48, 8


def tiny_head(width=32, layers=2):
    return FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=width, layers=layers))


def test_forward_shapes_and_zero_init():
    head = tiny_head()
    x = torch.randn(5, DL)
    t = torch.rand(5)
    c = torch.randn(5, DM)
    out = head(x, t, c)
    assert out.shape == (5, DL)
    assert torch.allclose(out, torch.zeros_like(out))  # AdaLN-zero + zero out_proj


def test_dim_mismatch_raises():
    head = tiny_head()
    with pytest.raises(ValueError, match="dim mismatch"):
        head(torch.randn(2, DL + 1), torch.rand(2), torch.randn(2, DM))


def test_default_config_param_count_near_15m():
    head = FlowHead(FlowHeadConfig(d_model=1536, d_latent=64))
    p = head.param_count()
    assert 12e6 < p < 20e6, f"{p/1e6:.1f}M params — resize width/layers"


def test_all_params_receive_gradients():
    head = tiny_head()
    loss = cfm_loss(head, torch.randn(8, DL), torch.randn(8, DM))
    loss.backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"


def test_cfm_loss_rejects_nonfinite_batch():
    head = tiny_head()
    bad = torch.randn(4, DL)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        cfm_loss(head, bad, torch.randn(4, DM))


def test_sway_grid_properties():
    for coef in (0.0, -1.0):
        g = sway_grid(4, coef)
        assert g.shape == (5,)
        assert abs(float(g[0])) < 1e-6 and abs(float(g[-1]) - 1) < 1e-6
        assert (g.diff() > 0).all()  # strictly increasing
    # negative coef front-loads: interior points pulled toward 0
    assert (sway_grid(4, -1.0)[1:-1] < sway_grid(4, 0.0)[1:-1]).all()


def test_euler_sample_shape_and_determinism():
    head = tiny_head()
    c = torch.randn(3, DM)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    s1 = euler_sample(head, c, DL, nfe=4, generator=g1)
    s2 = euler_sample(head, c, DL, nfe=4, generator=g2)
    assert s1.shape == (3, DL)
    assert torch.allclose(s1, s2)


def test_synthetic_learning_loss_drops_and_samples_track_targets():
    """The head must learn a deterministic linear condition->latent map."""
    torch.manual_seed(0)
    head = tiny_head(width=64, layers=2)
    w_true = torch.randn(DM, DL) / DM**0.5
    cond = torch.randn(256, DM)
    x1 = cond @ w_true  # deterministic target per condition
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    first = None
    for _step in range(300):
        idx = torch.randint(0, 256, (64,), generator=g)
        loss = cfm_loss(head, x1[idx], cond[idx], generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < 0.5 * first, f"loss {first:.3f} -> {float(loss):.3f}: not learning"
    # sampled latents should correlate with targets far better than chance
    test_c = torch.randn(64, DM)
    target = test_c @ w_true
    sample = euler_sample(head, test_c, DL, nfe=8, generator=torch.Generator().manual_seed(2))
    err = (sample - target).pow(2).mean()
    null = (sample - target[torch.randperm(64)]).pow(2).mean()
    assert err < 0.7 * null, f"samples don't track conditions (err {err:.3f} vs null {null:.3f})"


def test_heun_sample_shape_determinism_and_tracks_targets():
    from src.flow_head.cfm import heun_sample

    torch.manual_seed(0)
    head = tiny_head(width=64, layers=2)
    w_true = torch.randn(DM, DL) / DM**0.5
    cond = torch.randn(256, DM)
    x1 = cond @ w_true
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    for _ in range(300):
        idx = torch.randint(0, 256, (64,), generator=g)
        loss = cfm_loss(head, x1[idx], cond[idx], generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    c = torch.randn(16, DM)
    s1 = heun_sample(head, c, DL, nfe=4, generator=torch.Generator().manual_seed(0))
    s2 = heun_sample(head, c, DL, nfe=4, generator=torch.Generator().manual_seed(0))
    assert s1.shape == (16, DL) and torch.allclose(s1, s2)
    target = c @ w_true
    err = (s1 - target).pow(2).mean()
    null = (s1 - target[torch.randperm(16)]).pow(2).mean()
    assert err < null


def test_forward_rejects_3d_condition():
    """A [B, 1, d] condition must raise, not broadcast (pre-75K audit finding 3)."""
    import pytest

    head = FlowHead(FlowHeadConfig(d_model=24, d_latent=8, width=16, layers=1))
    x_t = torch.randn(4, 8)
    t = torch.rand(4)
    with pytest.raises(ValueError, match="2-D"):
        head(x_t, t, torch.randn(4, 1, 24))
    with pytest.raises(ValueError, match="2-D"):
        head(torch.randn(4, 1, 8), t, torch.randn(4, 24))
