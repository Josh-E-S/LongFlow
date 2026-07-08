"""Tier-1 tests for the versioned closed-loop patch (src/flow_head/integration.py)."""

import torch

from src.flow_head.integration import FlowHeadPatch
from src.flow_head.model import FlowHead, FlowHeadConfig

DM, DL = 24, 8


class StubModel:
    def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=3.0):
        return condition[..., :DL]


def make_patch(model):
    head = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=16, layers=1))
    return FlowHeadPatch(model, head, torch.zeros(DL), torch.ones(DL), nfe=2)


def test_patch_replaces_and_restores():
    model = StubModel()
    patch = make_patch(model)
    cond = torch.randn(1, DM, dtype=torch.bfloat16)
    with patch:
        out = model.sample_speech_tokens(cond, None, 1.3)
        assert out.shape == (1, DL) and out.dtype == torch.bfloat16
        assert patch.calls == 1
    assert "sample_speech_tokens" not in vars(model)  # class method restored
    assert model.sample_speech_tokens(cond).shape == (1, DL)  # original works


def test_fresh_noise_per_frame():
    """Two calls with the IDENTICAL condition must differ (fresh x0 each frame) —
    the exact hazard flagged in review-adversarial.md §2c."""
    model = StubModel()
    patch = make_patch(model)
    cond = torch.randn(1, DM)
    with patch:
        z1 = model.sample_speech_tokens(cond)
        z2 = model.sample_speech_tokens(cond)
    # head is AdaLN-zero init -> output = integrated x0 path -> pure noise; must differ
    assert not torch.allclose(z1, z2)
    assert patch.calls == 2


def test_latent_stats_drift_curve():
    model = StubModel()
    patch = make_patch(model)
    with patch:
        for _ in range(8):
            model.sample_speech_tokens(torch.randn(1, DM))
    stats = patch.latent_stats()
    assert stats["frames"] == 8
    assert len(stats["std_quarters"]) == 4
    assert all(s > 0 for s in stats["std_quarters"])
