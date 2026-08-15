"""Tier-1 tests for CFGFlowHeadPatch (the 2026-08-15 audit fix: the flow head
had been silently discarding neg_condition/cfg_scale since July)."""

import torch

from src.flow_head.integration import CFGFlowHeadPatch, FlowHeadPatch, _CFGField
from src.flow_head.model import FlowHead, FlowHeadConfig

DM, DL = 24, 8


class StubModel:
    def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=3.0):
        return condition[..., :DL]


def make_head():
    return FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=16, layers=1))


def test_cfg_field_combination_math():
    """v = v_neg + s*(v_cond - v_neg): at s=1 the neg branch must cancel; at
    s=0 only the neg branch must survive."""
    head = make_head()
    # break the zero-init so the two branches actually differ
    torch.nn.init.normal_(head.out_proj.weight, std=0.1)
    x_t = torch.randn(1, DL)
    t = torch.rand(1)
    cond, neg = torch.randn(1, DM), torch.randn(1, DM)
    v_cond = head(x_t, t, cond)
    v_neg = head(x_t, t, neg)
    assert torch.allclose(_CFGField(head, neg, 1.0)(x_t, t, cond), v_cond, atol=1e-6)
    assert torch.allclose(_CFGField(head, neg, 0.0)(x_t, t, cond), v_neg, atol=1e-6)
    s = 1.3
    assert torch.allclose(
        _CFGField(head, neg, s)(x_t, t, cond), v_neg + s * (v_cond - v_neg), atol=1e-6
    )


def test_cfg_patch_uses_neg_when_given_and_falls_back_when_not():
    model = StubModel()
    head = make_head()
    patch = CFGFlowHeadPatch(model, head, torch.zeros(DL), torch.ones(DL), nfe=2)
    cond = torch.randn(1, DM, dtype=torch.bfloat16)
    neg = torch.randn(1, DM, dtype=torch.bfloat16)
    with patch:
        out = model.sample_speech_tokens(cond, neg, 1.3)
        assert out.shape == (1, DL) and out.dtype == torch.bfloat16
        out2 = model.sample_speech_tokens(cond)  # no neg -> unguided fallback
        assert out2.shape == (1, DL)
        assert patch.calls == 2
    assert "sample_speech_tokens" not in vars(model)  # class method restored


def test_cfg_scale_override_pins_scale():
    """cfg_scale=1.0 override must bypass guidance even when the caller passes
    a scale — the sampled distribution then matches plain FlowHeadPatch's."""
    model = StubModel()
    head = make_head()
    cond = torch.randn(1, DM)
    neg = torch.randn(1, DM)

    torch.manual_seed(7)
    with CFGFlowHeadPatch(model, head, torch.zeros(DL), torch.ones(DL), nfe=2, cfg_scale=1.0):
        z_pinned = model.sample_speech_tokens(cond, neg, 1.3)
    torch.manual_seed(7)
    with FlowHeadPatch(model, head, torch.zeros(DL), torch.ones(DL), nfe=2):
        z_plain = model.sample_speech_tokens(cond, neg, 1.3)
    assert torch.allclose(z_pinned, z_plain, atol=1e-6)
