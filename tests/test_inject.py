"""Tier-1 tests for P0 Stage 3 injection (src/steering/inject.py).

Reuses the synthetic VibeVoice call pattern from test_steering: prefill,
positive AR calls continuing prompt positions, negative CFG calls at small
positions. Steering must hit ONLY the positive stream.
"""

import pytest
import torch
from torch import nn

from src.steering.inject import SegmentGate, SteeringInjector

D = 16
PROMPT = 50


class IdentityLayer(nn.Module):
    """Identity so output deltas are exactly the injected steering."""

    def forward(self, x, cache_position=None):
        return (x.clone(), None)


def call(layers, x, pos_last, seq_len=1):
    cp = torch.arange(pos_last - seq_len + 1, pos_last + 1)
    outs = []
    for layer in layers:
        x = layer(x, cache_position=cp)[0]
        outs.append(x)
    return outs


def test_only_positive_stream_steered():
    layers = nn.ModuleList([IdentityLayer(), IdentityLayer()])
    direction = torch.nn.functional.normalize(torch.ones(D), dim=0)
    inj = SteeringInjector(layers, {1: direction}, alpha=2.0, prompt_len=PROMPT)
    x = torch.zeros(1, 1, D)
    with inj:
        pre = call(layers, torch.zeros(1, PROMPT, D), PROMPT - 1, seq_len=PROMPT)
        assert pre[1].abs().sum() == 0  # prefill untouched
        pos1 = call(layers, x, PROMPT)  # positive: steered at layer 1 only
        assert pos1[0].abs().sum() == 0
        assert torch.allclose(pos1[1][0, 0], 2.0 * direction)
        neg = call(layers, x, 3)  # negative stream: untouched
        assert neg[1].abs().sum() == 0
        pos2 = call(layers, x, PROMPT + 1)  # chain continues
        assert torch.allclose(pos2[1][0, 0], 2.0 * direction)
        stale = call(layers, x, PROMPT)  # replayed old position: not expected -> untouched
        assert stale[1].abs().sum() == 0
    assert inj.steered_calls == 2


def test_multi_layer_injection_counts_once_per_step():
    layers = nn.ModuleList([IdentityLayer(), IdentityLayer(), IdentityLayer()])
    d = torch.nn.functional.normalize(torch.randn(D), dim=0)
    inj = SteeringInjector(layers, {0: d, 2: d}, alpha=1.0, prompt_len=PROMPT)
    with inj:
        outs = call(layers, torch.zeros(1, 1, D), PROMPT)
    # layer 0 steered; layer 1 passes it through (identity); layer 2 adds again
    assert torch.allclose(outs[2][0, 0], 2.0 * d, atol=1e-6)
    assert inj.steered_calls == 1


def test_gate_blocks_and_allows():
    layers = nn.ModuleList([IdentityLayer()])
    d = torch.nn.functional.normalize(torch.randn(D), dim=0)
    gate = SegmentGate(target_segments={1})
    inj = SteeringInjector(layers, {0: d}, alpha=3.0, prompt_len=PROMPT, gate=gate)
    with inj:
        out0 = call(layers, torch.zeros(1, 1, D), PROMPT)[0]  # segment 0: gated off
        assert out0.abs().sum() == 0
        gate.advance()  # now segment 1: gated on
        out1 = call(layers, torch.zeros(1, 1, D), PROMPT + 1)[0]
        assert torch.allclose(out1[0, 0], 3.0 * d)
    assert inj.steered_calls == 1  # only the gated-on call counts


def test_hooks_removed_and_bad_layer_index_raises():
    layers = nn.ModuleList([IdentityLayer()])
    d = torch.randn(D)
    with pytest.raises(ValueError, match="out of range"):
        SteeringInjector(layers, {5: d}, alpha=1.0, prompt_len=PROMPT)
    inj = SteeringInjector(layers, {0: d}, alpha=1.0, prompt_len=PROMPT)
    with inj:
        pass
    out = call(layers, torch.zeros(1, 1, D), PROMPT)[0]
    assert out.abs().sum() == 0  # no steering after exit


def test_dtype_cast_bf16():
    layers = nn.ModuleList([IdentityLayer()])
    d = torch.nn.functional.normalize(torch.randn(D), dim=0)  # fp32 direction
    inj = SteeringInjector(layers, {0: d}, alpha=4.0, prompt_len=PROMPT)
    with inj:
        out = call(layers, torch.zeros(1, 1, D, dtype=torch.bfloat16), PROMPT)[0]
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out[0, 0].float(), 4.0 * d, atol=0.05)  # bf16 tolerance
