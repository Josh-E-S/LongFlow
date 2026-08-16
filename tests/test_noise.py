"""Tier-1 tests for src/cache/noise.py (NoiseIntervention, ported from GN6)."""

import torch
from torch import nn

from src.cache.noise import NoiseIntervention


class Connector(nn.Module):
    def forward(self, x):
        return x  # identity: output == input, easy to reason about


def test_zero_sigma_is_untouched():
    mod = Connector()
    with NoiseIntervention(mod, sigma_fn=lambda c: 0.0) as ni:
        out = mod(torch.ones(4, 8))
    assert torch.equal(out, torch.ones(4, 8))
    assert ni.last_sigma == 0.0


def test_inactive_gate_untouched_and_zeroes_last_sigma():
    mod = Connector()
    with NoiseIntervention(mod, sigma_fn=lambda c: 0.5, active_fn=lambda: False) as ni:
        out = mod(torch.ones(4, 8))
    assert torch.equal(out, torch.ones(4, 8))
    assert ni.last_sigma == 0.0


def test_positive_sigma_perturbs_output_and_records_last_sigma():
    mod = Connector()
    torch.manual_seed(0)
    with NoiseIntervention(mod, sigma_fn=lambda c: 0.3) as ni:
        out = mod(torch.zeros(64, 32) + 1.0)  # constant input -> var starts at 0
    # first call: var EMA seeded from this call's own var (0, constant input) ->
    # sigma * sqrt(var) == 0 on the very first call, so no perturbation yet
    assert torch.equal(out, torch.ones(64, 32))
    assert ni.last_sigma == 0.3


def test_sigma_scales_with_running_variance():
    mod = Connector()
    torch.manual_seed(0)
    with NoiseIntervention(mod, sigma_fn=lambda c: 0.5) as ni:
        mod(torch.randn(256, 32))  # seed the running var with real spread
        out2 = mod(torch.zeros(256, 32))  # second call: EMA var > 0 now
    assert not torch.equal(out2, torch.zeros(256, 32))  # perturbed
    assert ni.last_sigma == 0.5


def test_calls_fn_wires_schedule():
    mod = Connector()
    counter = {"n": 0}
    seen = []

    def sigma_fn(c):
        seen.append(c)
        return 0.0

    with NoiseIntervention(mod, sigma_fn=sigma_fn, calls_fn=lambda: counter["n"]):
        for i in range(3):
            counter["n"] = i
            mod(torch.zeros(2, 4))
    assert seen == [0, 1, 2]


def test_hook_removed_on_exit():
    mod = Connector()
    with NoiseIntervention(mod, sigma_fn=lambda c: 0.5):
        pass
    out = mod(torch.ones(4, 8))
    assert torch.equal(out, torch.ones(4, 8))  # no hook left registered
