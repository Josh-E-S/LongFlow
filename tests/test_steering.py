"""Tier-1 tests for P0 Stage 1 activation capture (src/steering/contrast_pairs.py)."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.steering.contrast_pairs import (
    LayerActivationRecorder,
    load_records,
    pool_record,
    save_records,
    select_frames,
)

D = 16


class TupleLayer(nn.Module):
    """Mimics Qwen2DecoderLayer: returns (hidden_states, ...) tuple."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        return (self.proj(x), None)


def make_stack(n_layers=3):
    return nn.ModuleList([TupleLayer() for _ in range(n_layers)])


def run_forward(layers, x):
    for layer in layers:
        x = layer(x)[0]
    return x


def simulate_generation(layers, recorder_kwargs, prompt_len=7, steps=5, passes_per_step=1):
    """Prefill with [1, prompt_len, D], then `steps` AR steps of [1, 1, D],
    with `passes_per_step` forward passes per step (CFG simulation)."""
    rec = LayerActivationRecorder(layers, **recorder_kwargs)
    with rec:
        run_forward(layers, torch.randn(1, prompt_len, D))
        for _ in range(steps):
            for _ in range(passes_per_step):
                run_forward(layers, torch.randn(1, 1, D))
    return rec


def test_step_states_shape_and_prefill_skip():
    layers = make_stack()
    rec = simulate_generation(layers, {}, prompt_len=7, steps=5)
    states = rec.step_states()
    assert states.shape == (3, 5, D)
    assert rec.num_calls == 6  # prefill + 5 AR steps


def test_cfg_double_pass_keeps_selected_call():
    layers = make_stack(n_layers=1)
    torch.manual_seed(0)
    rec = LayerActivationRecorder(layers, calls_per_step=2, keep_call=0)
    inputs = [torch.randn(1, 1, D) for _ in range(6)]
    with rec:
        run_forward(layers, torch.randn(1, 4, D))  # prefill
        for x in inputs:
            run_forward(layers, x)
    states = rec.step_states()
    assert states.shape == (1, 3, D)
    # kept calls must be the even-indexed (positive-pass) AR calls
    with torch.no_grad():
        expected = layers[0](inputs[0])[0][0].float()
    assert torch.allclose(states[0, 0], expected[0], atol=1e-6)


def test_miscalibrated_calls_per_step_raises():
    layers = make_stack()
    rec = simulate_generation(layers, {"calls_per_step": 2}, steps=5)  # 5 not divisible by 2
    with pytest.raises(RuntimeError, match="recalibrate"):
        rec.step_states()


def test_hooks_removed_after_context_exit():
    layers = make_stack()
    rec = simulate_generation(layers, {}, steps=2)
    n = rec.num_calls
    run_forward(layers, torch.randn(1, 1, D))  # outside context
    assert rec.num_calls == n


def test_select_frames_mask_and_fallback():
    states = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1).expand(2, 12, D)
    mask = torch.zeros(12, dtype=torch.bool)
    mask[8:] = True
    assert select_frames(states, speech_frame_mask=mask).shape == (2, 4, D)
    assert select_frames(states, drop_first_fraction=0.25).shape == (2, 9, D)
    with pytest.raises(ValueError, match="0 frames"):
        select_frames(states, speech_frame_mask=torch.zeros(12, dtype=torch.bool))


def test_pool_record_mean_is_correct():
    states = torch.stack([torch.zeros(4, D), torch.ones(4, D)]).reshape(2, 4, D)
    rec = pool_record(
        states, script_id="s", axis="arousal", pole="pos", sample_idx=0, num_calls_total=5
    )
    assert rec.layer_vectors.shape == (2, D)
    assert torch.allclose(rec.layer_vectors[0], torch.zeros(D))
    assert torch.allclose(rec.layer_vectors[1], torch.ones(D))
    assert rec.num_frames_kept == 4
    assert not torch.isnan(rec.layer_vectors).any()


def test_records_roundtrip(tmp_path):
    states = torch.randn(3, 6, D)
    recs = [
        pool_record(
            states, script_id="s1", axis="valence", pole="neg", sample_idx=1, num_calls_total=7
        )
    ]
    p = tmp_path / "vectors.pt"
    save_records(recs, p)
    loaded = load_records(p)
    assert loaded[0].script_id == "s1" and loaded[0].pole == "neg"
    assert torch.allclose(loaded[0].layer_vectors, recs[0].layer_vectors)


def test_p0_contrast_config_is_wellformed():
    cfg = json.loads(Path("configs/p0_contrast.json").read_text())
    assert set(cfg["axes"]) == {"arousal", "valence"}
    for axis in cfg["axes"].values():
        assert set(axis) == {"pos", "neg"}
    assert len(cfg["scripts"]) == 10
    assert len(cfg["held_out_scripts"]) == 3
    all_texts = list(cfg["scripts"].values()) + list(cfg["held_out_scripts"].values())
    assert len(set(all_texts)) == len(all_texts)  # no duplicate scripts
    for text in all_texts:
        assert 2 <= text.count(".") <= 5  # 2-4 sentences per spec
