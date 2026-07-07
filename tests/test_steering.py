"""Tier-1 tests for P0 Stage 1 activation capture (src/steering/contrast_pairs.py).

Simulates VibeVoice's measured call pattern (Colab, 2026-07-06): prefill, then a
positive AR call per token, with an interleaved negative CFG call only on
speech-frame steps — non-uniform, so classification is by cache_position.
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.steering.contrast_pairs import (
    LayerActivationRecorder,
    drop_leading_true,
    load_records,
    pool_record,
    save_records,
    select_frames,
    target_frame_mask,
)

D = 16
FRAME, START, END = 100, 101, 102


class TupleLayer(nn.Module):
    """Mimics Qwen2DecoderLayer: returns a tuple, receives cache_position kwarg."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, D)

    def forward(self, x, cache_position=None):
        return (self.proj(x), None)


def make_stack(n_layers=3):
    return nn.ModuleList([TupleLayer() for _ in range(n_layers)])


def run_stack(layers, x, pos_last, seq_len=1):
    cp = torch.arange(pos_last - seq_len + 1, pos_last + 1)
    for layer in layers:
        x = layer(x, cache_position=cp)[0]
    return x


def simulate_vibevoice(layers, gen_ids, prompt_len=50, neg_seed_len=2):
    """Prefill, then per generated token g>=1 a positive call at position
    prompt_len+g-1, plus a negative call (small positions) after frame steps."""
    rec = LayerActivationRecorder(layers)
    neg_pos = neg_seed_len
    with rec:
        run_stack(layers, torch.randn(1, prompt_len, D), prompt_len - 1, seq_len=prompt_len)
        run_stack(layers, torch.randn(1, neg_seed_len, D), neg_seed_len - 1, seq_len=neg_seed_len)
        for g in range(1, len(gen_ids)):
            run_stack(layers, torch.randn(1, 1, D), prompt_len + g - 1)
            if gen_ids[g] == FRAME:  # negative pass fires only on frame steps
                run_stack(layers, torch.randn(1, 1, D), neg_pos)
                neg_pos += 1
    return rec


GEN = [FRAME, FRAME, FRAME, END, START, FRAME, FRAME, END, FRAME]  # 2 segments + stray


def test_positive_stream_recovered():
    layers = make_stack()
    rec = simulate_vibevoice(layers, GEN)
    states, tok_idx = rec.step_states(prompt_len=50)
    assert states.shape == (3, len(GEN) - 1, D)  # every token except token 0
    assert tok_idx.tolist() == list(range(1, len(GEN)))
    # expected total calls: 2 prefills + (n-1) positive + #frames-after-token-0 negative
    n_neg = sum(1 for t in GEN[1:] if t == FRAME)
    assert rec.num_calls == 2 + (len(GEN) - 1) + n_neg


def test_negative_positions_never_misclassified_even_when_large():
    # long generation where negative positions exceed prompt_len in absolute value
    layers = make_stack(1)
    gen = [FRAME] * 40
    rec = simulate_vibevoice(layers, gen, prompt_len=10)
    states, tok_idx = rec.step_states(prompt_len=10)
    assert states.shape[1] == 39
    assert tok_idx.tolist() == list(range(1, 40))


def test_missing_cache_position_raises():
    layers = make_stack(1)
    rec = LayerActivationRecorder(layers)
    with rec, pytest.raises(RuntimeError, match="cache_position"):
        layers[0](torch.randn(1, 1, D))  # no kwarg


def test_wrong_prompt_len_raises():
    layers = make_stack(1)
    rec = simulate_vibevoice(layers, GEN)
    with pytest.raises(RuntimeError, match="wrong prompt_len"):
        rec.step_states(prompt_len=999)


def test_hooks_removed_after_context_exit():
    layers = make_stack(1)
    rec = simulate_vibevoice(layers, GEN)
    n = rec.num_calls
    run_stack(layers, torch.randn(1, 1, D), 5)
    assert rec.num_calls == n


def test_target_frame_mask_excludes_lead_in_segment():
    tok_idx = torch.arange(1, len(GEN))
    mask = target_frame_mask(GEN, tok_idx, frame_id=FRAME, end_id=END)
    # frames at gen indices 5, 6, 8 are after the first END (index 3); 1, 2 are lead-in
    kept_gen_indices = tok_idx[mask].tolist()
    assert kept_gen_indices == [5, 6, 8]
    # without end_id all frames are kept
    mask_all = target_frame_mask(GEN, tok_idx, frame_id=FRAME, end_id=None)
    assert tok_idx[mask_all].tolist() == [1, 2, 5, 6, 8]


def test_drop_leading_true_fallback():
    mask = torch.tensor([True] * 8 + [False, True, True])
    out = drop_leading_true(mask, 0.3)  # 10 True -> drop first 3
    assert out.sum() == 7
    assert not out[:3].any()


def test_pool_record_mean_and_nan_guard():
    states = torch.stack([torch.zeros(4, D), torch.ones(4, D)])
    mask = torch.tensor([True, True, False, True])
    rec = pool_record(
        states, mask, script_id="s", axis="arousal", pole="pos", sample_idx=0, num_calls_total=9
    )
    assert rec.layer_vectors.shape == (2, D)
    assert torch.allclose(rec.layer_vectors[1], torch.ones(D))
    assert rec.num_frames_kept == 3
    bad = states.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        pool_record(
            bad, mask, script_id="s", axis="arousal", pole="pos", sample_idx=0, num_calls_total=9
        )


def test_select_frames_zero_kept_raises():
    with pytest.raises(ValueError, match="0 frames"):
        select_frames(torch.randn(2, 5, D), torch.zeros(5, dtype=torch.bool))


def test_records_roundtrip(tmp_path):
    states = torch.randn(3, 6, D)
    mask = torch.ones(6, dtype=torch.bool)
    recs = [
        pool_record(
            states,
            mask,
            script_id="s1",
            axis="valence",
            pole="neg",
            sample_idx=1,
            num_calls_total=7,
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
    assert len(set(all_texts)) == len(all_texts)
    for text in all_texts:
        assert 2 <= text.count(".") <= 5
