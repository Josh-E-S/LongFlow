"""Tier-1 tests for the flow-head trainer (full pipeline on synthetic cache files)."""

import pytest
import torch

from src.cache.capture import SampleCapture, save_utterance
from src.flow_head.model import FlowHead, FlowHeadConfig
from src.flow_head.trainer import (
    EMA,
    load_checkpoint,
    load_pairs,
    sample_latents,
    save_checkpoint,
    train,
)

DM, DL = 24, 8


class StubModel:
    def __init__(self, w):
        self.w = w

    def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=3.0):
        return (condition.float() @ self.w).to(condition.dtype)


def write_cache(tmp_path, n_utts=6, frames=20, seed=0):
    """Synthetic cache dir with a learnable linear condition->latent map."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(DM, DL, generator=g) / DM**0.5
    model = StubModel(w)
    for u in range(n_utts):
        with SampleCapture(model) as cap:
            for _f in range(frames):
                cond = torch.randn(1, DM, generator=g).to(torch.bfloat16)
                model.sample_speech_tokens(cond)
            utt = cap.to_utterance(f"u{u}", f"text {u}")
        save_utterance(utt, tmp_path / f"u{u:03d}.pt")
    return w


def test_load_pairs_flattens_and_standardizes(tmp_path):
    write_cache(tmp_path)
    data = load_pairs(tmp_path)
    assert data.hidden.shape == (120, DM) and data.latent.shape == (120, DL)
    assert data.latent.mean(dim=0).abs().max() < 1e-3
    assert (data.latent.std(dim=0) - 1).abs().max() < 1e-2
    assert load_pairs(tmp_path, limit=2).hidden.shape[0] == 40


def test_load_pairs_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cached"):
        load_pairs(tmp_path)


def test_ema_tracks_parameters():
    head = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=16, layers=1))
    ema = EMA(head, decay=0.5)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(1.0)
    ema.update(head)
    name, param = next(iter(head.named_parameters()))
    expected = 0.5 * (param.detach() - 1.0) + 0.5 * param.detach()
    assert torch.allclose(ema.shadow[name], expected, atol=1e-6)


def test_end_to_end_overfit_and_checkpoint_roundtrip(tmp_path):
    """Cache -> load -> train -> loss drops -> checkpoint -> sample tracks targets.

    This is the synthetic twin of the 1-clip overfit gate: the whole pipeline,
    real code paths, seconds on CPU.
    """
    torch.manual_seed(0)
    w = write_cache(tmp_path, n_utts=6, frames=30)
    data = load_pairs(tmp_path)
    head = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=64, layers=2))
    # ema_decay matched to step count: 0.9999 over 300 steps would leave the EMA
    # ~97% at init (the zero function) — same trap applies to short real runs
    out = train(head, data, steps=300, batch_size=64, lr=3e-3, ema_decay=0.98, log_every=10_000)
    losses = out["losses"]
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    assert last < 0.6 * first, f"loss {first:.3f} -> {last:.3f}: pipeline not learning"

    ckpt = tmp_path / "head.pt"
    save_checkpoint(ckpt, head, out["ema"], data, step=300)
    head2, mean, std = load_checkpoint(ckpt)
    assert head2.cfg.d_model == DM and head2.cfg.d_latent == DL

    cond = torch.randn(64, DM)
    target = cond @ w  # true head-space latents
    sample = sample_latents(head2, cond, mean, std, nfe=8, seed=1)
    err = (sample - target).pow(2).mean()
    null = (sample - target[torch.randperm(64)]).pow(2).mean()
    assert err < 0.7 * null, f"samples don't track conditions ({err:.3f} vs {null:.3f})"


def test_checkpoint_ema_differs_from_raw_weights(tmp_path):
    write_cache(tmp_path, n_utts=2, frames=10)
    data = load_pairs(tmp_path)
    head = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=16, layers=1))
    out = train(head, data, steps=20, batch_size=16, lr=1e-2, ema_decay=0.999, log_every=10_000)
    ckpt = tmp_path / "h.pt"
    save_checkpoint(ckpt, head, out["ema"], data, step=20)
    raw, _, _ = load_checkpoint(ckpt, use_ema=False)
    ema, _, _ = load_checkpoint(ckpt, use_ema=True)
    diffs = [
        (p1 - p2).abs().max() for p1, p2 in zip(raw.parameters(), ema.parameters(), strict=True)
    ]
    assert max(float(d) for d in diffs) > 0  # EMA lags fast-moving weights
