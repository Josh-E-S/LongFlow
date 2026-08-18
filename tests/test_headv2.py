"""Tier-1 tests for head-v2 (dual-stream + sigma-bucket conditioning).

Built 2026-08-17 after the review found the first v2 20K run trained cond-only
on a dual-stream cache. Design intent under test: missing OR extra conditioning
is refused loudly in every layer (model, loss, trainer, patch) — the silent
one-stream discard is the exact bug class of the 2026-08-15 CFG audit.
"""

import json

import pytest
import torch

from src.cache.capture import UtteranceCache, save_utterance
from src.flow_head.cfm import cfm_loss, heun_sample
from src.flow_head.integration import DualStreamFlowHeadPatch
from src.flow_head.model import (
    BoundField,
    FlowHead,
    FlowHeadConfig,
    sigma_to_bucket,
)
from src.flow_head.trainer import (
    filter_flagged,
    load_checkpoint,
    pairs_from_files,
    sample_latents,
    save_checkpoint,
    train,
)

DM, DL = 24, 8


def v2_head(width=32, layers=2, sigma_buckets=4):
    return FlowHead(
        FlowHeadConfig(
            d_model=DM,
            d_latent=DL,
            width=width,
            layers=layers,
            dual_stream=True,
            sigma_buckets=sigma_buckets,
        )
    )


def v2_inputs(b=5):
    return (
        torch.randn(b, DL),
        torch.rand(b),
        torch.randn(b, DM),
        torch.randn(b, DM),
        torch.randint(0, 4, (b,)),
    )


def test_dual_forward_shapes_and_zero_init():
    head = v2_head()
    x, t, c, n, s = v2_inputs()
    out = head(x, t, c, neg_condition=n, sigma_bucket=s)
    assert out.shape == (5, DL)
    assert torch.allclose(out, torch.zeros_like(out))  # AdaLN-zero + zero out_proj


def test_conditioning_mismatch_refused_both_directions():
    x, t, c, n, s = v2_inputs()
    with pytest.raises(ValueError, match="neg_condition missing"):
        v2_head()(x, t, c, sigma_bucket=s)
    with pytest.raises(ValueError, match="sigma_bucket missing"):
        v2_head()(x, t, c, neg_condition=n)
    v1 = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=32, layers=2))
    with pytest.raises(ValueError, match="neg_condition given"):
        v1(x, t, c, neg_condition=n)
    with pytest.raises(ValueError, match="sigma_bucket given"):
        v1(x, t, c, sigma_bucket=s)


def test_neg_stream_and_sigma_change_output():
    head = v2_head()
    # move off the zero function so conditioning has visible effect
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, t, c, n, s = v2_inputs()
    base = head(x, t, c, neg_condition=n, sigma_bucket=s)
    assert not torch.allclose(base, head(x, t, c, neg_condition=n + 1.0, sigma_bucket=s))
    assert not torch.allclose(base, head(x, t, c, neg_condition=n, sigma_bucket=(s + 1) % 4))


def test_sigma_to_bucket_edges():
    sig = torch.tensor([0.0, 0.04, 0.05, 0.19, 0.20, 0.34, 0.35, 0.40, 1.0])
    assert sigma_to_bucket(sig).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]


def test_bound_field_matches_direct_call():
    head = v2_head()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, t, c, n, _ = v2_inputs()
    bound = BoundField(head, n, sigma_bucket=2)
    direct = head(x, t, c, neg_condition=n, sigma_bucket=torch.full_like(t, 2, dtype=torch.long))
    assert torch.allclose(bound(x, t, c), direct)


def test_cfm_loss_dual_gradients_reach_new_params():
    head = v2_head()
    x1 = torch.randn(8, DL)
    loss = cfm_loss(
        head,
        x1,
        torch.randn(8, DM),
        neg_condition=torch.randn(8, DM),
        sigma_bucket=torch.randint(0, 4, (8,)),
    )
    loss.backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"
    assert any("neg_proj" in n for n, _ in head.named_parameters())
    assert any("sigma_emb" in n for n, _ in head.named_parameters())


def write_v2_cache(tmp_path, n_utts=4, frames=16, seed=0, drop_v2_fields=False):
    g = torch.Generator().manual_seed(seed)
    files = []
    for u in range(n_utts):
        T = frames
        utt = UtteranceCache(
            utt_id=f"cv2_300w_{u:08x}",
            text="synthetic",
            hidden=(torch.randn(T, DM, generator=g)).half(),
            latent=(torch.randn(T, DL, generator=g) * 5 + 3).half(),
            meta={"target_words": 300},
            neg_hidden=None if drop_v2_fields else torch.randn(T, DM, generator=g).half(),
            sigma=None if drop_v2_fields else (torch.rand(T, generator=g) * 0.4).half(),
        )
        p = tmp_path / f"{utt.utt_id}.pt"
        save_utterance(utt, p)
        files.append(p)
    return files


def test_pairs_from_files_dual_reads_v2_fields(tmp_path):
    files = write_v2_cache(tmp_path)
    data = pairs_from_files(files, dual_stream=True)
    n = 4 * 16
    assert data.hidden.shape == (n, DM)
    assert data.neg_hidden.shape == (n, DM)
    assert data.sigma_bucket.shape == (n,)
    assert data.sigma_bucket.dtype == torch.long
    assert data.sigma_bucket.max() <= 3


def test_pairs_from_files_dual_refuses_v1_files(tmp_path):
    files = write_v2_cache(tmp_path, drop_v2_fields=True)
    with pytest.raises(ValueError, match="neg_hidden/sigma missing"):
        pairs_from_files(files, dual_stream=True)


def test_filter_flagged(tmp_path):
    files = write_v2_cache(tmp_path)
    flags = {"flagged": [files[0].stem, "cv2_600w_deadbeef"]}
    fp = tmp_path / "flags.json"
    fp.write_text(json.dumps(flags))
    keep = filter_flagged(files, fp)
    assert len(keep) == len(files) - 1 and files[0] not in keep
    with pytest.raises(ValueError, match="matched no files"):
        filter_flagged(keep, fp) if not any(f.stem in flags["flagged"] for f in keep) else None


def test_train_and_roundtrip_dual(tmp_path):
    files = write_v2_cache(tmp_path)
    data = pairs_from_files(files, dual_stream=True)
    head = v2_head()
    out = train(head, data, steps=20, batch_size=16, ema_decay=0.999, log_every=10)
    assert all(torch.isfinite(torch.tensor(out["losses"])).tolist())
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, head, out["ema"], data, step=20)
    loaded, mean, std = load_checkpoint(p)
    assert loaded.cfg.dual_stream and loaded.cfg.sigma_buckets == 4
    z = sample_latents(
        loaded,
        data.hidden[:6],
        mean,
        std,
        nfe=2,
        neg_condition=data.neg_hidden[:6],
        sampler=heun_sample,
        seed=0,
    )
    assert z.shape == (6, DL) and torch.isfinite(z).all()


def test_train_refuses_dual_pool_on_v1_head(tmp_path):
    files = write_v2_cache(tmp_path)
    data = pairs_from_files(files, dual_stream=True)
    v1 = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=32, layers=2))
    with pytest.raises(ValueError, match="neg_condition given"):
        train(v1, data, steps=2, batch_size=8)


def test_v1_checkpoint_config_still_loads():
    # old checkpoints' config dicts lack the new keys — defaults must apply
    old_cfg = {"d_model": DM, "d_latent": DL, "width": 32, "layers": 2, "ffn_ratio": 2.0}
    head = FlowHead(FlowHeadConfig(**old_cfg))
    assert not head.cfg.dual_stream and head.cfg.sigma_buckets == 0


class StubGen:
    """Minimal model exposing sample_speech_tokens for patch tests."""

    def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=None):
        raise AssertionError("patch should have replaced this")


def test_dual_patch_passes_neg_and_refuses_missing():
    head = v2_head()
    mean, std = torch.zeros(DL), torch.ones(DL)
    model = StubGen()
    with DualStreamFlowHeadPatch(model, head, mean, std, nfe=2) as patch:
        z = model.sample_speech_tokens(
            torch.randn(3, DM), neg_condition=torch.randn(3, DM), cfg_scale=1.3
        )
        assert z.shape == (3, DL) and patch.calls == 1
        with pytest.raises(ValueError, match="no neg_condition"):
            model.sample_speech_tokens(torch.randn(3, DM))
    # context exit restored the class method
    with pytest.raises(AssertionError):
        model.sample_speech_tokens(torch.randn(3, DM))


def test_dual_patch_refuses_v1_head():
    v1 = FlowHead(FlowHeadConfig(d_model=DM, d_latent=DL, width=32, layers=2))
    with pytest.raises(ValueError, match="requires a dual_stream head"):
        DualStreamFlowHeadPatch(StubGen(), v1, torch.zeros(DL), torch.ones(DL))
