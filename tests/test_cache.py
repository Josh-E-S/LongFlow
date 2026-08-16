"""Tier-1 tests for P1 caching (src/cache/capture.py)."""

import pytest
import torch

from src.cache.capture import SampleCapture, load_utterance, save_utterance

DM, DL = 24, 8


class StubModel:
    """Stands in for VibeVoice: sample_speech_tokens maps condition -> latent."""

    def __init__(self):
        self.calls = 0

    def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=3.0):
        self.calls += 1
        return condition[..., :DL] * 2.0  # deterministic, shape [B, DL]


def run_frames(model, n, batch=1):
    outs = []
    for i in range(n):
        cond = torch.full((batch, DM), float(i), dtype=torch.bfloat16)
        outs.append(model.sample_speech_tokens(cond, None, 1.3))
    return outs


def test_capture_records_pairs_and_restores_method():
    model = StubModel()
    with SampleCapture(model) as cap:
        outs = run_frames(model, 5)
        assert cap.num_frames == 5
    # class-method lookup restored: no instance-level shadow left behind
    assert "sample_speech_tokens" not in vars(model)
    run_frames(model, 2)
    assert cap.num_frames == 5  # no capture outside context
    # captured latents match what the wrapped call returned
    assert torch.allclose(cap.latents[0].float(), outs[0].float())


def test_wrapped_call_is_transparent():
    model = StubModel()
    with SampleCapture(model):
        out = model.sample_speech_tokens(torch.ones(1, DM), None, 1.3)
    assert out.shape == (1, DL)
    assert torch.allclose(out, torch.ones(1, DL) * 2.0)


def test_to_utterance_shapes_and_dtype():
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 7)
    utt = cap.to_utterance("utt1", "hello world", meta={"revision": "abc"})
    assert utt.hidden.shape == (7, DM) and utt.latent.shape == (7, DL)
    assert utt.hidden.dtype == torch.float16 and utt.latent.dtype == torch.float16
    assert utt.meta["revision"] == "abc"


def test_empty_capture_raises():
    model = StubModel()
    with SampleCapture(model) as cap:
        pass
    with pytest.raises(ValueError, match="no frames"):
        cap.to_utterance("utt1", "text")


def test_reset_between_utterances():
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 3)
        first = cap.to_utterance("u1", "a")
        cap.reset()
        run_frames(model, 4)
        second = cap.to_utterance("u2", "b")
    assert first.hidden.shape[0] == 3 and second.hidden.shape[0] == 4


def test_save_load_roundtrip_with_alignment_check(tmp_path):
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 6)
    utt = cap.to_utterance("u1", "some text", meta={"speaker": "s1"})
    p = tmp_path / "u1.pt"
    save_utterance(utt, p)
    loaded = load_utterance(p)
    assert loaded.utt_id == "u1" and loaded.text == "some text"
    assert torch.allclose(loaded.hidden, utt.hidden)
    assert torch.allclose(loaded.latent, utt.latent)


def test_load_rejects_corrupt_pair(tmp_path):
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 6)
    utt = cap.to_utterance("u1", "t")
    d = {
        "utt_id": utt.utt_id,
        "text": utt.text,
        "meta": utt.meta,
        "hidden": utt.hidden,
        "latent": utt.latent[:-1],  # truncated latent
    }
    p = tmp_path / "bad.pt"
    torch.save(d, p)
    with pytest.raises(ValueError, match="misalignment"):
        load_utterance(p)


def test_nan_condition_refused():
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 3)
        bad = torch.full((1, DM), float("nan"), dtype=torch.bfloat16)
        model.sample_speech_tokens(bad, None, 1.3)
    with pytest.raises(ValueError, match="non-finite"):
        cap.to_utterance("u1", "t")


def test_to_utterance_rejects_multi_row_calls():
    """A [2, d] condition per call must raise, not flatten to [1, 2d] (audit finding 3)."""
    import pytest

    class MultiRowStub:
        def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=3.0):
            return condition[..., :4]

    with SampleCapture(MultiRowStub()) as cap:
        cap.model.sample_speech_tokens(torch.randn(2, 16))
    with pytest.raises(ValueError, match="multi-row"):
        cap.to_utterance("u0", "text")


# ---------------------------------------------------------------------------
# Capture v2: dual-stream (neg_condition) + sigma bucket (GN8 amendment)
# ---------------------------------------------------------------------------


def test_dual_stream_capture_records_neg_hidden():
    model = StubModel()
    with SampleCapture(model) as cap:
        for i in range(5):
            cond = torch.full((1, DM), float(i), dtype=torch.bfloat16)
            neg = torch.full((1, DM), float(-i), dtype=torch.bfloat16)
            model.sample_speech_tokens(cond, neg, 1.3)
    utt = cap.to_utterance("u1", "text")
    assert utt.neg_hidden is not None
    assert utt.neg_hidden.shape == (5, DM)
    assert torch.allclose(utt.neg_hidden.float(), -utt.hidden.float())


def test_no_neg_condition_leaves_neg_hidden_none():
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 4)  # run_frames passes neg_condition=None
    utt = cap.to_utterance("u1", "text")
    assert utt.neg_hidden is None


def test_partial_dual_stream_refused():
    import pytest

    model = StubModel()
    with SampleCapture(model) as cap:
        model.sample_speech_tokens(torch.ones(1, DM), torch.ones(1, DM), 1.3)
        model.sample_speech_tokens(torch.ones(1, DM), None, 1.3)  # missing on frame 2
    with pytest.raises(ValueError, match="not all"):
        cap.to_utterance("u1", "text")


class StubNoise:
    """Minimal stand-in for NoiseIntervention: exposes .last_sigma only."""

    def __init__(self):
        self.last_sigma = 0.0


def test_sigma_recorded_per_frame_when_noise_wired():
    model = StubModel()
    noise = StubNoise()
    with SampleCapture(model, noise=noise) as cap:
        for sigma in (0.0, 0.2, 0.3):
            noise.last_sigma = sigma
            model.sample_speech_tokens(torch.ones(1, DM), None, 1.3)
    utt = cap.to_utterance("u1", "text")
    assert utt.sigma is not None
    assert torch.allclose(utt.sigma.float(), torch.tensor([0.0, 0.2, 0.3]), atol=1e-3)


def test_sigma_absent_without_noise_wired():
    model = StubModel()
    with SampleCapture(model) as cap:
        run_frames(model, 3)
    utt = cap.to_utterance("u1", "text")
    assert utt.sigma is None


def test_v2_save_load_roundtrip(tmp_path):
    model = StubModel()
    noise = StubNoise()
    with SampleCapture(model, noise=noise) as cap:
        for i, sigma in enumerate((0.1, 0.2)):
            noise.last_sigma = sigma
            cond = torch.full((1, DM), float(i), dtype=torch.bfloat16)
            neg = torch.full((1, DM), float(10 + i), dtype=torch.bfloat16)
            model.sample_speech_tokens(cond, neg, 1.3)
    utt = cap.to_utterance("u1", "text")
    p = tmp_path / "u1.pt"
    save_utterance(utt, p)
    loaded = load_utterance(p)
    assert torch.allclose(loaded.neg_hidden, utt.neg_hidden)
    assert torch.allclose(loaded.sigma, utt.sigma)


def test_v1_cache_loads_without_v2_fields(tmp_path):
    """An old .pt saved before the v2 fields existed must still load clean."""
    p = tmp_path / "old.pt"
    torch.save(
        {
            "utt_id": "u1",
            "text": "t",
            "hidden": torch.zeros(3, DM, dtype=torch.float16),
            "latent": torch.zeros(3, DL, dtype=torch.float16),
            "meta": {},
        },
        p,
    )
    utt = load_utterance(p)
    assert utt.neg_hidden is None
    assert utt.sigma is None
