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
