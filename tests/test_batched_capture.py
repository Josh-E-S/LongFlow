"""Tier-1 tests for BatchedSampleCapture (src/cache/capture.py).

Simulates VibeVoice batched generation: elements finish at different steps;
each sample_speech_tokens call carries rows only for that step's active
elements (ascending batch order). Attribution must be exact or refuse loudly.
"""

import pytest
import torch

from src.cache.capture import BatchedSampleCapture

DM, DL = 12, 4
FRAME, END = 100, 102


class StubModel:
    def sample_speech_tokens(self, condition, neg_condition=None, cfg_scale=3.0):
        return condition[..., :DL] * 2.0


def simulate(streams):
    """Run a batched 'generation': at each step, call once with rows for the
    active elements. Row values encode (element, per-element frame index) so
    attribution can be verified exactly."""
    model = StubModel()
    frame_counter = [0] * len(streams)
    cap = BatchedSampleCapture(model)
    with cap:
        n_steps = max(len(s) for s in streams)
        for t in range(n_steps):
            active = [b for b, s in enumerate(streams) if t < len(s) and s[t] == FRAME]
            if not active:
                continue
            rows = []
            for b in active:
                # ids must stay < 256: bf16 rounds larger integers
                v = torch.full((DM,), float(b * 100 + frame_counter[b]))
                frame_counter[b] += 1
                rows.append(v)
            model.sample_speech_tokens(torch.stack(rows).to(torch.bfloat16), None, 1.3)
    return cap


def test_split_reassigns_rows_exactly():
    streams = [
        [FRAME, FRAME, FRAME, END],  # element 0: 3 frames, finishes early
        [FRAME, FRAME, FRAME, FRAME, FRAME, END],  # element 1: 5 frames
        [FRAME, END, FRAME, FRAME, END],  # element 2: gap mid-stream
    ]
    cap = simulate(streams)
    parts = cap.split_utterances(streams, frame_id=FRAME)
    assert [p[0].shape for p in parts] == [(3, DM), (5, DM), (3, DM)]
    for b, (hidden, latent) in enumerate(parts):
        for i in range(hidden.shape[0]):
            assert float(hidden[i, 0]) == b * 100 + i  # right element, right order
        assert torch.allclose(latent.float(), hidden[:, :DL].float() * 2.0)


def test_call_step_mismatch_refuses():
    streams = [[FRAME, FRAME]]
    cap = simulate(streams)
    cap.calls.append(cap.calls[-1])  # phantom extra call
    with pytest.raises(RuntimeError, match="call/step mismatch"):
        cap.split_utterances(streams, frame_id=FRAME)


def test_row_count_mismatch_refuses():
    streams = [[FRAME], [FRAME]]
    cap = simulate(streams)
    cond, lat = cap.calls[0]
    cap.calls[0] = (cond[:1], lat[:1])  # drop a row
    with pytest.raises(RuntimeError, match="row/active mismatch"):
        cap.split_utterances(streams, frame_id=FRAME)


def test_frame_total_mismatch_refuses():
    streams = [[FRAME, FRAME, FRAME]]
    cap = simulate(streams)
    with pytest.raises(RuntimeError, match="call/step mismatch|frames assigned"):
        cap.split_utterances([[FRAME, FRAME]], frame_id=FRAME)  # stream claims fewer


def test_method_restored_after_exit():
    model = StubModel()
    with BatchedSampleCapture(model):
        model.sample_speech_tokens(torch.ones(2, DM), None, 1.3)
    assert "sample_speech_tokens" not in vars(model)


def test_uneven_stream_lengths_pad_safe():
    # shorter stream simply has no entries at late steps (as if padded post-hoc)
    streams = [[FRAME], [FRAME, FRAME, FRAME]]
    cap = simulate(streams)
    parts = cap.split_utterances(streams, frame_id=FRAME)
    assert parts[0][0].shape == (1, DM) and parts[1][0].shape == (3, DM)
