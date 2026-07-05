"""T-alignment convention tests (CLAUDE.md hard constraint 3)."""

import pytest
import torch

from src.cache.alignment import assert_frame_aligned


def test_aligned_pair_passes():
    assert_frame_aligned(torch.zeros(2, 100, 1536), torch.zeros(2, 100, 64))


def test_t_mismatch_raises():
    with pytest.raises(ValueError, match="misalignment"):
        assert_frame_aligned(torch.zeros(2, 100, 1536), torch.zeros(2, 99, 64))


def test_nan_pair_refused():
    hidden = torch.randn(2, 50, 1536)
    hidden[0, 3, 7] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        assert_frame_aligned(hidden, torch.zeros(2, 50, 64))
