"""Tier-1 tests for P0 Stage 2 extraction (src/steering/extract.py).

Synthetic records with a planted per-layer direction + noise: extraction must
recover the planted direction, consistency must separate signal from noise, and
independence must detect shared vs orthogonal axes.
"""

import pytest
import torch

from src.steering.contrast_pairs import PairRecord, save_records
from src.steering.extract import (
    candidate_layers,
    consistency,
    extract_all,
    extract_axis,
    independence,
    per_script_directions,
)

L, D = 6, 32
SIGNAL_LAYER = 3


def planted_records(axis, direction, n_scripts=6, k=2, strength=3.0, noise=0.5, seed=0):
    """pos = base + strength*direction (+noise), neg = base (+noise); the planted
    direction lives only on SIGNAL_LAYER, other layers are pure noise."""
    g = torch.Generator().manual_seed(seed)
    records = []
    for s in range(n_scripts):
        base = torch.randn(L, D, generator=g)
        for pole in ("pos", "neg"):
            for kk in range(k):
                v = base + noise * torch.randn(L, D, generator=g)
                if pole == "pos":
                    v[SIGNAL_LAYER] += strength * direction
                records.append(
                    PairRecord(
                        script_id=f"s{s}",
                        axis=axis,
                        pole=pole,
                        sample_idx=kk,
                        layer_vectors=v,
                        num_frames_kept=10,
                        num_calls_total=20,
                    )
                )
    return records


DIR_A = torch.nn.functional.normalize(
    torch.randn(D, generator=torch.Generator().manual_seed(7)), dim=0
)
DIR_B = torch.nn.functional.normalize(
    torch.randn(D, generator=torch.Generator().manual_seed(8))
    - (torch.randn(D, generator=torch.Generator().manual_seed(8)) @ DIR_A) * DIR_A,
    dim=0,
)  # orthogonal to DIR_A


def test_extraction_recovers_planted_direction():
    ext = extract_axis(planted_records("arousal", DIR_A), "arousal")
    cos = torch.dot(ext.direction[SIGNAL_LAYER], DIR_A).abs()
    assert cos > 0.9
    assert ext.norms.argmax().item() == SIGNAL_LAYER
    assert ext.num_scripts == 6


def test_consistency_high_on_signal_low_on_noise_layers():
    dirs = per_script_directions(planted_records("arousal", DIR_A), "arousal")
    cons = consistency(dirs)
    assert cons[SIGNAL_LAYER] > 0.5
    noise_layers = [i for i in range(L) if i != SIGNAL_LAYER]
    assert cons[noise_layers].mean() < 0.3


def test_independence_detects_shared_vs_orthogonal():
    ext_a = extract_axis(planted_records("arousal", DIR_A), "arousal")
    ext_same = extract_axis(planted_records("valence", DIR_A, seed=1), "valence")
    ext_orth = extract_axis(planted_records("valence", DIR_B, seed=2), "valence")
    assert independence(ext_a, ext_same)[SIGNAL_LAYER].abs() > 0.8
    assert independence(ext_a, ext_orth)[SIGNAL_LAYER].abs() < 0.4


def test_candidate_layers_rank_signal_first():
    ext = extract_axis(planted_records("arousal", DIR_A), "arousal")
    assert candidate_layers(ext, top_k=3)[0] == SIGNAL_LAYER


def test_incomplete_pairs_skipped_and_empty_axis_raises():
    records = planted_records("arousal", DIR_A, n_scripts=3)
    # drop every neg record of script s0 -> s0 incomplete, still 2 usable scripts
    records = [r for r in records if not (r.script_id == "s0" and r.pole == "neg")]
    dirs = per_script_directions(records, "arousal")
    assert set(dirs) == {"s1", "s2"}
    with pytest.raises(ValueError, match="no complete"):
        per_script_directions(records, "valence")


def test_extract_all_roundtrip_and_summary(tmp_path):
    records = planted_records("arousal", DIR_A) + planted_records("valence", DIR_B, seed=3)
    p = tmp_path / "vectors.pt"
    save_records(records, p)
    result = extract_all(p)
    assert set(result["directions"]) == {"arousal", "valence"}
    assert result["directions"]["arousal"].shape == (L, D)
    assert "independence" in result
    from src.steering.extract import summarize

    text = summarize(result)
    assert "arousal" in text and "valence" in text and "cosine" in text
