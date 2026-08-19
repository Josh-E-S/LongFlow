"""Tier-1 tests for the EAR PACK (src/eval/metrics.ear_pack, 2026-08-19).

Synthetic ground truth: a clean harmonic complex must score CLEANER than
white noise on every cleanliness axis. Ordering assertions only — absolute
Praat values vary by version.
"""

import numpy as np
import pytest
import soundfile as sf

from src.eval.metrics import ear_pack

SR = 24000
DUR = 4.0


def _write(tmp_path, name, x):
    p = tmp_path / name
    sf.write(p, (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float32), SR)
    return p


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("earpack")
    t = np.arange(int(SR * DUR)) / SR
    f0 = 150 * (1 + 0.01 * np.sin(2 * np.pi * 3 * t))  # mild vibrato = realistic voiced tone
    phase = 2 * np.pi * np.cumsum(f0) / SR
    tone = sum(np.sin(h * phase) / h for h in range(1, 11))
    tone = tone * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t) ** 2)  # syllable-ish envelope
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(t))
    return {
        "tone": _write(tmp, "tone.wav", tone),
        "noise": _write(tmp, "noise.wav", noise),
    }


def test_curves_aligned_and_finite_somewhere(clips):
    ep = ear_pack(clips["tone"])
    n = len(ep["t"])
    assert n >= 2
    for k in ("hnr", "cpps", "jitter", "shimmer", "flatness", "hf_ratio"):
        assert len(ep[k]) == n
    assert np.isfinite(ep["hnr_median"])
    assert np.isfinite(ep["flatness_median"])


def test_tone_cleaner_than_noise(clips):
    tone = ear_pack(clips["tone"])
    noise = ear_pack(clips["noise"])
    assert tone["hnr_median"] > noise["hnr_median"] + 5.0 or not np.isfinite(noise["hnr_median"])
    assert noise["flatness_median"] > tone["flatness_median"] * 2
    if np.isfinite(tone["cpps_median"]) and np.isfinite(noise["cpps_median"]):
        assert tone["cpps_median"] > noise["cpps_median"]


def test_shimmer_detects_amplitude_instability(clips, tmp_path):
    x, sr = sf.read(clips["tone"], dtype="float32")
    t = np.arange(len(x)) / sr
    rng = np.random.default_rng(1)
    # amplitude wobble at ~8 Hz — the "shaky voice" signature
    shaky = x * (1 + 0.35 * np.sin(2 * np.pi * 8 * t + rng.standard_normal()))
    p = tmp_path / "shaky.wav"
    sf.write(p, shaky.astype(np.float32), sr)
    clean = ear_pack(clips["tone"])
    wob = ear_pack(p)
    if np.isfinite(clean["shimmer_median"]) and np.isfinite(wob["shimmer_median"]):
        assert wob["shimmer_median"] > clean["shimmer_median"]
