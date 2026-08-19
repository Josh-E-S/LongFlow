"""Objective eval metrics (P0 Stage 3 gates; reused by later phases).

Pinned stack per docs/resources.md §3: faster-whisper large-v3 + jiwer +
whisper-normalizer for WER; speechbrain ECAPA for speaker similarity (iteration
gate — WavLM-large SV is the paper-table metric, added later); parselmouth for
F0; RMS for energy. All models lazy-loaded and cached at module level so a
sweep pays load cost once. CPU-friendly (int8 whisper).
"""

import numpy as np
import soundfile as sf
import torch

_WHISPER = None
_WHISPER_DEVICE = None
_ECAPA = None
_ECAPA_DEVICE = None
_NORMALIZER = None


def _whisper(device: str = "cpu"):
    """device: 'cpu' (default -- Mac-side scoring) or 'cuda' (Colab GPU
    scoring, dramatically faster on long clips). Re-loads if a different
    device is requested than what's cached -- fine since a single process
    only ever uses one device in practice."""
    global _WHISPER, _WHISPER_DEVICE
    if _WHISPER is None or device != _WHISPER_DEVICE:
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        _WHISPER = WhisperModel("large-v3", device=device, compute_type=compute_type)
        _WHISPER_DEVICE = device
    return _WHISPER


def _ecapa(device: str = "cpu"):
    global _ECAPA, _ECAPA_DEVICE
    if _ECAPA is None or device != _ECAPA_DEVICE:
        from speechbrain.inference.speaker import EncoderClassifier

        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="~/.cache/speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _ECAPA_DEVICE = device
    return _ECAPA


def _normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _NORMALIZER = EnglishTextNormalizer()
    return _NORMALIZER


def load_16k_mono(path) -> torch.Tensor:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    t = torch.from_numpy(x)
    if sr != 16000:
        import torchaudio.functional as taf

        t = taf.resample(t, sr, 16000)
    return t


def wer(path, reference_text: str, device: str = "cpu") -> dict:
    """Whisper-large-v3 transcription WER vs reference (both normalized).
    device='cuda' for GPU scoring (Colab) -- dramatically faster on long
    clips than the default CPU path (Mac-side scoring)."""
    import jiwer

    segments, _info = _whisper(device).transcribe(str(path), language="en", beam_size=5)
    segments = list(segments)
    hyp = " ".join(s.text for s in segments)
    norm = _normalizer()
    ref_n, hyp_n = norm(reference_text), norm(hyp)
    # asr_confidence: mean Whisper log-prob — a free fluency signal the project
    # discarded until 2026-08-19; low confidence on CORRECT words flags
    # "intelligible but degraded" audio that WER alone scores as clean
    conf = (
        float(np.mean([s.avg_log_prob for s in segments]))
        if segments and hasattr(segments[0], "avg_log_prob")
        else (float(np.mean([s.avg_logprob for s in segments])) if segments else float("nan"))
    )
    return {"wer": jiwer.wer(ref_n, hyp_n), "transcript": hyp.strip(), "asr_confidence": conf}


def speaker_similarity(path, reference_path, device: str = "cpu") -> float:
    """ECAPA cosine similarity between a clip and the reference voice."""
    model = _ecapa(device)
    embs = []
    for p in (path, reference_path):
        wav = load_16k_mono(p).unsqueeze(0).to(device)
        with torch.no_grad():
            embs.append(model.encode_batch(wav).squeeze())
    return float(torch.nn.functional.cosine_similarity(embs[0], embs[1], dim=-1))


def prosody(path) -> dict:
    """F0 mean/std over voiced frames (parselmouth/Praat) + RMS energy (dBFS)."""
    import parselmouth

    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    x, _sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)
    return {
        "f0_mean": float(voiced.mean()) if len(voiced) else 0.0,
        "f0_std": float(voiced.std()) if len(voiced) else 0.0,
        "voiced_fraction": float(len(voiced) / max(len(f0), 1)),
        "rms_dbfs": float(rms_dbfs),
        "duration_s": float(len(x) / _sr),
    }


def ear_pack(path, win_s: float = 2.0, hop_s: float = 1.0) -> dict:
    """Per-window voice-cleanliness curves — the EAR PACK (2026-08-19).

    Seven ear-beats-metrics incidents happened because WER measures content
    and ECAPA measures identity; NEITHER measures voice cleanliness, the axis
    Josh's ear actually ranks by. These clinical/phonetic metrics do:

    - hnr: harmonics-to-noise ratio, dB (the "cold/snow" axis — teacher holds
      ~13.9 flat; forensics 2026-08-19)
    - cpps: smoothed cepstral peak prominence (clinical gold standard for
      breathiness; typically more sensitive than HNR)
    - jitter / shimmer: cycle-to-cycle pitch / amplitude instability
      ("shaky/wobbly")
    - flatness: spectral flatness 0.2–8 kHz (noisiness / "fuzz")
    - hf_ratio: 4–10 kHz vs 0.2–4 kHz energy (hiss/snow band)

    Returns {"t": centers, <metric>: curve, <metric>_median: float}. Praat
    calls that fail on a window (silence, too short) yield NaN, excluded
    from medians. CPU-only (parselmouth + numpy); a 5-min clip ≈ tens of
    seconds.
    """
    import parselmouth
    from parselmouth.praat import call as praat

    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    harm = snd.to_harmonicity_cc(time_step=0.05)
    hnr_t = np.array(harm.xs())
    hnr_v = harm.values[0]

    n_win = max(int((dur - win_s) / hop_s) + 1, 1)
    win_n = int(win_s * sr)
    freqs = np.fft.rfftfreq(win_n, 1 / sr)
    hann = np.hanning(win_n)
    band = (freqs >= 200) & (freqs <= 8000)
    low_b = (freqs >= 200) & (freqs < 4000)
    hf_b = (freqs >= 4000) & (freqs < 10000)

    out = {k: [] for k in ("t", "hnr", "cpps", "jitter", "shimmer", "flatness", "hf_ratio")}
    for i in range(n_win):
        t0 = i * hop_s
        seg = x[int(t0 * sr) : int(t0 * sr) + win_n]
        if len(seg) < win_n:
            break
        out["t"].append(t0 + win_s / 2)
        m = (hnr_t >= t0) & (hnr_t < t0 + win_s)
        hv = hnr_v[m]
        hv = hv[hv > -50]  # Praat marks silence ~-200
        out["hnr"].append(float(np.mean(hv)) if len(hv) else float("nan"))
        P = np.abs(np.fft.rfft(seg * hann)) ** 2 + 1e-12
        out["flatness"].append(float(np.exp(np.mean(np.log(P[band]))) / np.mean(P[band])))
        out["hf_ratio"].append(float(P[hf_b].sum() / (P[low_b].sum() + 1e-9)))
        sw = parselmouth.Sound(seg, sampling_frequency=sr)
        try:
            pcg = praat(sw, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
            out["cpps"].append(
                float(
                    praat(
                        pcg,
                        "Get CPPS",
                        False,
                        0.02,
                        0.0005,
                        60.0,
                        330.0,
                        0.05,
                        "parabolic",
                        0.001,
                        0.05,
                        "Straight",
                        "Robust",
                    )
                )
            )
        except Exception:
            out["cpps"].append(float("nan"))
        try:
            pp = praat(sw, "To PointProcess (periodic, cc)", 75.0, 500.0)
            out["jitter"].append(float(praat(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)))
            out["shimmer"].append(
                float(praat([sw, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            )
        except Exception:
            out["jitter"].append(float("nan"))
            out["shimmer"].append(float("nan"))

    result = {k: v for k, v in out.items()}
    for k in ("hnr", "cpps", "jitter", "shimmer", "flatness", "hf_ratio"):
        v = np.array(out[k], dtype=float)
        v = v[np.isfinite(v)]
        result[f"{k}_median"] = float(np.median(v)) if len(v) else float("nan")
    return result


def clip_metrics(path, reference_text: str, reference_voice_path, device: str = "cpu") -> dict:
    return {
        **wer(path, reference_text, device=device),
        "speaker_sim": speaker_similarity(path, reference_voice_path, device=device),
        **prosody(path),
    }


def clip_metrics_ear(path, reference_text: str, reference_voice_path, device: str = "cpu") -> dict:
    """clip_metrics + the ear-pack medians (curves dropped for table rows)."""
    ep = ear_pack(path)
    return {
        **clip_metrics(path, reference_text, reference_voice_path, device=device),
        **{k: v for k, v in ep.items() if k.endswith("_median")},
    }
