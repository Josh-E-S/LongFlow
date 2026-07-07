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
_ECAPA = None
_NORMALIZER = None


def _whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel("large-v3", device="cpu", compute_type="int8")
    return _WHISPER


def _ecapa():
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="~/.cache/speechbrain/spkrec-ecapa-voxceleb",
        )
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


def wer(path, reference_text: str) -> dict:
    """Whisper-large-v3 transcription WER vs reference (both normalized)."""
    import jiwer

    segments, _info = _whisper().transcribe(str(path), language="en", beam_size=5)
    hyp = " ".join(s.text for s in segments)
    norm = _normalizer()
    ref_n, hyp_n = norm(reference_text), norm(hyp)
    return {"wer": jiwer.wer(ref_n, hyp_n), "transcript": hyp.strip()}


def speaker_similarity(path, reference_path) -> float:
    """ECAPA cosine similarity between a clip and the reference voice."""
    model = _ecapa()
    embs = []
    for p in (path, reference_path):
        wav = load_16k_mono(p).unsqueeze(0)
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


def clip_metrics(path, reference_text: str, reference_voice_path) -> dict:
    return {
        **wer(path, reference_text),
        "speaker_sim": speaker_similarity(path, reference_voice_path),
        **prosody(path),
    }
