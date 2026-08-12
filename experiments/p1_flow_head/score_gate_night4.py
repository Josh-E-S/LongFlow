"""Mac-side scoring for Gate Night 4.

Usage, from repo root:
    unzip -o ~/Downloads/gate_night4_bundle.zip -d experiments/p1_flow_head/audio/gate_night4
    .venv/bin/python experiments/p1_flow_head/score_gate_night4.py

Arm A vs the pre-registered criteria (NOTES, Gate Night 4 entry), using the
GN3 teacher render t1_turnsplit_p0.wav as the direct A/B reference.
Arm B is self-reporting (per_module_ms in the report json) — echoed here.
Arm C solo-vs-batched per utterance, judged against the GN1 reseed floor.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval.metrics import _ecapa, _whisper, load_16k_mono  # noqa: E402

AUD = Path(__file__).parent / "audio" / "gate_night4"
TEACHER_T1 = Path(__file__).parent / "audio" / "gate_night3" / "t1_turnsplit_p0.wav"
with open(AUD / "gate_night4_report.json") as f:
    report = json.load(f)

SCRIPT_WORDS = 3229
TEACHER_T1_WPM = 168.2  # GN3 official


def transcribe(path):
    segs, _ = _whisper().transcribe(str(path), language="en", word_timestamps=False, beam_size=1)
    return " ".join((s.text or "").strip() for s in segs)


def ecapa_curve(path, win=4.0, hop=2.0):
    import torchaudio.functional as taf

    x, sr = sf.read(path, dtype="float32")
    x16 = taf.resample(torch.from_numpy(x), sr, 16000).numpy()
    dur = len(x) / sr
    e = _ecapa()
    embs = []
    for i in range(int((dur - win) // hop) + 1):
        seg = x16[int(i * hop * 16000) : int((i * hop + win) * 16000)]
        if len(seg) < 16000:
            break
        embs.append(e.encode_batch(torch.from_numpy(seg)[None])[0, 0].detach())
    E = torch.stack(embs)
    sim = torch.nn.functional.cosine_similarity(E, E.median(0).values[None], dim=-1).numpy()
    return sim, dur


def rms_halves(path):
    x, _ = sf.read(path, dtype="float32")
    h = len(x) // 2
    db = lambda a: 20 * np.log10(np.sqrt((a**2).mean()) + 1e-9)  # noqa: E731
    return round(db(x[:h]), 1), round(db(x[h:]), 1)


results = {"armA": {}, "armB": report.get("profile", []), "armC": {}, "verdicts": {}}

# ---- Arm A ----
for tag in ("a_head20_turnsplit_euler4", "a_head20_turnsplit_heun8"):
    p = AUD / f"{tag}.wav"
    if not p.exists():
        continue
    words = len(transcribe(p).split())
    dur = sf.info(p).duration
    sim, _ = ecapa_curve(p)
    half = len(sim) // 2
    front, back = float(sim[:half].mean()), float(sim[half:].mean())
    rms_f, rms_b = rms_halves(p)
    a = {
        "audio_min": round(dur / 60, 2),
        "wpm": round(words / (dur / 60), 1),
        "coverage_words": words,
        "coverage_pct": round(100 * words / SCRIPT_WORDS, 1),
        "ecapa_front": round(front, 3),
        "ecapa_back": round(back, 3),
        "rms_front_db": rms_f,
        "rms_back_db": rms_b,
    }
    # pre-registered: PASS = coverage >=95%, no fade (rms_back >= rms_front - 3dB),
    # identity flat (back >= front - 0.05)
    a["fade"] = rms_b < rms_f - 3.0
    a["identity_flat"] = back >= front - 0.05
    a["coverage_ok"] = a["coverage_pct"] >= 95.0
    results["armA"][tag] = a
    print(f"{tag}: {json.dumps(a)}", flush=True)

for tag, a in results["armA"].items():
    s = (
        "PASS"
        if (a["coverage_ok"] and not a["fade"] and a["identity_flat"])
        else ("FAIL (E3-style collapse)" if a["coverage_pct"] < 60 else "PARTIAL")
    )
    results["verdicts"][tag] = s

# ---- Arm C ----
floor_wer = 0.076  # GN1 reseed floor (mean, n=6 — noisy; see GN1 caveats)
e = _ecapa()
tags = [k for k in report.get("parity_texts", {})]
rows = []
for t in tags:
    solo, batch = AUD / f"c_solo_{t}.wav", AUD / f"c_batch_{t}.wav"
    if not (solo.exists() and batch.exists()):
        continue
    import jiwer
    from eval.metrics import _normalizer

    n = _normalizer()
    pair_wer = jiwer.wer(n(transcribe(solo)), n(transcribe(batch)))
    embs = [e.encode_batch(load_16k_mono(p)[None])[0, 0].detach() for p in (solo, batch)]
    pair_sim = float(torch.nn.functional.cosine_similarity(embs[0], embs[1], dim=-1))
    rows.append(
        {"tag": t, "solo_vs_batch_wer": round(pair_wer, 3), "solo_vs_batch_sim": round(pair_sim, 3)}
    )
    print(rows[-1], flush=True)
results["armC"]["pairs"] = rows
if rows:
    mean_wer = float(np.mean([r["solo_vs_batch_wer"] for r in rows]))
    results["armC"]["mean_wer"] = round(mean_wer, 3)
    results["verdicts"]["armC"] = (
        "PARITY OK (within ~reseed noise)"
        if mean_wer <= 1.5 * floor_wer
        else "PARITY DEGRADED — batch generation shifts output"
    )

dest = Path(__file__).parent / "gate_night4_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print("\nVERDICTS:", json.dumps(results["verdicts"], indent=2))
if results["armB"]:
    print("\nBOTTLENECK MAP (ms/frame):")
    for row in results["armB"]:
        print(f"  {row['label']}: total {row['ms_per_frame_total']}  ->  {row['per_module_ms']}")
print(f"wrote {dest}")
print("\nListening (never skipped): a_head20_turnsplit_euler4.wav vs the GN3 teacher")
print(f"render ({TEACHER_T1}) — same script, same prompt. Texture, fade, seams.")
