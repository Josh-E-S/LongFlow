"""Mac-side scoring for Gate Night 5.

Usage, from repo root:
    unzip -o ~/Downloads/gate_night5_bundle.zip -d experiments/p1_flow_head/audio/gate_night5
    .venv/bin/python experiments/p1_flow_head/score_gate_night5.py

Arm F: collapse horizons per feedback condition (per-window ECAPA vs the GN3
teacher reference). Arm R: reseed-floor median + IQR pair-WER. Arm S: seam
ECAPA vs within-chunk baseline (ears remain the authority). Arm D and H are
echoed from the Colab report. Criteria: NOTES, Gate Night 5 entry.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval.metrics import _ecapa, _normalizer, _whisper  # noqa: E402

AUD = Path(__file__).parent / "audio" / "gate_night5"
TEACHER = Path(__file__).parent / "audio" / "gate_night3" / "t1_turnsplit_p0.wav"
with open(AUD / "gate_night5_report.json") as f:
    report = json.load(f)

WIN, HOP = 4.0, 2.0
ecapa = _ecapa()


def win_embs(path):
    import torchaudio.functional as taf

    x, sr = sf.read(path, dtype="float32")
    x16 = taf.resample(torch.from_numpy(x), sr, 16000).numpy()
    dur = len(x) / sr
    E, ts = [], []
    for i in range(int((dur - WIN) // HOP) + 1):
        seg = x16[int(i * HOP * 16000) : int((i * HOP + WIN) * 16000)]
        if len(seg) < 16000:
            break
        E.append(ecapa.encode_batch(torch.from_numpy(seg)[None])[0, 0].detach())
        ts.append(i * HOP)
    return torch.stack(E), np.array(ts), dur


def transcribe(path):
    segs, _ = _whisper().transcribe(str(path), language="en", beam_size=1)
    return " ".join((s.text or "").strip() for s in segs)


results = {"armF": {}, "armR": {}, "armS": {}, "verdicts": {}}
ref_E, _, _ = win_embs(TEACHER)
ref = ref_E.median(0).values

# ---- Arm F: horizons per condition (prefer v2 — v1 corrupted the prompt path) ----
horizons = {}
for mode in ("base", "zero", "mean", "noise"):
    p = AUD / f"f2_abl_{mode}.wav"
    if not p.exists():
        p = AUD / f"f_abl_{mode}.wav"
    if not p.exists():
        continue
    E, ts, dur = win_embs(p)
    sim = torch.nn.functional.cosine_similarity(E, ref[None], dim=-1).numpy()
    voice = sim >= 0.5
    horizon = 0.0
    for i in range(len(ts)):
        if voice[i]:
            horizon = ts[i] + WIN
    words = len(transcribe(p).split())
    horizons[mode] = {
        "duration_s": round(dur, 1),
        "horizon_s": horizon,
        "pct_voice": round(100 * float(voice.mean()), 1),
        "whisper_words": words,
    }
    print(
        f"F/{mode}: horizon {horizon:.0f}s, {horizons[mode]['pct_voice']}% voice, {words} words",
        flush=True,
    )
results["armF"] = horizons
if "base" in horizons:
    base_h = max(horizons["base"]["horizon_s"], 1.0)
    best = max(
        (m for m in horizons if m != "base"), key=lambda m: horizons[m]["horizon_s"], default=None
    )
    if best:
        ratio = horizons[best]["horizon_s"] / base_h
        results["verdicts"]["armF"] = (
            f"ACOUSTIC CHANNEL CONFIRMED — {best} survives {ratio:.1f}x baseline; inference-time fix exists"
            if ratio >= 3.0
            else f"NO PLUMBING RESCUE (best {best} {ratio:.1f}x) — training path confirmed"
        )

# ---- Arm R: reseed floor ----
import jiwer  # noqa: E402

n = _normalizer()
manifest = report.get("reseed_manifest", {})
pair = []
for uid in manifest:
    a, b = AUD / f"r_{uid}_s0.wav", AUD / f"r_{uid}_s1.wav"
    if a.exists() and b.exists():
        w = jiwer.wer(n(transcribe(a)), n(transcribe(b)))
        pair.append({"uid": uid, "pair_wer": round(w, 3)})
        print(f"R/{uid}: {w:.3f}", flush=True)
if pair:
    ws = sorted(r["pair_wer"] for r in pair)
    med = ws[len(ws) // 2]
    q1, q3 = ws[len(ws) // 4], ws[3 * len(ws) // 4]
    results["armR"] = {
        "n": len(pair),
        "median": round(med, 3),
        "iqr": [round(q1, 3), round(q3, 3)],
        "pairs": pair,
    }
    results["verdicts"]["armR"] = f"FLOOR: median {med:.3f}, IQR [{q1:.3f},{q3:.3f}], n={len(pair)}"

# ---- Arm S: seams ----
chunk_wavs = sorted(AUD.glob("s_chunk_*.wav"))
W = 3 * 24000
seams, base = [], []


def emb_seg(x):
    import torchaudio.functional as taf

    return ecapa.encode_batch(
        taf.resample(torch.from_numpy(x.astype(np.float32)), 24000, 16000)[None]
    )[0, 0].detach()


for i in range(len(chunk_wavs) - 1):
    a, _ = sf.read(chunk_wavs[i], dtype="float32")
    b, _ = sf.read(chunk_wavs[i + 1], dtype="float32")
    if len(a) < 2 * W or len(b) < W:
        continue
    seams.append(
        float(torch.nn.functional.cosine_similarity(emb_seg(a[-W:]), emb_seg(b[:W]), dim=-1))
    )
    base.append(
        float(
            torch.nn.functional.cosine_similarity(emb_seg(a[-2 * W : -W]), emb_seg(a[-W:]), dim=-1)
        )
    )
if seams:
    results["armS"] = {
        "seam_mean": round(float(np.mean(seams)), 3),
        "within_mean": round(float(np.mean(base)), 3),
    }
    results["verdicts"]["armS"] = (
        "metric-pass"
        if np.mean(seams) >= np.mean(base) - 0.10
        else "metric-flag (ears are the authority)"
    )

# ---- echo D and H ----
results["armH"] = report.get("hygiene_AA")
if results["armH"]:
    results["verdicts"]["armH"] = (
        "CLEAN — no cross-contamination"
        if results["armH"]["same_length"] and results["armH"]["maxdiff"] < 1e-4
        else "CONTAMINATION — batched rows diverge; investigate masking"
    )
fd = report.get("fd_curves", {})
if fd:
    results["armD_fd_summary"] = {
        k: {
            "first": round(v[0], 1),
            "median": round(sorted(v)[len(v) // 2], 1),
            "last": round(v[-1], 1),
        }
        for k, v in fd.items()
    }

dest = Path(__file__).parent / "gate_night5_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print("\nVERDICTS:", json.dumps(results["verdicts"], indent=2))
print(f"wrote {dest}")
print(
    "\nListening (never skipped): s_chunked_stitched.wav (3 seams) and the best-surviving f_abl_* render."
)
