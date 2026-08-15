"""Mac-side scoring for Gate Night 8 (the CFG repair test).

Usage, from repo root:
    unzip -o ~/Downloads/gate_night8_bundle.zip -d experiments/p1_flow_head/audio/gate_night8
    .venv/bin/python experiments/p1_flow_head/score_gate_night8.py

Compares CFG-corrected closed-loop renders against their same-sampler plain
controls. Per the GN7 instrument rule, Whisper numbers are content-survival
only; the graded sim curve + Josh's ear own the identity/listenability
claims. Criteria: NOTES, Gate Night 8 entry.
"""

import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval.metrics import _ecapa, _normalizer, _whisper  # noqa: E402

AUD = Path(__file__).parent / "audio" / "gate_night8"
TEACHER = Path(__file__).parent / "audio" / "gate_night3" / "t1_turnsplit_p0.wav"
with open(AUD / "gate_night8_report.json") as f:
    report = json.load(f)

TAGS = [
    "c8_euler4_plain_s0",
    "c8_euler4_s0",
    "c8_euler4_s1",
    "c8_heun8_plain_s0",
    "c8_heun8_s0",
    "c8_heun8_s1",
    "t8_cfg10",
]
PAIRS = {  # cfg tag -> its plain control
    "c8_euler4_s0": "c8_euler4_plain_s0",
    "c8_euler4_s1": "c8_euler4_plain_s0",
    "c8_heun8_s0": "c8_heun8_plain_s0",
    "c8_heun8_s1": "c8_heun8_plain_s0",
}
WIN, HOP = 4.0, 2.0
ecapa = _ecapa()
n = _normalizer()

script_words = []
for line in report["sweep_script"].splitlines():
    if ":" in line:
        line = line.split(":", 1)[1]
    script_words.append(line.strip())
SCRIPT = n(" ".join(w for w in script_words if w))
N_SCRIPT = len(SCRIPT.split())


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


def score(path):
    E, ts, dur = win_embs(path)
    sim = torch.nn.functional.cosine_similarity(E, ref[None], dim=-1).numpy()
    voice = sim >= 0.5
    horizon = 0.0
    for i in range(len(ts)):
        if voice[i]:
            horizon = ts[i] + WIN
    hyp = n(transcribe(path))
    n_words = len(hyp.split())
    return {
        "duration_s": round(dur, 1),
        "wer_vs_script": round(jiwer.wer(SCRIPT, hyp) if hyp else 1.0, 3),
        "coverage_pct": round(100 * min(n_words, N_SCRIPT) / N_SCRIPT, 1),
        "voice_pct": round(100 * float(voice.mean()), 1),
        "horizon_s": float(horizon),
        "sim_median": round(float(np.median(sim)), 3),
        "sim_final_third": round(float(np.median(sim[-max(1, len(sim) // 3) :])), 3),
        "rate_wpm": round(60 * n_words / dur, 1),
    }


results = {"conditions": {}, "deltas": {}, "verdicts": {}}
ref_E, _, _ = win_embs(TEACHER)
ref = ref_E.median(0).values

for tag in TAGS:
    p = AUD / f"{tag}.wav"
    if not p.exists():
        continue
    results["conditions"][tag] = score(p)
    print(f"{tag}: {results['conditions'][tag]}", flush=True)

c = results["conditions"]

# ---- per-pair deltas (cfg minus its plain control) ----
for tag, ctrl in PAIRS.items():
    if tag in c and ctrl in c:
        results["deltas"][tag] = {
            k: round(c[tag][k] - c[ctrl][k], 3)
            for k in ("wer_vs_script", "coverage_pct", "sim_median", "horizon_s")
        }


# ---- pre-registered verdict (NOTES, GN8 entry) ----
def strong(tag):
    ctrl = PAIRS[tag]
    return (
        tag in c
        and ctrl in c
        and (
            c[tag]["sim_median"] >= c[ctrl]["sim_median"] + 0.10
            or c[tag]["horizon_s"] >= 3.0 * max(c[ctrl]["horizon_s"], 4.0)
        )
    )


def content_up(tag):
    ctrl = PAIRS[tag]
    return (
        tag in c
        and ctrl in c
        and (
            c[tag]["wer_vs_script"] <= c[ctrl]["wer_vs_script"] - 0.06
            or c[tag]["coverage_pct"] >= c[ctrl]["coverage_pct"] + 10.0
        )
    )


strong_euler = strong("c8_euler4_s0") and strong("c8_euler4_s1")
strong_heun = strong("c8_heun8_s0") and strong("c8_heun8_s1")
content_euler = content_up("c8_euler4_s0") and content_up("c8_euler4_s1")
content_heun = content_up("c8_heun8_s0") and content_up("c8_heun8_s1")

if strong_euler or strong_heun:
    which = [s for s, f in (("euler4", strong_euler), ("heun8", strong_heun)) if f]
    results["verdicts"]["cfg"] = (
        f"CFG REPAIR (strong, metric half) on {which}, both seeds — Josh's ear confirms or denies; "
        "if confirmed, re-baseline on guided sampling; capture v2 + training go dual-stream"
    )
elif content_euler or content_heun:
    which = [s for s, f in (("euler4", content_euler), ("heun8", content_heun)) if f]
    results["verdicts"]["cfg"] = (
        f"PARTIAL — content improves on {which} (both seeds) but identity curve flat; "
        "guided sampling becomes default inference config; capture v2 unchanged, higher priority"
    )
else:
    results["verdicts"]["cfg"] = (
        "NULL — no consistent improvement across seeds; the July head's bias is baked in; "
        "capture v2 proceeds as planned (still recording neg)"
    )

# teacher read: t8_cfg10 vs GN7 clean control (t7_sig000: WER 0.281, voice 98.7, rate 183)
GN7_CTRL = {"wer_vs_script": 0.281, "voice_pct": 98.7, "rate_wpm": 183.0}
if "t8_cfg10" in c:
    r = c["t8_cfg10"]
    degraded = (
        r["wer_vs_script"] > GN7_CTRL["wer_vs_script"] + 0.06
        or r["voice_pct"] < 90.0
        or not (150.0 <= r["rate_wpm"] <= 200.0)
    )
    results["verdicts"]["teacher_cfg"] = (
        "GUIDANCE LOAD-BEARING — teacher degrades at cfg 1.0; raises the ceiling CFG repair can reach"
        if degraded
        else "guidance is polish for the teacher — fine at cfg 1.0"
    )

fd = report.get("fd_curves", {})
if fd:
    results["fd_summary"] = {
        k: {
            "first": round(v[0], 1),
            "median": round(sorted(v)[len(v) // 2], 1),
            "last": round(v[-1], 1),
        }
        for k, v in fd.items()
    }

dest = Path(__file__).parent / "gate_night8_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print("\nDELTAS (cfg minus control):", json.dumps(results["deltas"], indent=2))
print("\nVERDICTS:", json.dumps(results["verdicts"], indent=2))
print(f"wrote {dest}")
print("\nListening (never skipped): the A/B that matters is c8_euler4_s0 vs c8_euler4_plain_s0.")
