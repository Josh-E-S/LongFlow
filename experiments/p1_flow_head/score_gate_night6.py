"""Mac-side scoring for Gate Night 6 (the sigma sweep).

Usage, from repo root:
    unzip -o ~/Downloads/gate_night6_bundle.zip -d experiments/p1_flow_head/audio/gate_night6
    .venv/bin/python experiments/p1_flow_head/score_gate_night6.py

Per condition: WER vs the sweep script (primary content metric; script text
comes from the Colab report), Whisper word coverage, per-window ECAPA vs the
GN3 teacher reference (voice fraction, horizon), FD summary echoed from the
report. Verdict table pre-registered in NOTES, Gate Night 6 entry. Ears
remain the authority on the listenability half of SWEET SPOT.
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

AUD = Path(__file__).parent / "audio" / "gate_night6"
TEACHER = Path(__file__).parent / "audio" / "gate_night3" / "t1_turnsplit_p0.wav"
with open(AUD / "gate_night6_report.json") as f:
    report = json.load(f)

CONDITIONS = [
    "g6_sig000",
    "g6_sig010",
    "g6_sig020",
    "g6_sig030",
    "g6_sig040",
    "g6_sig050",
    "g6_ramp050",
]
WIN, HOP = 4.0, 2.0
ecapa = _ecapa()
n = _normalizer()

# strip speaker tags from the script; WER target is the spoken words
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


results = {"conditions": {}, "verdicts": {}}
ref_E, _, _ = win_embs(TEACHER)
ref = ref_E.median(0).values

for tag in CONDITIONS:
    p = AUD / f"{tag}.wav"
    if not p.exists():
        continue
    E, ts, dur = win_embs(p)
    sim = torch.nn.functional.cosine_similarity(E, ref[None], dim=-1).numpy()
    voice = sim >= 0.5
    horizon = 0.0
    for i in range(len(ts)):
        if voice[i]:
            horizon = ts[i] + WIN
    third = max(1, len(sim) // 3)
    hyp = n(transcribe(p))
    wer = jiwer.wer(SCRIPT, hyp) if hyp else 1.0
    results["conditions"][tag] = {
        "duration_s": round(dur, 1),
        "wer_vs_script": round(wer, 3),
        "coverage_pct": round(100 * min(len(hyp.split()), N_SCRIPT) / N_SCRIPT, 1),
        "voice_pct": round(100 * float(voice.mean()), 1),
        "horizon_s": horizon,
        "final_third_sim": round(float(np.median(sim[-third:])), 3),
    }
    print(f"{tag}: {results['conditions'][tag]}", flush=True)

c = results["conditions"]

# ---- pre-registered verdict table (NOTES, GN6 entry) ----
sweet = [
    t
    for t, r in c.items()
    if t != "g6_sig000" and r["wer_vs_script"] <= 0.15 and r["voice_pct"] >= 90.0
]
content_saved = [t for t, r in c.items() if t != "g6_sig000" and r["coverage_pct"] >= 90.0]
low_dead = [
    t for t in ("g6_sig010", "g6_sig020", "g6_sig030") if t in c and c[t]["horizon_s"] < 60.0
]
if sweet:
    results["verdicts"]["sweep"] = (
        f"SWEET SPOT (metric half) at {sweet} — Josh's ear decides the other half; "
        "if he passes it, this is the head's default inference config"
    )
elif content_saved:
    if len(low_dead) == 3 and all(
        t in ("g6_sig040", "g6_sig050", "g6_ramp050") for t in content_saved
    ):
        results["verdicts"]["sweep"] = (
            f"CLIFF — sigma<=0.3 die base-like, only {content_saved} save content; "
            "decorrelation threshold ~= head error scale; training samples sigma across it"
        )
    else:
        results["verdicts"]["sweep"] = (
            f"TRADEOFF CURVE ONLY — content saved at {content_saved} but identity never passes; "
            "stage-2 training carries the full burden"
        )
else:
    results["verdicts"]["sweep"] = (
        "NO CONTENT RESCUE AT ANY SIGMA — inconsistent with GN5; investigate before concluding"
    )

# replication gate: g6_sig050 vs GN5 f2_abl_noise (WER 0.296 vs teacher
# rendition, full-length survival). Different WER target text (script vs
# teacher rendition) means compare loosely: full-length + coverage>=85%.
if "g6_sig050" in c:
    r = c["g6_sig050"]
    ok = r["coverage_pct"] >= 85.0 and r["duration_s"] >= 0.9 * c.get("g6_sig000", r)["duration_s"]
    results["verdicts"]["replication"] = (
        "g6_sig050 reproduces GN5 f2_abl_noise (full-length, content survives)"
        if ok
        else "REPLICATION FLAG — g6_sig050 does not match GN5 f2_abl_noise; distrust the night"
    )

# H3 read: ramp vs constant 0.5
if "g6_ramp050" in c and "g6_sig050" in c:
    ramp, const = c["g6_ramp050"], c["g6_sig050"]
    if ramp["coverage_pct"] >= const["coverage_pct"] - 5.0:
        better = ramp["voice_pct"] > const["voice_pct"] + 5.0
        results["verdicts"]["h3_schedule"] = (
            "SCHEDULE WINS — ramp keeps more voice at equal coverage"
            if better
            else "schedule neutral — constant sigma is fine"
        )
    else:
        results["verdicts"]["h3_schedule"] = (
            "EARLY NOISE IS LOAD-BEARING — ramp loses content; compounding seeds before 60 s"
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

dest = Path(__file__).parent / "gate_night6_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print("\nVERDICTS:", json.dumps(results["verdicts"], indent=2))
print(f"wrote {dest}")
print(
    "\nListening (never skipped): all seven renders, in sigma order; verdicts verbatim into NOTES."
)
