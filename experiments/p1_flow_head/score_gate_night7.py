"""Mac-side scoring for Gate Night 7 (capture prerequisites).

Usage, from repo root:
    unzip -o ~/Downloads/gate_night7_bundle.zip -d experiments/p1_flow_head/audio/gate_night7
    .venv/bin/python experiments/p1_flow_head/score_gate_night7.py

Arm T (teacher under feedback noise): WER vs script, voice fraction and
GRADED similarity vs the GN3 teacher reference, speaking rate — gate:
capture greenlit / partial / dead. Arm A (sigma=0.4 reroll): coverage per
seed — variance vs dead zone. Criteria: NOTES, Gate Night 7 entry. Josh's
ear is the final gate on Arm T target quality.
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

AUD = Path(__file__).parent / "audio" / "gate_night7"
TEACHER = Path(__file__).parent / "audio" / "gate_night3" / "t1_turnsplit_p0.wav"
with open(AUD / "gate_night7_report.json") as f:
    report = json.load(f)

T_TAGS = ["t7_sig000", "t7_sig020", "t7_sig030"]
A_TAGS = ["a7_sig040_s1", "a7_sig040_s2"]
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
    hyp = n(transcribe(path))
    n_words = len(hyp.split())
    return {
        "duration_s": round(dur, 1),
        "wer_vs_script": round(jiwer.wer(SCRIPT, hyp) if hyp else 1.0, 3),
        "coverage_pct": round(100 * min(n_words, N_SCRIPT) / N_SCRIPT, 1),
        "voice_pct": round(100 * float((sim >= 0.5).mean()), 1),
        "sim_median": round(float(np.median(sim)), 3),
        "sim_final_third": round(float(np.median(sim[-max(1, len(sim) // 3) :])), 3),
        "rate_wpm": round(60 * n_words / dur, 1),
    }


results = {"armT": {}, "armA": {}, "verdicts": {}, "gate_audit": {}}
ref_E, _, _ = win_embs(TEACHER)
ref = ref_E.median(0).values

# hook-gate audit: prompt call must have been skipped in every teacher run
for run in report["runs"]:
    if run["tag"].startswith("t7_"):
        results["gate_audit"][run["tag"]] = {
            "hook_shapes": run.get("hook_shapes"),
            "noised_calls": run.get("noised_calls"),
            "skipped_calls": run.get("skipped_calls"),
        }
        if run.get("skipped_calls", 0) < 1:
            results["verdicts"]["gate_audit"] = (
                f"HOOK GATE SUSPECT on {run['tag']} — no multi-frame call skipped; "
                "the prompt may have been noised (GN5 v1 failure mode); distrust Arm T"
            )

for tag in T_TAGS + A_TAGS:
    p = AUD / f"{tag}.wav"
    if not p.exists():
        continue
    r = score(p)
    (results["armT"] if tag in T_TAGS else results["armA"])[tag] = r
    print(f"{tag}: {r}", flush=True)

t = results["armT"]

# ---- Arm T verdict (pre-registered) ----
if "t7_sig000" in t:
    base = t["t7_sig000"]

    def passes(r):
        return (
            r["wer_vs_script"] <= base["wer_vs_script"] + 0.06
            and r["voice_pct"] >= 90.0
            and 150.0 <= r["rate_wpm"] <= 200.0
        )

    ok = [tag for tag in ("t7_sig020", "t7_sig030") if tag in t and passes(t[tag])]
    if len(ok) == 2:
        results["verdicts"]["armT"] = (
            "CAPTURE GREENLIT (metric half) at sigma<=0.3 — Josh's ear is the final gate; "
            "if he passes it, capture v2 proceeds"
        )
    elif ok:
        results["verdicts"]["armT"] = (
            f"PARTIAL — only {ok} passes; capture v2 caps sigma at that level (ear gate pending)"
        )
    else:
        results["verdicts"]["armT"] = (
            "CAPTURE DESIGN DEAD — teacher degrades under feedback noise at both sigmas; "
            "rethink targets (lower sigma, or clean-teacher targets on noised hidden states)"
        )

# ---- Arm A verdict (pre-registered) ----
a = results["armA"]
if a:
    survivors = [tag for tag, r in a.items() if r["coverage_pct"] >= 90.0]
    results["verdicts"]["armA"] = (
        f"VARIANCE (bistability) — {survivors} survive; GN6 sigma=0.4 was an unlucky basin; "
        "random-sigma training design stands"
        if survivors
        else "DEAD ZONE — both seeds die at sigma=0.4; training range excludes ~0.4; paper figure flag"
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

dest = Path(__file__).parent / "gate_night7_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print("\nVERDICTS:", json.dumps(results["verdicts"], indent=2))
print(f"wrote {dest}")
print("\nListening (never skipped): all five; the two noised-teacher renders matter most —")
print("are they target-quality to Josh's ear?")
