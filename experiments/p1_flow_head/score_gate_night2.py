"""Mac-side scoring for Gate Night 2 (run after extracting the bundle).

Usage, from repo root:
    unzip -o ~/Downloads/gate_night2_bundle.zip -d experiments/p1_flow_head/audio/gate_night2
    .venv/bin/python experiments/p1_flow_head/score_gate_night2.py

Scores H1 (rate sweep) and H2 (chunked reveal) against the pre-registered
criteria in NOTES.md (Gate Night 2 entry) and writes gate_night2_metrics.json.

H1 primary metric: wpm over the SHARED first ~100 words (identical text in all
four conditions), timed from first-word start to Nth-word end via Whisper word
timestamps. Whole-clip wpm is secondary.
H2: whole-run wpm of the chunked render vs the standalone 500-word condition
and vs the full-script condition; seam ECAPA (3 s window either side of each
chunk boundary) vs within-chunk adjacent-window baseline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval.metrics import _ecapa, _whisper  # noqa: E402

AUD = Path(__file__).parent / "audio" / "gate_night2"
with open(AUD / "gate_night2_report.json") as f:
    report = json.load(f)
N_SHARED = report["shared_prefix_words"]


def words_of(path):
    segs, _ = _whisper().transcribe(str(path), language="en", word_timestamps=True, beam_size=1)
    out = []
    for s in segs:
        out.extend({"w": w.word.strip(), "t": w.start, "e": w.end} for w in (s.words or []))
    return out


def wpm_stats(path):
    ws = words_of(path)
    dur = sf.info(path).duration
    n = len(ws)
    whole = n / (dur / 60) if dur else 0.0
    shared = None
    if n >= N_SHARED:
        span = ws[N_SHARED - 1]["e"] - ws[0]["t"]
        shared = N_SHARED / (span / 60) if span > 0 else None
    return {
        "words": n,
        "audio_s": round(dur, 1),
        "wpm": round(whole, 1),
        "shared_seg_wpm": round(shared, 1) if shared else None,
    }


results = {"sweep": [], "chunked": {}, "verdicts": {}}

# ---- H1: sweep ----
print("H1 — rate sweep (shared-segment wpm is the pre-registered metric)")
for row in report["sweep"]:
    m = wpm_stats(AUD / f"sweep_{row['tag']}.wav")
    m.update(
        tag=row["tag"],
        condition=row["condition"],
        prompt=row["prompt"],
        script_words=row["script_words"],
        near_cap=row.get("near_cap"),
    )
    results["sweep"].append(m)
    print(
        f"  {row['tag']:10s} {row['script_words']:5d}w  whole {m['wpm']:6.1f}  "
        f"shared {m['shared_seg_wpm'] or float('nan'):6.1f} wpm",
        flush=True,
    )

by = {(m["prompt"], m["condition"]): m for m in results["sweep"]}
prompts = sorted({m["prompt"] for m in results["sweep"]})
ratios = []
for p in prompts:
    a, b = by.get((p, "100")), by.get((p, "full"))
    if a and b and a["shared_seg_wpm"] and b["shared_seg_wpm"]:
        ratios.append(b["shared_seg_wpm"] / a["shared_seg_wpm"])
results["verdicts"]["H1_ratios_full_over_100"] = [round(r, 3) for r in ratios]
if ratios and all(r >= 1.2 for r in ratios):
    h1 = "CONFIRMED"
elif ratios and all(r <= 1.05 for r in ratios):
    h1 = "REFUTED"
else:
    h1 = "AMBIGUOUS (add prompts)"
results["verdicts"]["H1"] = h1

# ---- H2: chunked ----
print("\nH2 — chunked reveal")
full_m = wpm_stats(AUD / "chunked_full.wav")
results["chunked"]["overall"] = full_m
ref500 = by.get((prompts[0], "500"), {}).get("wpm")
reffull = by.get((prompts[0], "full"), {}).get("wpm")
results["chunked"]["ref_500_wpm"] = ref500
results["chunked"]["ref_full_wpm"] = reffull
print(f"  chunked {full_m['wpm']} wpm  vs standalone-500 {ref500}  vs full {reffull}")

# seam ECAPA: boundary windows vs within-chunk adjacent-window baseline
ecapa = _ecapa()
import torchaudio.functional as taf  # noqa: E402


def emb(x24, sr=24000):
    x16 = taf.resample(torch.from_numpy(x24.astype(np.float32)), sr, 16000)
    return ecapa.encode_batch(x16[None])[0, 0].detach()


W = 3 * 24000
chunk_wavs = sorted(AUD.glob("chunk_*.wav"))
seams, baseline = [], []
for i in range(len(chunk_wavs) - 1):
    a, _ = sf.read(chunk_wavs[i], dtype="float32")
    b, _ = sf.read(chunk_wavs[i + 1], dtype="float32")
    if len(a) < 2 * W or len(b) < W:
        continue
    seams.append(float(torch.nn.functional.cosine_similarity(emb(a[-W:]), emb(b[:W]), dim=-1)))
    baseline.append(
        float(torch.nn.functional.cosine_similarity(emb(a[-2 * W : -W]), emb(a[-W:]), dim=-1))
    )
results["chunked"]["seam_ecapa_mean"] = round(float(np.mean(seams)), 3) if seams else None
results["chunked"]["within_chunk_baseline"] = (
    round(float(np.mean(baseline)), 3) if baseline else None
)

pace_ok = ref500 and full_m["wpm"] <= 1.10 * ref500 and reffull and full_m["wpm"] <= 0.85 * reffull
seam_ok = seams and baseline and np.mean(seams) >= np.mean(baseline) - 0.10
results["verdicts"]["H2_pacing"] = "CURED" if pace_ok else "NOT CURED"
results["verdicts"]["H2_seams"] = (
    ("ACCEPTABLE" if seam_ok else "NEEDS KV-CACHE VERSION") if seams else "N/A"
)

dest = Path(__file__).parent / "gate_night2_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print(
    f"\nVERDICTS  H1: {h1}   H2 pacing: {results['verdicts']['H2_pacing']}   "
    f"H2 seams: {results['verdicts']['H2_seams']}"
)
print(
    f"seam ECAPA {results['chunked']['seam_ecapa_mean']} vs within-chunk "
    f"{results['chunked']['within_chunk_baseline']}"
)
print(f"wrote {dest}")
print("\nListening step is NOT automated: chunked_full.wav (seams) + one full sweep clip.")
