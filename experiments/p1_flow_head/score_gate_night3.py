"""Mac-side scoring for Gate Night 3.

Usage, from repo root:
    unzip -o ~/Downloads/gate_night3_bundle.zip -d experiments/p1_flow_head/audio/gate_night3
    .venv/bin/python experiments/p1_flow_head/score_gate_night3.py

Verdicts against the pre-registered criteria in NOTES.md (Gate Night 3 entry).
GN2 baselines (same stack, seeded): natural 165-169 wpm; monolithic-full
230.7 (p0) / 233.3 (p1); 119w 199.5 (p0) / 166.7 (p1); 1500w 241.3 (p0).
"""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval.metrics import _ecapa, _whisper, load_16k_mono  # noqa: E402

AUD = Path(__file__).parent / "audio" / "gate_night3"
with open(AUD / "gate_night3_report.json") as f:
    report = json.load(f)

GN2 = {"natural": 167.0, "p0_full": 230.7, "p1_full": 233.3, "p0_119": 199.5, "p0_1500": 241.3}


def wpm(path):
    segs, _ = _whisper().transcribe(str(path), language="en", word_timestamps=False, beam_size=1)
    n = sum(len((s.text or "").split()) for s in segs)
    dur = sf.info(path).duration
    return round(n / (dur / 60), 1), n


def clip_sim(path_a, path_b):
    e = _ecapa()
    embs = [e.encode_batch(load_16k_mono(p)[None])[0, 0].detach() for p in (path_a, path_b)]
    return round(float(torch.nn.functional.cosine_similarity(embs[0], embs[1], dim=-1)), 3)


results = {"runs": {}, "verdicts": {}}
for row in report["runs"]:
    r, n = wpm(AUD / f"{row['tag']}.wav")
    results["runs"][row["tag"]] = {"wpm": r, "words": n, "audio_s": row["audio_s"]}
    print(f"  {row['tag']:16s} {r:6.1f} wpm  ({n} words / {row['audio_s']}s)", flush=True)

runs = results["runs"]

# T1 — turn-split (CURED <=185 both prompts; PARTIAL 185-215; FAIL >215)
t1 = [runs[t]["wpm"] for t in ("t1_turnsplit_p0", "t1_turnsplit_p1") if t in runs]
if t1:
    worst = max(t1)
    results["verdicts"]["T1"] = (
        "CURED — pace is per-turn scoped"
        if worst <= 185
        else "PARTIAL"
        if worst <= 215
        else "FAIL — official remedy does not work either"
    )

# T2 — stretched prompt (TRANSFERS if monotone 0.8<0.9 and stretch80 <= 208; identity delta Mac-judged)
if "t2_stretch80" in runs and "t2_stretch90" in runs:
    w80, w90 = runs["t2_stretch80"]["wpm"], runs["t2_stretch90"]["wpm"]
    monotone = w80 < w90 < GN2["p0_full"]
    results["verdicts"]["T2_pace"] = (
        "TRANSFERS" if monotone and w80 <= 208 else "WEAK/PARTIAL" if monotone else "NO TRANSFER"
    )
    orig_prompt = sorted((AUD.parent / "gate_night2").glob("sweep_p0_full.wav"))
    for tag in ("t2_stretch80", "t2_stretch90"):
        runs[tag]["sim_vs_gn2_full"] = (
            clip_sim(AUD / f"{tag}.wav", orig_prompt[0]) if orig_prompt else None
        )

# T3 — 7B (PRESENT if 1500/119 ratio >= 1.2; ABSENT <= 1.05)
if "t3_7b_119" in runs and "t3_7b_1500" in runs:
    ratio = runs["t3_7b_1500"]["wpm"] / runs["t3_7b_119"]["wpm"]
    results["verdicts"]["T3_ratio"] = round(ratio, 3)
    results["verdicts"]["T3"] = (
        "DEFECT PRESENT AT 7B"
        if ratio >= 1.2
        else "DEFECT ABSENT AT 7B"
        if ratio <= 1.05
        else "AMBIGUOUS"
    )

dest = Path(__file__).parent / "gate_night3_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print("\nVERDICTS:", json.dumps(results["verdicts"], indent=2))
print(f"wrote {dest}")
print("\nListening (never skipped): t1_turnsplit_p0.wav — does turn-split sound natural,")
print("any per-turn seam artifacts?  t2_stretch80.wav — does the voice still sound like p0?")
