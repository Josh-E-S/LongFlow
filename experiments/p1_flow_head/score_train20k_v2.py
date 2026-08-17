"""Mac-side scoring for the capture v2 20K-step training run.

Usage, from repo root:
    unzip -o ~/Downloads/train20k_v2_eval.zip -d experiments/p1_flow_head/audio/train20k_v2
    .venv/bin/python experiments/p1_flow_head/score_train20k_v2.py

Held-out WER + speaker-sim (vs the teacher's own decode of the same
utterance) per checkpoint step, stratified by word bin. The point: verify
empirically whether 20K is the sweet spot on capture v2's cache (July's
E3 finding: 20K was optimal, 80K overfit, on a similarly-sized cache) --
if held-out WER/sim improves through 15K then gets WORSE at 20K, that's
the same overfitting signature and the 15K checkpoint should be the
operating one instead.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eval.metrics import clip_metrics  # noqa: E402

AUD = Path(__file__).parent / "audio" / "train20k_v2"
with open(AUD / "manifest.json") as f:
    manifest = json.load(f)

STEPS = sorted(int(s) for s in manifest["checkpoints"])
results = {"held_out_per_bin": manifest["held_out_per_bin"], "steps": {}}

for step in STEPS:
    entries = manifest["checkpoints"][str(step)]
    print(f"\n=== step {step} ({len(entries)} utterances) ===")
    rows = []
    for e in entries:
        m = clip_metrics(AUD / e["audio"], e["text"], AUD / e["teacher_audio"])
        rows.append({**m, "utt_id": e["utt_id"], "target_words": e["target_words"]})
        print(
            f"  {e['utt_id']} (w={e['target_words']:>4}): "
            f"wer={m['wer']:.3f}  sim={m['speaker_sim']:.3f}  "
            f"voiced={m['voiced_fraction']:.2f}  dur={m['duration_s']:.0f}s"
        )

    by_bin = {}
    for r in rows:
        by_bin.setdefault(r["target_words"], []).append(r)
    bin_summary = {
        tw: {
            "n": len(rs),
            "wer_median": sorted(r["wer"] for r in rs)[len(rs) // 2],
            "sim_median": sorted(r["speaker_sim"] for r in rs)[len(rs) // 2],
        }
        for tw, rs in sorted(by_bin.items())
    }
    overall = {
        "n": len(rows),
        "wer_median": sorted(r["wer"] for r in rows)[len(rows) // 2],
        "sim_median": sorted(r["speaker_sim"] for r in rows)[len(rows) // 2],
    }
    print(
        f"  overall: wer_median={overall['wer_median']:.3f}  sim_median={overall['sim_median']:.3f}"
    )
    results["steps"][step] = {"overall": overall, "by_bin": bin_summary, "rows": rows}

print("\n" + "=" * 60)
print("SUMMARY (overall median WER / speaker-sim per checkpoint)")
print("=" * 60)
for step in STEPS:
    o = results["steps"][step]["overall"]
    print(f"  step {step:>6}: n={o['n']:>3}  wer={o['wer_median']:.3f}  sim={o['sim_median']:.3f}")

# overfitting check: does WER/sim get WORSE at the final step vs an earlier one?
# (July's E3 signature: 20K optimal, 80K overfit, on a similarly-sized cache)
if len(STEPS) >= 2:
    best_step = min(STEPS, key=lambda s: results["steps"][s]["overall"]["wer_median"])
    last_step = STEPS[-1]
    if best_step != last_step:
        print(
            f"\nWATCH: best held-out WER is at step {best_step}, not the final "
            f"step {last_step} -- possible overfitting past {best_step}, matching "
            "the July E3 signature (20K optimal / 80K overfit). Consider "
            f"{best_step} as the operating checkpoint instead of {last_step}."
        )
    else:
        print(f"\nFinal step {last_step} has the best held-out WER -- no overfitting signature.")

dest = Path(__file__).parent / "train20k_v2_metrics.json"
with open(dest, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {dest}")
print("\nListening (never skipped): spot-check a short and a long utterance at the")
print("operating checkpoint against its teacher reference before calling this done.")
