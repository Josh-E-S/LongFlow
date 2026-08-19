"""Frame-level forensics of the chunked engine A/B renders.
Per-window acoustic metrics, within-chunk decay curves, golden-window gap."""

import json

import matplotlib
import numpy as np
import parselmouth
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DL = "/Users/joshest/Downloads"
OUT = "/private/tmp/claude-501/-Users-joshest-Desktop-TTS-Projects-LongFlow/7123b45b-11ef-4439-a8ab-f76cadd956a2/scratchpad"

# chunk durations from the run log; crossfade 0.25s between chunks
FILES = {
    "teacher": (f"{DL}/ce_teacher.wav", [85.3, 107.2, 69.1, 52.0]),
    "july": (f"{DL}/ce_july.wav", [72.9, 79.5, 59.1, 48.0]),
    "july_p2": (f"{DL}/ce_july_p2.wav", [81.1, 76.5, 68.4, 51.9]),
    "cleanabl_p2": (f"{DL}/ce_cleanabl_p2.wav", [79.1, 88.3, 60.7, 47.6]),
    "cleanabl": (f"{DL}/ce_cleanabl.wav", [70.5, 87.9, 65.7, 43.2]),
}
FADE = 0.25
WIN, HOP = 1.0, 0.5


def chunk_starts(durs):
    starts = [0.0]
    for d in durs[:-1]:
        starts.append(starts[-1] + d - FADE)
    return starts


def analyze(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    harm = snd.to_harmonicity_cc(time_step=0.05)
    pitch = snd.to_pitch(time_step=0.05)
    hnr_t = harm.xs()
    hnr_v = harm.values[0]
    f0_t = pitch.xs()
    f0_v = pitch.selected_array["frequency"]

    n_win = int((dur - WIN) / HOP) + 1
    rows = []
    freqs = np.fft.rfftfreq(int(WIN * sr), 1 / sr)
    hann = np.hanning(int(WIN * sr))
    for i in range(n_win):
        t0 = i * HOP
        seg = x[int(t0 * sr) : int((t0 + WIN) * sr)]
        if len(seg) < int(WIN * sr):
            break
        P = np.abs(np.fft.rfft(seg * hann)) ** 2 + 1e-12
        rms = 20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-9)
        centroid = float((freqs * P).sum() / P.sum())
        # flatness over speech band 0.2-8k
        band = (freqs >= 200) & (freqs <= 8000)
        flat = float(np.exp(np.mean(np.log(P[band]))) / np.mean(P[band]))
        e_low = P[(freqs >= 200) & (freqs < 4000)].sum()
        e_hf = P[(freqs >= 4000) & (freqs < 10000)].sum()
        hf_ratio = float(e_hf / (e_low + 1e-9))
        m = (hnr_t >= t0) & (hnr_t < t0 + WIN)
        hv = hnr_v[m]
        hv = hv[hv > -50]
        hnr = float(np.mean(hv)) if len(hv) else np.nan
        mp = (f0_t >= t0) & (f0_t < t0 + WIN)
        fv = f0_v[mp]
        voiced = float((fv > 0).mean()) if len(fv) else 0.0
        f0s = float(np.std(fv[fv > 0])) if (fv > 0).sum() > 3 else np.nan
        # envelope peak rate (syllable-ish proxy)
        env = np.abs(seg)
        k = int(0.02 * sr)
        env = np.convolve(env, np.ones(k) / k, mode="same")
        thr = env.mean() * 1.2
        peaks = ((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > thr)).sum()
        rows.append((t0 + WIN / 2, rms, centroid, flat, hf_ratio, hnr, voiced, f0s, peaks))
    return np.array(rows), dur


results = {}
for name, (path, durs) in FILES.items():
    arr, dur = analyze(path)
    results[name] = {"arr": arr, "starts": chunk_starts(durs), "durs": durs, "dur": dur}
    print(f"{name}: {dur:.1f}s, {len(arr)} windows")

COLS = {
    "rms": 1,
    "centroid": 2,
    "flatness": 3,
    "hf_ratio": 4,
    "hnr": 5,
    "voiced": 6,
    "f0_std": 7,
    "peaks": 8,
}


def within_chunk(name, col, tmax=80, bin_s=5):
    r = results[name]
    arr = r["arr"]
    bins = np.arange(0, tmax + bin_s, bin_s)
    vals = [[] for _ in range(len(bins) - 1)]
    for ci, s in enumerate(r["starts"]):
        end = s + r["durs"][ci]
        m = (arr[:, 0] >= s) & (arr[:, 0] < end)
        trel = arr[m, 0] - s
        v = arr[m, col]
        for j in range(len(bins) - 1):
            mm = (trel >= bins[j]) & (trel < bins[j + 1]) & np.isfinite(v)
            vals[j].extend(v[mm].tolist())
    return bins[:-1] + bin_s / 2, np.array([np.median(x) if x else np.nan for x in vals])


# ---------- Figure 1: within-chunk decay curves ----------
metrics_plot = [
    ("hnr", "HNR dB (voice cleanliness)"),
    ("flatness", "Spectral flatness (noisiness)"),
    ("hf_ratio", "HF/LF energy ratio"),
    ("peaks", "Envelope peaks/s (rate proxy)"),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
colors = {
    "teacher": "k",
    "july": "tab:red",
    "july_p2": "tab:orange",
    "cleanabl_p2": "tab:blue",
    "cleanabl": "tab:cyan",
}
for ax, (mk, label) in zip(axes.flat, metrics_plot, strict=False):
    for name in FILES:
        t, v = within_chunk(name, COLS[mk])
        ax.plot(
            t,
            v,
            color=colors[name],
            label=name,
            lw=2 if name in ("teacher", "july") else 1.2,
            alpha=1.0 if name in ("teacher", "july") else 0.65,
        )
    ax.set_title(label)
    ax.set_xlabel("seconds since chunk start")
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Within-chunk decay (median across the 4 chunks)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_decay.png", dpi=110)

# ---------- Figure 2: full timelines, HNR + flatness, teacher vs july ----------
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
for ax, mk, label in [(axes[0], "hnr", "HNR dB"), (axes[1], "flatness", "Spectral flatness")]:
    for name in ("teacher", "july"):
        arr = results[name]["arr"]
        ax.plot(arr[:, 0], arr[:, COLS[mk]], color=colors[name], label=name, lw=1)
        for s in results[name]["starts"][1:]:
            ax.axvline(s, color=colors[name], ls=":", alpha=0.5)
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[1].set_xlabel("seconds (dotted lines = chunk stitches)")
fig.suptitle("Full timelines — teacher vs july (each on its own clock)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_timeline.png", dpi=110)

# ---------- Tables ----------
print("\n=== GOLDEN WINDOW (first 15s of each chunk) vs LATE (30s+) — medians ===")
hdr = f"{'engine':12} {'HNR_early':>9} {'HNR_late':>8} {'flat_early':>10} {'flat_late':>9} {'HF_early':>8} {'HF_late':>7} {'rate_early':>10} {'rate_late':>9}"
print(hdr)
summary = {}
for name in FILES:
    r = results[name]
    arr = r["arr"]
    early_m = np.zeros(len(arr), bool)
    late_m = np.zeros(len(arr), bool)
    for ci, s in enumerate(r["starts"]):
        end = s + r["durs"][ci]
        trel = arr[:, 0] - s
        inchunk = (arr[:, 0] >= s) & (arr[:, 0] < end)
        early_m |= inchunk & (trel < 15)
        late_m |= inchunk & (trel >= 30)

    def med(col, m, arr=arr):
        v = arr[m, col]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    row = {
        k: (med(COLS[k], early_m), med(COLS[k], late_m))
        for k in ("hnr", "flatness", "hf_ratio", "peaks")
    }
    summary[name] = row
    print(
        f"{name:12} {row['hnr'][0]:9.2f} {row['hnr'][1]:8.2f} {row['flatness'][0]:10.4f} "
        f"{row['flatness'][1]:9.4f} {row['hf_ratio'][0]:8.3f} {row['hf_ratio'][1]:7.3f} "
        f"{row['peaks'][0]:10.1f} {row['peaks'][1]:9.1f}"
    )

print("\n=== JULY vs TEACHER deltas ===")
for k in ("hnr", "flatness", "hf_ratio", "peaks"):
    te, tl = summary["teacher"][k]
    je, jl = summary["july"][k]
    print(
        f"{k:9}: golden-window gap (july-teacher early): {je-te:+.3f}   "
        f"july within-chunk drift (late-early): {jl-je:+.3f}   teacher drift: {tl-te:+.3f}"
    )

with open(f"{OUT}/forensics_summary.json", "w") as f:
    json.dump(
        {k: {m: [float(a), float(b)] for m, (a, b) in v.items()} for k, v in summary.items()},
        f,
        indent=1,
    )
print("\nfigures: fig1_decay.png, fig2_timeline.png")
