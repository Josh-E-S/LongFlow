# Capture v2 truncation audit — 2026-08-17 (no GPU; Drive file-size forensics)

Executes step 1 of the 2026-08-17 session close-out ("Audit, don't guess").
Method: listed all 248 `cv2_*.pt` files in Drive `longflow_p1_cache_v2` with
exact byte sizes + creation timestamps (Drive API — no Colab session needed).
Bytes/frame is constant across the v2 schema (calibrated at 6,269 B/frame from
the close-out's known-truncated exemplar `cv2_1200w_4f19765a` = 7,475,787 B =
159 s). Implied duration = size / 6269 / 7.5 Hz; implied wpm = filename-bin
words / duration. Turn-split natural rate is 150–200 wpm (GN3; noised windows
slow to ~140), so implied wpm > 210 means the file physically cannot contain
its full script → TRUNCATED.

## Verdict: 25 TRUNCATED + 2 SUSPECT out of 248; damage is bounded and filterable

- **25 truncated files, 24 of them sitting within 1% of exactly two size
  values: ~7.48 MB (= 1,193 frames ≈ 159 s) and ~14.97 MB (= 2,387 frames ≈
  318 s).** That is the token-budget-cap signature predicted by the close-out's
  mechanism (`max_new_tokens` sized from whichever bin triggered the flush):
  the 7.48 MB cluster is the 300w-bin budget, the 14.97 MB cluster the
  600w-bin budget. Independent confirmation: every one of these files was
  created in a *mixed-bin* batch (different filename bins sharing one
  timestamp), which the fixed code cannot produce.
- **Timing: all 25 truncated files were created 10:08–15:44 on 2026-08-16 —
  the first session.** Every batch from 17:00 onward is single-bin with
  healthy sizes. Note this means the mid-session mixed-bin "fix" did NOT take
  effect during the first session (stale notebook/kernel); the entire pre-17:00
  window (91 files) ran buggy code — but only batches whose flush was
  triggered by a smaller bin actually truncated. 26 of the 91 pre-fix files
  are flagged; the other 65 pre-fix files pass the same wpm test as post-fix
  files and are presumed sound.
- **Clean-file sanity check (validates the method):** implied wpm across the
  221 clean files is exactly the turn-split natural band — per-bin medians
  150 / 157 / 164 / 166 / 168, max 179–192. No clean file exceeds 192.
- **The 2 SUSPECT files** (`cv2_150w_966bfa0b` 203 wpm on a 44 s clip;
  `cv2_1200w_85330b6d` 195 wpm, post-fix): marginal, not cap-cluster members.
  Cheapest handling: exclude both alongside the 25.

Full flagged list: `capture_v2_audit_flags.json` (same directory).

## Per-bin damage

| bin | total | flagged | remaining |
|---|---|---|---|
| 150w | 39 | 1 (suspect) | 38 |
| 300w | 77 | 0 | 77 |
| 600w | 47 | 5 | 42 |
| 1200w | 47 | 9 (8 trunc + 1 suspect) | 38 |
| 2400w | 38 | 12 | 26 |

Frames retained after excluding all 27 flagged files: ~470K of 510K (~92%) —
still at the ~480K design target. **No wholesale re-capture is needed.** The
one real loss is 2400w-bin coverage (38 → 26 scripts), the bin that carries
the long-context story; a targeted top-up (~12 × 2400w + ~8 × 1200w scripts
with the current fixed notebook, ~3–4 A100-h) restores the designed spread.

## What the truncated files poison, and what they don't

- **Poisoned:** any use of `meta["target_words"]` (the corrupted flush-bin
  label — this is what skewed the 20K run's held-out stratification), and any
  text-vs-audio eval on these files (stored script ≠ rendered content — this
  is the bimodal WER 0.7–0.8 population in the step-20000 results).
- **Mostly NOT poisoned:** the per-frame (hidden, latent) training pairs
  inside a truncated file are individually valid (the teacher was cut off,
  not corrupted). The v2 20K checkpoint's *training* is therefore probably
  not badly damaged — its *evaluation* was. Re-scoring on a clean held-out
  split is required before judging the checkpoint; retraining on the filtered
  pool is cheap and preferable for a result that will be cited.

## Actions

1. Exclude the 27 flagged files from every training pool and held-out split
   (load `capture_v2_audit_flags.json`, filter by filename).
2. Stratify held-out by **filename** bin, never `meta["target_words"]`, until
   metas are rewritten from filenames.
3. Optional top-up capture for the 1200w/2400w bins (~3–4 A100-h).
4. Re-score the existing `v2_20k` checkpoints on a clean 25-script held-out
   before drawing any conclusion from that run.
