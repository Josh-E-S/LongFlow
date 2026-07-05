# P0 — Steering Gate Check (C2 go/no-go)

**Question:** Can activation steering vectors, extracted from VibeVoice's own contrast-pair generations and injected into the frozen Qwen2.5 backbone, audibly shift the emotional character of generated speech — without degrading intelligibility or speaker identity?

**Why first:** C2 is the highest-risk, highest-novelty contribution. This experiment needs no training, runs on a single RTX 3090, and delivers a go/no-go in ~1 day of work. If steering doesn't land, the paper reshapes around C1+C3+C4 before any flow-head compute is spent.

**Budget:** ≤ $15 Vast.ai. **Hardware:** 1× RTX 3090 24GB (VibeVoice-1.5B in bf16 fits comfortably).

---

## Stage 0 — Environment + baseline sanity (½ day)

1. Spin up Vast.ai 3090, install the VibeVoice community fork, download VibeVoice-1.5B.
2. Generate one short single-speaker sample and one 2-speaker dialogue sample from stock VibeVoice. Listen. This is the "nothing is broken" baseline and your first look at the codebase's inference path.
3. Locate in the code: (a) the Qwen2.5 decoder layer modules (for hooks), (b) where speaker turns map to token positions in the input sequence (needed for turn-localized masking later), (c) the generation entry point that runs the full AR loop with the tokenizer feedback.

**Deliverable:** `experiments/p0_steering/NOTES.md` stage-0 section with model/version pins and the module paths for hook placement.

## Stage 1 — Contrast-pair generation (½ day)

Build `src/steering/contrast_pairs.py`:

1. Write 10 emotionally neutral scripts (2–4 sentences each, single speaker). Vary content domain so vectors don't encode topic.
2. For each script, generate K=2 samples per pole per axis using VibeVoice's own prompting/reference-audio mechanism to push affect:
   - **Arousal axis:** excited/energetic delivery vs. calm/flat delivery.
   - **Valence axis:** warm/happy delivery vs. somber/cold delivery.
   - Same text, same target speaker reference, opposite affect → differences in activations should isolate affect.
3. During each generation, forward hooks on every Qwen2.5 decoder layer capture mean-pooled hidden states over the generated speech frames (not the text prompt region). Save as `{script_id, axis, pole, layer, vector}` in a single `.pt`.
4. **Honesty check:** listen to the pairs. If VibeVoice's own prompting can't produce audibly different affect, note it — the extraction has no signal to find and the run design must change (fall back to Expresso utterances re-encoded through the tokenizer for extraction). Note: the HF `ylacombe/expresso` repo is read-speech only (happy/sad/laughing/whisper); high-arousal styles (angry, projected, fast) are improvised-only and require Meta's official tarball via textlesslib — see `docs/resources.md` §4.

## Stage 2 — Vector extraction (2 hrs)

Build `src/steering/extract.py`:

1. Per axis, per layer: steering direction = mean(positive-pole activations) − mean(negative-pole activations), normalized.
2. Diagnostics before any injection:
   - Cosine similarity of per-script directions within an axis (consistency — if per-script directions don't agree, the "axis" is noise; expect >0.4 mean pairwise cosine on mid layers).
   - Cosine between the arousal and valence directions (independence — if ~1.0, we have one "expressiveness" axis, not two; that's still a result, log it).
   - Per-layer direction norms → identifies candidate injection layers (expect strongest, most consistent signal in middle third of the stack, mirroring EmoSteer's findings).

## Stage 3 — Injection + listening protocol (½ day)

Build `src/steering/inject.py`:

1. Injection: forward hook adds `α · direction` to the residual stream at chosen layer(s) during generation, applied only at speech-frame positions (Stage-0 finding (b) gives the position mask).
2. Sweep matrix — 3 held-out scripts × {arousal, valence} × α ∈ {0.5, 1, 2, 4, 8} × layer choice ∈ {best single layer, middle-third band}. ~90 short generations, minutes each on the 3090.
3. For every sample, log automatically: Whisper-large-v3 WER vs. script, ECAPA cosine to the speaker reference, F0 mean/var and energy (arousal should move F0/energy monotonically with α — a free objective signal).
4. **Listen to everything.** Metrics rank; ears decide. Note the α where affect becomes audible and the α where speech degrades — the gap between them is the usable control range.
5. Multi-speaker probe (the actual C2 claim, keep it minimal at P0): one 2-speaker dialogue, steer only speaker B's turns via the position mask, α at the sweet spot. Verify speaker A is unaffected (ECAPA + ears).

## Success criteria

| Verdict | Condition |
|---|---|
| **PASS** | Some (α, layer) gives clearly audible affect shift, WER within +10% relative of unsteered, ECAPA to reference ≥ 0.85× unsteered, and the multi-speaker probe leaves speaker A unchanged |
| **PARTIAL** | Audible shift exists but control range is narrow or axes collapse into one — C2 reshapes to single-axis "intensity steering," still novel per-speaker |
| **FAIL** | No audible shift before intelligibility breaks at any α/layer — C2 dropped; paper proceeds on C1+C3+C4; write the negative result |

## Deliverables

- `experiments/p0_steering/NOTES.md` — hypothesis, config, sweep table, verdict (per repo convention, filled in even on FAIL).
- `experiments/p0_steering/audio/` — baseline, best-α per axis, the multi-speaker probe, and one over-steered failure sample (papers love the failure sample).
- The three `src/steering/` modules above with shape tests in `tests/test_steering.py`.
- On PASS: tag `p0-pass`, then proceed to P1 (caching pipeline `src/cache/` — the hooks written here are 80% of the caching code, reuse them).

## Known risks / pre-answered questions

- **"Prompted contrast pairs might encode prompt-following, not emotion."** Mitigated by using identical text + reference audio across poles and pooling over 10 scripts; residual concern checked by the Expresso fallback in Stage 1.4.
- **"Steering the backbone might shift timbre, not prosody."** That's what the ECAPA gate and the F0/energy monotonicity check are for; if timbre moves with affect, log it — it informs whether C3 anchoring must run concurrently with steering.
- **"Which layers?"** Don't theorize; the Stage 2 per-layer diagnostics plus the layer arm of the sweep answer it empirically.
