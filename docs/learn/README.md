# Learn — the concepts behind LongFlow, explained

Three notebooks for understanding the project's machinery. All run on CPU (no
GPU, no model downloads) — read them rendered on GitHub, or upload to Colab /
run locally (`uv run jupyter` from the repo root; code assumes repo-root cwd).

Reading order:

1. **[01_how_vibevoice_works](01_how_vibevoice_works.ipynb)** (~25 min read) —
   the system we build on: next-token diffusion, thoughts → frames → audio,
   the feedback loop, CFG and the double-stream discovery.
2. **[02_flow_matching_from_scratch](02_flow_matching_from_scratch.ipynb)**
   (~40 min with running) — the speed technique: train a toy flow-matching
   model on two-moons in seconds, watch NFE 1/2/4/16 side by side, then map
   it line-by-line onto the real `src/flow_head/cfm.py`. Ends with the
   MeanFlow (P2) preview.
3. **[03_our_pipeline_annotated](03_our_pipeline_annotated.ipynb)** (~35 min
   read) — our actual code with the stories: the capture wrapper
   (monkeypatching), the April 7 alignment guard, EMA and its gotcha, the
   batched un-shuffling invariants, and how WER/speaker-sim/gate checks work.

These are teaching material, not specs — where they disagree with
`docs/resources.md` or the experiment NOTES, the latter win.
