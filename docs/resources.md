# Resources — verified 2026-07-05

Pinned libraries, models, datasets, and implementation strategy. Everything here was
verified live (HF API / GitHub / papers) on the date above by four research passes.
Re-verify anything marked UNVERIFIED before depending on it.

## 1. Backbone: VibeVoice

| Item | Pin |
|---|---|
| Code | `github.com/vibevoice-community/VibeVoice` @ `07cb79fea` (main HEAD, 2026-06-12; alive, preservation-mode, MIT) |
| Install | source only: `git clone` + `pip install -e .` — **not on PyPI** |
| Hard dep pin | `transformers==4.51.3` (fork pins it; "later version may not be compatible"), `accelerate==1.6.0` |
| 1.5B weights | `microsoft/VibeVoice-1.5B` — **live again on HF** (restored Jan 2026), MIT, ungated, safetensors |
| 7B weights | `vibevoice/VibeVoice-7B` or `aoi-ot/VibeVoice-Large` (mirrors, MIT). `microsoft/VibeVoice-Large` never existed; `WestZhang/VibeVoice-Large-pt` is gone |
| Attention | flash-attn 2 optional (3090 = SM86 supported); auto-fallback to SDPA with a quality warning |
| VRAM | 1.5B bf16 ≈ 7–10 GB (3090 comfortable); 7B ≈ 18–24 GB (3090 edge; Q8/AWQ quants exist) |
| Context | 1.5B: 64K tokens ≈ 90 min; 7B: 32K ≈ 45 min |

### Hook map (P0 Stage 0 deliverable, pre-verified from source)

- **Decoder layers (steering hooks):** `model.model.language_model.layers[i]` — stock HF
  `Qwen2Model`, 28 × `Qwen2DecoderLayer`, **hidden_size 1536** (confirmed from
  `microsoft/VibeVoice-1.5B/config.json`; Qwen2.5-1.5B geometry, GQA 12/2 heads).
- **Generation entry point (full AR loop + tokenizer feedback):**
  `VibeVoiceForConditionalGenerationInference.generate()` in
  `vibevoice/modular/modeling_vibevoice_inference.py:327-697`. Per speech token:
  last hidden state → diffusion head → 64-dim latent → acoustic decode → semantic
  re-encode → connectors → next-step `inputs_embeds`. This is the feedback loop that
  hard constraint 2 requires cached states to include.
- **Speaker→position mapping (turn-localized steering mask):** `VibeVoiceProcessor`
  (`vibevoice/processor/vibevoice_processor.py`) — `_parse_script()` (line 604),
  per-turn `" Speaker {id}:{text}\n"` lines, voice prompts as `<|vision_pad|>` runs
  marked by `speech_input_mask` (`_create_voice_prompt()`, line 406). No per-speaker
  special tokens — turn localization must come from tracking generated positions per turn.
- **Diffusion head (what the flow head replaces):** `VibeVoiceDiffusionHead`
  (`vibevoice/modular/modular_vibevoice_diffusion_head.py:191`) —
  `forward(noisy_images, timesteps, condition)`, condition = `[B, d_model]` last hidden
  state per AR step; 4 layers of RMSNorm → AdaLN (shift/scale/gate) → SwiGLU; conditioning
  = `proj(hidden) + timestep_emb`, summed. **v-prediction**, cosine schedule, DPM-Solver++.
  Latent: 64-dim σ-VAE (`fix_std=0.5`) at 7.5 Hz (compress ratio 3200 @ 24 kHz); learned
  `speech_scaling_factor`/`speech_bias_factor` un-scaling before decode. Semantic feedback
  dim 128; both connectors Linear→RMSNorm→Linear.
- **CFG detail:** inference keeps a parallel negative KV cache and applies CFG
  (`sample_speech_tokens`, default scale 3.0, ~20 DPM-Solver steps in code — the README's
  "10-step" figure needs reconciling against measured defaults; if 20 steps × 2× CFG
  passes is the real baseline, the speedup headline improves).

## 2. Flow head implementation strategy

**Hand-roll both objectives** (verdict backed by unanimous field practice — F5-TTS,
ZipVoice, Matcha lineage all hand-roll; `torchcfm` is alive but adds nothing for a
conditional head).

- **OT-CFM (P1):** linear interpolant, 4 lines: `t~U(0,1)`, `φ=(1−t)x0+tx1`,
  target `x1−x0`, MSE. Euler loop by hand (no torchdiffeq). Consider F5-TTS's Sway
  Sampling (cosine-warped t schedule) for the 4-NFE schedule.
- **MeanFlow (P2):** ~60 lines around `torch.func.jvp`. Reference implementations:
  `Gsunshine/py-meanflow` (official PyTorch), `zhuyu-cs/MeanFlow` (best community
  repro), `haidog-yaqub/MeanFlow` (audio-adjacent; JVP-under-no_grad memory trick).
  Paper-verified recipe: **JVP tangent `(v, 0, 1)`** (wrong tangent = silent
  catastrophic failure, FID 61→329); embed `(t, t−r)` summed into AdaLN; (r,t) ~
  lognorm(−0.4, 1.0), swap so t>r; **25% r≠t**; adaptive weight
  `sg(1/(‖Δ‖²+1e−3)^1.0)`. Time-embedding MLPs and AdaLN modulation **must live inside
  the jvp'd closure** or ∂t u silently zeroes.
- **Precision:** **train the head fp32.** Open PyTorch bug #165324: `torch.func.jvp`
  inside bf16 autocast breaks; forward-mode AD + AMP effectively unsupported. At 15M
  params fp32 is free. No attention in the head ⇒ the FlashAttention-JVP
  incompatibility doesn't apply.
- **Optimizer (starting point):** Adam, lr 1e-4–3e-4 constant (no decay), betas
  (0.9, 0.95), wd 0, EMA 0.9999. LatentLM trick worth stealing: sample ~4 diffusion
  timesteps per cached condition per forward pass.
- **Head architecture starting point:** VibeVoice's own 4-layer RMSNorm+SwiGLU+AdaLN
  head, retargeted from v-prediction to FM velocity.
- **AdaLN vs token-conditioning ablation (P2):** DSFlow (arXiv:2602.09041) removed
  adaLN-Zero entirely at fixed 1-step schedules — K learnable step tokens, ~1.5K params
  vs 38M, slightly *better* MOS. Applies to fixed discrete schedules post-distillation,
  not continuous-t training. No open-source DSFlow code found.
- **Fallback if MeanFlow-from-scratch is unstable:** JVP-free distillation from the P1
  teacher — finite-difference mean velocity over trajectory intervals + endpoint mix
  α≈0.7 (DSFlow / IntMeanFlow pattern). P1 teacher is the safety net; keep it good.

## 3. Eval stack

All models want 16 kHz mono. IO pattern: `soundfile.read` → torch →
`torchaudio.functional.resample` once → feed everything.

| Component | Pin | Gotchas |
|---|---|---|
| ASR | `faster-whisper` (v1.2.1+), model `large-v3` (`Systran/faster-whisper-large-v3`), `BatchedInferencePipeline` | ctranslate2 needs CUDA12+cuDNN9; pin `ctranslate2==4.4.0` on cuDNN8 boxes |
| WER | `jiwer>=4.0` + `whisper-normalizer` (`EnglishTextNormalizer`, applied to both ref and hyp) | jiwer 4.0 changed empty-reference semantics vs 3.x |
| Speaker sim (easy) | `speechbrain>=1.0` + `speechbrain/spkrec-ecapa-voxceleb`, cosine on 192-dim embeddings | import from `speechbrain.inference`, not `.pretrained` |
| Speaker sim (paper-comparable "SIM-o") | WavLM-large SV checkpoint (`wavlm_large_finetune.pth`) from `microsoft/UniSpeech` GitHub — vendor `verification.py` + ckpt like F5-TTS does | **not on HF**; this is what Seed-TTS/F5/CosyVoice numbers use. `microsoft/wavlm-base-plus-sv` is the easy-but-not-comparable alternative |
| MOS | UTMOS22 via `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")` | frozen-not-dead; pin the tag. Optionally add UTMOSv2 (`git+...UTMOSv2@v1.3.0`) where UTMOS saturates |
| F0 | `praat-parselmouth` (autocorrelation pitch, 75–600 Hz defaults; mask unvoiced=0 frames before stats) | CPU-only, fast enough |
| Audio IO | `soundfile` + `torchaudio` (matching torch version) | **torchaudio ≥2.9 removed native IO** — `torchaudio.load` now needs `torchcodec`; use soundfile for IO, torchaudio for DSP only |

## 4. Datasets

| Dataset | Source | License | Notes |
|---|---|---|---|
| LibriTTS-R (flow-head caching, 75K) | `mythicinfinity/libritts_r` (HF, parquet) or OpenSLR 141 | CC BY 4.0 | Sample 75K from `train.clean.360` (116K utts, ~33 GB) — one download, uniform quality. Skip `train-other-500`. **Filter against OpenSLR's `libritts_r_failed_speech_restoration_examples` list** |
| Expresso (steering validation) | `ylacombe/expresso` (HF) = **read-speech only** (11.6K rows, 8 styles: happy/sad/laughing/whisper/...) | CC BY-NC 4.0 | High-arousal styles (angry, projected, fast, excited-adjacent) are **improvised-only → Meta tarball** via `facebookresearch/textlesslib` examples/expresso. NC is fine for validation-only academic use; don't train released weights on it |
| Dialogue scripts (caching gen input) | `allenai/soda` (1.5M dialogues) | CC BY 4.0 | Safest primary. `li2017dailydialog/daily_dialog` (CC BY-NC-SA) has per-utterance emotion labels — useful for contrast-pair prompt construction. MediaSum = real interview register but research-only, script-based loader |
| Nonverbal (P-stretch) | SoulX-Podcast (arXiv:2510.23541) released weights only — **dataset + mining pipeline NOT public** | — | "SoulX-style" = reimplement their recipe: audio-event tagger pass + LLM-judge verification. Their tags: laughter/sigh/breathing/coughing/throat_clearing |

## 5. Doc reconciliations owed (from this research)

1. README "10-step DDPM head × 40,500 frames" — code default appears to be ~20
   DPM-Solver steps + CFG (2× passes). Measure the actual shipped default in P0 Stage 0
   and update the README math; the speedup claim likely gets *better*.
2. P0 spec Stage 1 fallback ("Expresso utterances re-encoded through the tokenizer"):
   note that high-arousal contrast material needs the Meta improvised tarball, not the
   HF repo.
3. Eval code should default to ECAPA for iteration-speed gates and report WavLM-large
   SV (UniSpeech) for paper tables.
