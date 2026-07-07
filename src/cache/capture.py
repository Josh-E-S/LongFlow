"""P1 — (hidden state, acoustic latent) pair caching for flow-head training.

Hook point: VibeVoice's `sample_speech_tokens(condition, neg_condition, cfg_scale)`
is called exactly once per speech frame during full inference (feedback loop
included — hard constraint 2). Its `condition` argument IS the LM hidden state
the diffusion head conditions on ([B, d_model]); its return IS the sampled
acoustic latent ([B, d_latent], in the head's scaled space — the same space the
flow head must learn to generate in). Wrapping this one callable yields exactly
the training pairs, pre-aligned by construction, with no layer-hook reassembly.

Duck-typed: works on any object with a sample-method attribute; unit-testable
with a stub. Every finished utterance passes assert_frame_aligned before write
(hard constraint 3).
"""

from dataclasses import dataclass, field

import torch

from src.cache.alignment import assert_frame_aligned


@dataclass
class UtteranceCache:
    utt_id: str
    text: str
    hidden: torch.Tensor  # [T, d_model] fp16
    latent: torch.Tensor  # [T, d_latent] fp16
    meta: dict = field(default_factory=dict)


class SampleCapture:
    """Context manager that wraps `getattr(model, method_name)` to record
    (condition, latent) per call. Restores the original method on exit.

    d_model / d_latent are read from the first captured tensors at runtime,
    never hardcoded (hard constraint 3).
    """

    def __init__(self, model, method_name: str = "sample_speech_tokens"):
        self.model = model
        self.method_name = method_name
        self._orig = None
        self.conditions: list[torch.Tensor] = []
        self.latents: list[torch.Tensor] = []

    def __enter__(self):
        # If the attribute lives on the instance we must restore it on exit;
        # if it lives on the class we must delete our instance-level shadow.
        self._was_instance_attr = self.method_name in vars(self.model)
        self._orig = getattr(self.model, self.method_name)
        capture = self

        def wrapped(condition, *args, **kwargs):
            out = capture._orig(condition, *args, **kwargs)
            capture.conditions.append(condition.detach().to("cpu", torch.float16))
            capture.latents.append(out.detach().to("cpu", torch.float16))
            return out

        setattr(self.model, self.method_name, wrapped)
        return self

    def __exit__(self, *exc):
        if self._was_instance_attr:
            setattr(self.model, self.method_name, self._orig)
        else:
            delattr(self.model, self.method_name)  # restore class-method lookup
        self._orig = None
        return False

    @property
    def num_frames(self) -> int:
        return len(self.conditions)

    def to_utterance(self, utt_id: str, text: str, meta: dict | None = None) -> UtteranceCache:
        if not self.conditions:
            raise ValueError(f"no frames captured for {utt_id!r}")
        hidden = torch.cat([c.reshape(1, -1) for c in self.conditions]).unsqueeze(0)  # [1, T, dm]
        latent = torch.cat([z.reshape(1, -1) for z in self.latents]).unsqueeze(0)  # [1, T, dl]
        assert_frame_aligned(hidden.float(), latent.float())
        return UtteranceCache(
            utt_id=utt_id,
            text=text,
            hidden=hidden[0],
            latent=latent[0],
            meta=meta or {},
        )

    def reset(self) -> None:
        self.conditions.clear()
        self.latents.clear()


def save_utterance(utt: UtteranceCache, path) -> None:
    torch.save(
        {
            "utt_id": utt.utt_id,
            "text": utt.text,
            "hidden": utt.hidden,
            "latent": utt.latent,
            "meta": utt.meta,
        },
        path,
    )


def load_utterance(path) -> UtteranceCache:
    d = torch.load(path, weights_only=True)
    utt = UtteranceCache(**d)
    assert_frame_aligned(
        utt.hidden.float().unsqueeze(0), utt.latent.float().unsqueeze(0)
    )  # verify on read too — cheap insurance against corrupt/truncated files
    return utt
