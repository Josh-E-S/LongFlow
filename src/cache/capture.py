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
    hidden: torch.Tensor  # [T, d_model] fp16 -- cond-stream hidden state
    latent: torch.Tensor  # [T, d_latent] fp16 -- teacher's DDPM output (target)
    meta: dict = field(default_factory=dict)
    # v2 fields (GN8 dual-stream + noise-augmented capture); None on v1 caches.
    neg_hidden: torch.Tensor | None = None  # [T, d_model] fp16 -- neg-stream hidden state
    sigma: torch.Tensor | None = None  # [T] fp16 -- feedback-noise sigma bucket applied per frame


class SampleCapture:
    """Context manager that wraps `getattr(model, method_name)` to record
    (condition, neg_condition, latent, sigma) per call. Restores the original
    method on exit.

    d_model / d_latent are read from the first captured tensors at runtime,
    never hardcoded (hard constraint 3). `noise`: an active
    `src.cache.noise.NoiseIntervention` to read `.last_sigma` from after each
    call (capture v2); omit for v1-style clean captures.
    """

    def __init__(self, model, method_name: str = "sample_speech_tokens", noise=None):
        self.model = model
        self.method_name = method_name
        self.noise = noise
        self._orig = None
        self.conditions: list[torch.Tensor] = []
        self.neg_conditions: list[torch.Tensor | None] = []
        self.latents: list[torch.Tensor] = []
        self.sigmas: list[float] = []

    def __enter__(self):
        # If the attribute lives on the instance we must restore it on exit;
        # if it lives on the class we must delete our instance-level shadow.
        self._was_instance_attr = self.method_name in vars(self.model)
        self._orig = getattr(self.model, self.method_name)
        capture = self

        def wrapped(condition, *args, **kwargs):
            # forward exactly what the caller passed -- injecting explicit
            # defaults for omitted args would override the real method's own
            out = capture._orig(condition, *args, **kwargs)
            neg_condition = kwargs.get("neg_condition", args[0] if args else None)
            capture.conditions.append(condition.detach().to("cpu", torch.float16))
            capture.neg_conditions.append(
                neg_condition.detach().to("cpu", torch.float16)
                if neg_condition is not None
                else None
            )
            capture.latents.append(out.detach().to("cpu", torch.float16))
            capture.sigmas.append(capture.noise.last_sigma if capture.noise is not None else 0.0)
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
        for c, z in zip(self.conditions, self.latents, strict=True):
            # exactly one frame row per call: a multi-row condition (e.g. a fork
            # update stacking pos+neg streams) would otherwise be silently
            # flattened into one [1, k*d] row (pre-75K audit finding 3)
            if c.numel() != c.shape[-1] or z.numel() != z.shape[-1]:
                raise ValueError(
                    f"{utt_id!r}: multi-row capture call (cond {tuple(c.shape)}, "
                    f"latent {tuple(z.shape)}) — unbatched capture expects one frame per call"
                )
        hidden = torch.cat([c.reshape(1, -1) for c in self.conditions]).unsqueeze(0)  # [1, T, dm]
        latent = torch.cat([z.reshape(1, -1) for z in self.latents]).unsqueeze(0)  # [1, T, dl]
        assert_frame_aligned(hidden.float(), latent.float())

        neg_hidden = None
        if all(n is not None for n in self.neg_conditions):
            for n in self.neg_conditions:
                if n.numel() != n.shape[-1]:
                    raise ValueError(
                        f"{utt_id!r}: multi-row neg_condition capture call {tuple(n.shape)} "
                        "— unbatched capture expects one frame per call"
                    )
            neg_hidden = torch.cat([n.reshape(1, -1) for n in self.neg_conditions]).unsqueeze(0)
            assert_frame_aligned(hidden.float(), neg_hidden.float())
            neg_hidden = neg_hidden[0]
        elif any(n is not None for n in self.neg_conditions):
            raise ValueError(
                f"{utt_id!r}: neg_condition present on some frames but not all "
                "— dual-stream capture requires every frame to carry it"
            )

        sigma = torch.tensor(self.sigmas, dtype=torch.float16) if self.noise is not None else None

        return UtteranceCache(
            utt_id=utt_id,
            text=text,
            hidden=hidden[0],
            latent=latent[0],
            meta=meta or {},
            neg_hidden=neg_hidden,
            sigma=sigma,
        )

    def reset(self) -> None:
        self.conditions.clear()
        self.neg_conditions.clear()
        self.latents.clear()
        self.sigmas.clear()


class BatchedSampleCapture:
    """Batched variant of SampleCapture for multi-utterance generation.

    In a batch, elements finish at different times, so each
    sample_speech_tokens call carries condition/latent rows for only the
    batch elements emitting a speech frame that step (VibeVoice slices with
    `diffusion_indices`, ascending batch order). We record raw per-call row
    blocks, then `split_utterances` reassigns rows to elements by replaying
    the per-element token streams — with hard invariants: per-step row count
    must equal that step's active count, and per-element frame totals must
    match their token streams exactly.
    """

    def __init__(self, model, method_name: str = "sample_speech_tokens", noise=None):
        self.model = model
        self.method_name = method_name
        self.noise = noise  # active NoiseIntervention to read .last_sigma from (capture v2)
        self._orig = None
        self._was_instance_attr = False
        # (cond, lat, neg_or_None, sigma) [rows, d] per call
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]] = []

    def __enter__(self):
        self._was_instance_attr = self.method_name in vars(self.model)
        self._orig = getattr(self.model, self.method_name)
        capture = self

        def wrapped(condition, *args, **kwargs):
            out = capture._orig(condition, *args, **kwargs)
            neg_condition = kwargs.get("neg_condition", args[0] if args else None)
            capture.calls.append(
                (
                    condition.detach().to("cpu", torch.float16).reshape(condition.shape[0], -1),
                    out.detach().to("cpu", torch.float16).reshape(out.shape[0], -1),
                    (
                        neg_condition.detach()
                        .to("cpu", torch.float16)
                        .reshape(neg_condition.shape[0], -1)
                        if neg_condition is not None
                        else None
                    ),
                    capture.noise.last_sigma if capture.noise is not None else 0.0,
                )
            )
            return out

        setattr(self.model, self.method_name, wrapped)
        return self

    def __exit__(self, *exc):
        if self._was_instance_attr:
            setattr(self.model, self.method_name, self._orig)
        else:
            delattr(self.model, self.method_name)
        self._orig = None
        return False

    def _attribute(
        self, token_streams: list[list[int]], frame_id: int
    ) -> tuple[dict[int, list[tuple]], set[int]]:
        """Replays token_streams against self.calls, call by call, attributing
        each captured row to its originating batch element.

        Elements finish at different times in a batch, and -- especially with
        long, duration-varied renders (capture v2) -- an element's very LAST
        frame token is sometimes never rendered (generation stops right
        before its sample_speech_tokens call). That is tolerated per element,
        at whichever step it actually occurs -- not just at the batch's
        global final step (the original, narrower tolerance, sized for
        cache10k-scale short uniform-length batches where all elements
        finished around the same time and a mid-batch drop was never
        observed; GN1-style long duration-varied batches hit it routinely,
        2026-08-16). A deficit is only ever auto-explained when the number of
        elements whose CURRENT position is their own last remaining frame
        exactly equals the deficit -- any ambiguity, or any row surplus, is a
        hard abort. This must never silently misattribute rows.
        """
        n = len(token_streams)
        frame_positions = [[t for t, tok in enumerate(s) if tok == frame_id] for s in token_streams]
        ptr = [0] * n
        dropped: set[int] = set()
        per_b: dict[int, list[tuple]] = {b: [] for b in range(n)}

        call_idx = 0
        while call_idx < len(self.calls):
            candidates = [
                b for b in range(n) if b not in dropped and ptr[b] < len(frame_positions[b])
            ]
            if not candidates:
                break
            t_min = min(frame_positions[b][ptr[b]] for b in candidates)
            active = [b for b in candidates if frame_positions[b][ptr[b]] == t_min]
            cond, lat, neg, sigma = self.calls[call_idx]
            if cond.shape[0] > len(active):
                raise RuntimeError(
                    f"attribution unsafe at step {t_min}: {cond.shape[0]} rows vs "
                    f"{len(active)} active elements — more rows than expected, aborting"
                )
            keep = active
            if cond.shape[0] < len(active):
                deficit = len(active) - cond.shape[0]
                drop_candidates = [b for b in active if ptr[b] == len(frame_positions[b]) - 1]
                if len(drop_candidates) != deficit:
                    raise RuntimeError(
                        f"attribution unsafe at step {t_min}: {cond.shape[0]} rows vs "
                        f"{len(active)} active elements, {len(drop_candidates)} explainable "
                        f"drop candidates for a deficit of {deficit} — aborting"
                    )
                dropped.update(drop_candidates)
                keep = [b for b in active if b not in drop_candidates]
            for row, b in enumerate(keep):
                per_b[b].append((cond[row], lat[row], neg[row] if neg is not None else None, sigma))
                ptr[b] += 1
            call_idx += 1

        if call_idx < len(self.calls):
            raise RuntimeError(
                f"attribution unsafe: {len(self.calls) - call_idx} capture calls left over "
                "with no more expected frames — aborting"
            )
        for b in range(n):
            remaining = len(frame_positions[b]) - ptr[b]
            if remaining == 0:
                continue
            if remaining == 1:
                dropped.add(b)  # this element's final frame token was never rendered
                continue
            raise RuntimeError(
                f"attribution unsafe: element {b} has {remaining} unconsumed frame "
                "tokens with no more capture calls — aborting"
            )
        return per_b, dropped

    def split_utterances(
        self, token_streams: list[list[int]], frame_id: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """token_streams[b] = generated token ids for batch element b (aligned
        steps: index t is the same generation step for every element; pad
        finished elements with any non-frame id). Returns per-element
        (hidden [T_b, d_model], latent [T_b, d_latent]) fp16 tensors.

        v1-compatible: drops neg_condition/sigma even if this capture recorded
        them. Use split_utterances_v2 to get the dual-stream/sigma fields."""
        per_b, dropped = self._attribute(token_streams, frame_id)
        out = []
        for b, stream in enumerate(token_streams):
            expected = sum(1 for tok in stream if tok == frame_id)
            if b in dropped:
                expected -= 1
            got = len(per_b[b])
            if got != expected:
                raise RuntimeError(f"element {b}: {got} frames assigned vs {expected} frame tokens")
            hidden = torch.stack([r[0] for r in per_b[b]])
            latent = torch.stack([r[1] for r in per_b[b]])
            assert_frame_aligned(hidden.float().unsqueeze(0), latent.float().unsqueeze(0))
            out.append((hidden, latent))
        return out

    def split_utterances_v2(
        self, token_streams: list[list[int]], frame_id: int
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]]:
        """Dual-stream + sigma variant for capture v2. Returns per-element
        (hidden [T_b, d_model], latent [T_b, d_latent], neg_hidden or None
        [T_b, d_model], sigma [T_b] fp16) tensors. neg_hidden is None for
        every element if this capture recorded no neg_condition anywhere
        (v1-style clean capture); partial dual-stream (present on some calls,
        absent on others) is refused rather than silently dropped."""
        any_neg = any(neg is not None for _, _, neg, _ in self.calls)
        all_neg = all(neg is not None for _, _, neg, _ in self.calls)
        if any_neg and not all_neg:
            raise RuntimeError(
                "neg_condition present on some capture calls but not all — "
                "dual-stream capture requires every call to carry it"
            )
        per_b, dropped = self._attribute(token_streams, frame_id)
        out = []
        for b, stream in enumerate(token_streams):
            expected = sum(1 for tok in stream if tok == frame_id)
            if b in dropped:
                expected -= 1
            got = len(per_b[b])
            if got != expected:
                raise RuntimeError(f"element {b}: {got} frames assigned vs {expected} frame tokens")
            hidden = torch.stack([r[0] for r in per_b[b]])
            latent = torch.stack([r[1] for r in per_b[b]])
            assert_frame_aligned(hidden.float().unsqueeze(0), latent.float().unsqueeze(0))
            neg_hidden = torch.stack([r[2] for r in per_b[b]]) if all_neg else None
            if neg_hidden is not None:
                assert_frame_aligned(hidden.float().unsqueeze(0), neg_hidden.float().unsqueeze(0))
            sigma_t = torch.tensor([r[3] for r in per_b[b]], dtype=torch.float16)
            out.append((hidden, latent, neg_hidden, sigma_t))
        return out


def save_utterance(utt: UtteranceCache, path) -> None:
    torch.save(
        {
            "utt_id": utt.utt_id,
            "text": utt.text,
            "hidden": utt.hidden,
            "latent": utt.latent,
            "meta": utt.meta,
            "neg_hidden": utt.neg_hidden,
            "sigma": utt.sigma,
        },
        path,
    )


def load_utterance(path) -> UtteranceCache:
    d = torch.load(path, weights_only=True)
    utt = UtteranceCache(**d)
    assert_frame_aligned(
        utt.hidden.float().unsqueeze(0), utt.latent.float().unsqueeze(0)
    )  # verify on read too — cheap insurance against corrupt/truncated files
    if utt.neg_hidden is not None:
        assert_frame_aligned(utt.hidden.float().unsqueeze(0), utt.neg_hidden.float().unsqueeze(0))
    if utt.sigma is not None and utt.sigma.shape[0] != utt.hidden.shape[0]:
        raise ValueError(
            f"{utt.utt_id!r}: sigma length {utt.sigma.shape[0]} != T {utt.hidden.shape[0]} "
            "— non-finite/truncated sigma track"
        )
    return utt
