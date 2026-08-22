"""P1 flow-head trainer.

Recipe per docs/resources.md §2 (MeanFlow-paper lineage, applies to the CFM
baseline too): Adam, constant LR, betas (0.9, 0.95), weight decay 0, EMA 0.9999.
Head trains in fp32 (cheap at 15M; also required later for the P2 JVP).

Latents are standardized per-dim with dataset stats before training (flow
matching starts from N(0,I); matched scales help few-NFE). The stats live in
every checkpoint; `sample_latents` inverts them, so downstream decode always
sees head-space latents.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from src.cache.capture import load_utterance
from src.flow_head.cfm import cfm_loss, euler_sample
from src.flow_head.model import FlowHead, FlowHeadConfig


@dataclass
class PairData:
    hidden: torch.Tensor  # [N, d_model] fp32
    latent: torch.Tensor  # [N, d_latent] fp32, standardized
    mean: torch.Tensor  # [d_latent]
    std: torch.Tensor  # [d_latent]
    # head-v2 (2026-08-17): None on v1-style pools
    neg_hidden: torch.Tensor | None = None  # [N, d_model] fp32
    sigma_bucket: torch.Tensor | None = None  # [N] long (see model.sigma_to_bucket)

    @property
    def d_model(self) -> int:
        return self.hidden.shape[1]

    @property
    def d_latent(self) -> int:
        return self.latent.shape[1]


def load_pairs(cache_dir, limit: int | None = None) -> PairData:
    """Flatten cached utterances into frame pairs; compute latent stats."""
    files = sorted(Path(cache_dir).glob("*.pt"))[:limit]
    if not files:
        raise FileNotFoundError(f"no cached utterances in {cache_dir}")
    return pairs_from_files(files)


def pairs_from_files(files, dual_stream: bool = False) -> PairData:
    """Flatten an explicit file list into frame pairs; compute latent stats.

    dual_stream=True reads the v2 fields (neg_hidden + sigma) and REFUSES any
    file missing them — silently training one-stream on a dual-stream pool is
    the exact discard bug of the 2026-08-15 CFG audit (and of the first v2
    20K run, which never read neg_hidden at all)."""
    from src.flow_head.model import sigma_to_bucket

    if not files:
        raise FileNotFoundError("empty file list")
    hiddens, latents, negs, sigmas = [], [], [], []
    for f in files:
        utt = load_utterance(f)
        hiddens.append(utt.hidden.float())
        latents.append(utt.latent.float())
        if dual_stream:
            if utt.neg_hidden is None or utt.sigma is None:
                raise ValueError(f"{f}: dual_stream pool but neg_hidden/sigma missing")
            negs.append(utt.neg_hidden.float())
            sigmas.append(utt.sigma.float())
    hidden = torch.cat(hiddens)
    latent = torch.cat(latents)
    mean = latent.mean(dim=0)
    std = latent.std(dim=0).clamp_min(1e-4)
    return PairData(
        hidden=hidden,
        latent=(latent - mean) / std,
        mean=mean,
        std=std,
        neg_hidden=torch.cat(negs) if dual_stream else None,
        sigma_bucket=sigma_to_bucket(torch.cat(sigmas)) if dual_stream else None,
    )


def filter_flagged(files, flags_path) -> list:
    """Drop cache files named in a capture_v2_audit_flags.json ('flagged' key).

    Loud by design: prints the counts and raises if the flag list matches
    nothing (wrong dir would silently train on poisoned data otherwise)."""
    import json

    with open(flags_path) as fh:
        flagged = set(json.load(fh)["flagged"])
    keep = [f for f in files if Path(f).stem not in flagged]
    hit = len(files) - len(keep)
    print(
        f"filter_flagged: {len(files)} files -> {len(keep)} (removed {hit}/{len(flagged)} flagged)"
    )
    if flagged and hit == 0:
        raise ValueError("flag list matched no files — wrong cache dir or stale flags?")
    return keep


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            p.copy_(self.shadow[n])


def train(
    head: FlowHead,
    data: PairData,
    steps: int = 5000,
    batch_size: int = 512,
    lr: float = 2e-4,
    lr_final: float | None = None,
    ema_decay: float = 0.9999,
    device: str = "cpu",
    log_every: int = 500,
    seed: int = 0,
    checkpoint_every: int | None = None,
    checkpoint_path_fn=None,
    loss_fn=None,
) -> dict:
    """lr_final: if set, cosine-decay from lr to lr_final over the run
    (review-adversarial.md §3 — constant-lr-to-the-end leaves terminal loss
    on the table).

    loss_fn: optional callable (head, x1, condition) -> scalar loss replacing
    the default CFM objective (P2: pass a meanflow_loss lambda). The custom
    path is cond-only — dual-stream/sigma pools are refused with it.

    checkpoint_every/checkpoint_path_fn: if both given, saves an intermediate
    checkpoint every `checkpoint_every` steps via `checkpoint_path_fn(step)`
    -> path (process rule, 2026-08-11: "save intermediate checkpoints in ALL
    runs" — this is what let the project catch the 20K-vs-80K overfit finding
    in the first place; a single final-step save can't).
    """
    import math

    head = head.float().to(device).train()
    hidden = data.hidden.to(device)
    latent = data.latent.to(device)
    # dual-stream/sigma mismatches between head config and pool are refused by
    # the head's own forward on step 1 — no silent one-stream fallback here
    neg = data.neg_hidden.to(device) if data.neg_hidden is not None else None
    sig = data.sigma_bucket.to(device) if data.sigma_bucket is not None else None
    opt = torch.optim.Adam(head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    ema = EMA(head, ema_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for step in range(1, steps + 1):
        if lr_final is not None:
            frac = 0.5 * (1 + math.cos(math.pi * step / steps))
            opt.param_groups[0]["lr"] = lr_final + (lr - lr_final) * frac
        idx = torch.randint(0, hidden.shape[0], (batch_size,), generator=g)
        if loss_fn is not None:
            if neg is not None or sig is not None:
                raise ValueError(
                    "custom loss_fn path is cond-only — load the pool without v2 fields"
                )
            loss = loss_fn(head, latent[idx], hidden[idx])
        else:
            loss = cfm_loss(
                head,
                latent[idx],
                hidden[idx],
                neg_condition=neg[idx] if neg is not None else None,
                sigma_bucket=sig[idx] if sig is not None else None,
            )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        ema.update(head)
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            recent = sum(losses[-log_every:]) / len(losses[-log_every:])
            print(f"step {step}/{steps}  loss {recent:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
        if checkpoint_every and checkpoint_path_fn and step % checkpoint_every == 0:
            save_checkpoint(checkpoint_path_fn(step), head, ema, data, step=step)
    return {"losses": losses, "ema": ema}


def save_checkpoint(path, head: FlowHead, ema: EMA, data: PairData, step: int) -> None:
    torch.save(
        {
            "config": vars(head.cfg),
            "model": head.state_dict(),
            "ema": ema.shadow,
            "latent_mean": data.mean,
            "latent_std": data.std,
            "step": step,
        },
        path,
    )


def load_checkpoint(path, use_ema: bool = True):
    """Returns (head, latent_mean, latent_std); EMA weights by default."""
    ckpt = torch.load(path, weights_only=True)
    head = FlowHead(FlowHeadConfig(**ckpt["config"]))
    head.load_state_dict(ckpt["model"])
    if use_ema:
        for n, p in head.named_parameters():
            p.data.copy_(ckpt["ema"][n])
    head.eval()
    return head, ckpt["latent_mean"], ckpt["latent_std"]


@torch.no_grad()
def sample_latents(
    head: FlowHead,
    condition: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    nfe: int = 4,
    sway: float = 0.0,
    seed: int | None = None,
    neg_condition: torch.Tensor | None = None,
    sigma_bucket=0,
    sampler=None,
) -> torch.Tensor:
    """condition [T, d_model] -> head-space latents [T, d_latent] (de-standardized).

    seed defaults to None (fresh entropy): a deterministic-by-default sampler
    inside an AR loop is a footgun — identical x0 every call reproduces the
    under-dispersion signature (review-adversarial.md §2c). Pass a seed only
    for reproducible offline evals. sway default changed -1.0 -> 0.0: the
    front-loaded grid leaves a giant final step in the re-expansion phase and
    under-disperses even a perfect field (§2b); sweep it explicitly in E1.

    neg_condition [T, d_model] + sigma_bucket: required for dual-stream/
    sigma-conditioned heads (bound via model.BoundField); the head refuses a
    config mismatch either way. sampler: euler_sample default; pass
    cfm.heun_sample for dispersion-correct evals (GN8 operating config).
    """
    from src.flow_head.model import BoundField

    device = next(head.parameters()).device
    g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    field = head
    if head.cfg.dual_stream or head.cfg.sigma_buckets > 0:
        neg = neg_condition.float().to(device) if neg_condition is not None else None
        field = BoundField(head, neg, sigma_bucket)
    sample_fn = sampler or euler_sample
    z = sample_fn(
        field, condition.float().to(device), head.cfg.d_latent, nfe=nfe, sway=sway, generator=g
    )
    return z * latent_std.to(device) + latent_mean.to(device)
