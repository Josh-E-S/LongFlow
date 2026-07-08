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
    hiddens, latents = [], []
    for f in files:
        utt = load_utterance(f)
        hiddens.append(utt.hidden.float())
        latents.append(utt.latent.float())
    hidden = torch.cat(hiddens)
    latent = torch.cat(latents)
    mean = latent.mean(dim=0)
    std = latent.std(dim=0).clamp_min(1e-4)
    return PairData(hidden=hidden, latent=(latent - mean) / std, mean=mean, std=std)


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
    ema_decay: float = 0.9999,
    device: str = "cpu",
    log_every: int = 500,
    seed: int = 0,
) -> dict:
    head = head.float().to(device).train()
    hidden = data.hidden.to(device)
    latent = data.latent.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    ema = EMA(head, ema_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, hidden.shape[0], (batch_size,), generator=g)
        loss = cfm_loss(head, latent[idx], hidden[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        ema.update(head)
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            recent = sum(losses[-log_every:]) / len(losses[-log_every:])
            print(f"step {step}/{steps}  loss {recent:.4f}")
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
) -> torch.Tensor:
    """condition [T, d_model] -> head-space latents [T, d_latent] (de-standardized).

    seed defaults to None (fresh entropy): a deterministic-by-default sampler
    inside an AR loop is a footgun — identical x0 every call reproduces the
    under-dispersion signature (review-adversarial.md §2c). Pass a seed only
    for reproducible offline evals. sway default changed -1.0 -> 0.0: the
    front-loaded grid leaves a giant final step in the re-expansion phase and
    under-disperses even a perfect field (§2b); sweep it explicitly in E1.
    """
    device = next(head.parameters()).device
    g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    z = euler_sample(
        head, condition.float().to(device), head.cfg.d_latent, nfe=nfe, sway=sway, generator=g
    )
    return z * latent_std.to(device) + latent_mean.to(device)
