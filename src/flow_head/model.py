"""P1 flow head — ~15M-param conditional velocity network.

Architecture mirrors VibeVoice's own diffusion head (the proven design for this
conditioning: docs/resources.md §1/§2): per-frame MLP, no attention — the
backbone already captured temporal context. Blocks are RMSNorm → AdaLN
(shift/scale/gate) → SwiGLU FFN, conditioning = proj(LM hidden) + timestep
embedding, summed, driving per-layer AdaLN. Differences from VibeVoice's head:
width 640 instead of 1536 (123M → ~15M, constraint 5: thin adapters) and the
prediction target is flow-matching velocity, not v-prediction diffusion.

AdaLN-zero init: modulation and final projection start at zero, so the head is
the zero function at init — training moves it away from a known-safe start.

d_model and d_latent always come from the caller (checkpoint config / cached
tensors), never hardcoded (hard constraint 3).
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FlowHeadConfig:
    d_model: int  # conditioning width, from checkpoint config at runtime
    d_latent: int  # acoustic latent width, from sigma-VAE config at runtime
    width: int = 640
    layers: int = 4
    ffn_ratio: float = 2.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * self.weight if self.weight is not None else x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        inner = int(dim * ratio)
        self.gate = nn.Linear(dim, inner, bias=False)
        self.up = nn.Linear(dim, inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.ffn = SwiGLU(dim, ratio)
        self.modulation = nn.Linear(dim, 3 * dim, bias=True)
        nn.init.zeros_(self.modulation.weight)  # AdaLN-zero
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, cond):
        shift, scale, gate = self.modulation(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1]; t shape [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None, :] * 1000.0
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class FlowHead(nn.Module):
    """velocity = head(x_t, t, condition); all inputs per-frame ([B, d])."""

    def __init__(self, cfg: FlowHeadConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.in_proj = nn.Linear(cfg.d_latent, w)
        self.cond_proj = nn.Linear(cfg.d_model, w)
        self.time_mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([AdaLNBlock(w, cfg.ffn_ratio) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(w, affine=False)
        self.out_proj = nn.Linear(w, cfg.d_latent)
        nn.init.zeros_(self.out_proj.weight)  # zero function at init
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor):
        if x_t.shape[-1] != self.cfg.d_latent or condition.shape[-1] != self.cfg.d_model:
            raise ValueError(
                f"dim mismatch: x_t {tuple(x_t.shape)}, cond {tuple(condition.shape)} "
                f"vs config d_latent={self.cfg.d_latent}, d_model={self.cfg.d_model}"
            )
        if x_t.ndim != 2 or condition.ndim != 2:
            # a [B, 1, d] condition would silently broadcast through AdaLN into
            # wrong-shaped garbage (pre-75K audit finding 3) — refuse loudly
            raise ValueError(
                f"expected 2-D [B, d] inputs: x_t {tuple(x_t.shape)}, cond {tuple(condition.shape)}"
            )
        cond = self.cond_proj(condition) + self.time_mlp(timestep_embedding(t, self.cfg.width))
        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.final_norm(h))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
