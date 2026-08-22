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
    # head-v2 (2026-08-17): defaults keep every v1 checkpoint loading unchanged.
    # dual_stream: condition on BOTH LM streams (cond + neg) — the head sees
    # everything the teacher's DDPM head saw, dissolving the one-stream
    # hidden-information bias the 2026-08-15 CFG audit found (guidance is
    # absorbed into the learned field; no inference-time CFG combination).
    dual_stream: bool = False
    # sigma_buckets: >0 adds a GameNGen-style learned embedding telling the
    # head how much feedback noise the captured condition was generated under
    # (reading-pass decision, 2026-08-14). 0 disables.
    sigma_buckets: int = 0
    # meanflow (P2, 2026-08-21): the head becomes an AVERAGE-velocity field
    # u(x_t, t -> r) with a second time input r (target interval end). r = t
    # degenerates to the instantaneous field (CFM-compatible). Defaults keep
    # every earlier checkpoint loading unchanged.
    meanflow: bool = False


# Bucket edges for the capture-v2 sigma schedule (50% clean / 40% U(0.1,0.3) /
# 10% at 0.4, plus fp16-rounding dust just above 0). 4 buckets:
# [0, .05) clean | [.05, .2) light | [.2, .35) medium | [.35, inf) heavy.
SIGMA_BUCKET_EDGES = (0.05, 0.20, 0.35)


def sigma_to_bucket(sigma: torch.Tensor) -> torch.Tensor:
    """Continuous per-frame sigma [B] -> bucket index LongTensor [B]."""
    edges = torch.tensor(SIGMA_BUCKET_EDGES, device=sigma.device, dtype=torch.float32)
    return torch.bucketize(sigma.float(), edges, right=True)  # left-closed buckets


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
    """Sinusoidal embedding of t in [0, 1]; t shape [B] -> [B, dim].
    Preserves the input's floating dtype (float64 passes stay float64 —
    needed by the MeanFlow JVP-vs-finite-difference verification)."""
    half = dim // 2
    dtype = t.dtype if t.is_floating_point() else torch.float32
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=dtype) / half)
    ang = t.to(dtype)[:, None] * freqs[None, :] * 1000.0
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
        if cfg.dual_stream:
            # summed projections == concat + one Linear, without reshaping
            self.neg_proj = nn.Linear(cfg.d_model, w)
        if cfg.sigma_buckets > 0:
            self.sigma_emb = nn.Embedding(cfg.sigma_buckets, w)
            nn.init.zeros_(self.sigma_emb.weight)  # sigma info is a no-op at init
        if cfg.meanflow:
            self.r_mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([AdaLNBlock(w, cfg.ffn_ratio) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(w, affine=False)
        self.out_proj = nn.Linear(w, cfg.d_latent)
        nn.init.zeros_(self.out_proj.weight)  # zero function at init
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        neg_condition: torch.Tensor | None = None,
        sigma_bucket: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
    ):
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
        # missing/extra inputs are refused loudly in BOTH directions — the
        # silent-discard pattern is exactly what the 2026-08-15 CFG audit found
        if self.cfg.dual_stream != (neg_condition is not None):
            raise ValueError(
                f"dual_stream={self.cfg.dual_stream} but neg_condition "
                f"{'missing' if neg_condition is None else 'given'}"
            )
        if (self.cfg.sigma_buckets > 0) != (sigma_bucket is not None):
            raise ValueError(
                f"sigma_buckets={self.cfg.sigma_buckets} but sigma_bucket "
                f"{'missing' if sigma_bucket is None else 'given'}"
            )
        if self.cfg.meanflow != (r is not None):
            raise ValueError(
                f"meanflow={self.cfg.meanflow} but r {'missing' if r is None else 'given'}"
            )
        cond = self.cond_proj(condition) + self.time_mlp(timestep_embedding(t, self.cfg.width))
        if r is not None:
            if r.shape != t.shape:
                raise ValueError(f"r {tuple(r.shape)} != t {tuple(t.shape)}")
            cond = cond + self.r_mlp(timestep_embedding(r, self.cfg.width))
        if neg_condition is not None:
            if neg_condition.shape != condition.shape:
                raise ValueError(
                    f"neg_condition {tuple(neg_condition.shape)} != condition {tuple(condition.shape)}"
                )
            cond = cond + self.neg_proj(neg_condition)
        if sigma_bucket is not None:
            if sigma_bucket.shape != t.shape:
                raise ValueError(f"sigma_bucket {tuple(sigma_bucket.shape)} != t {tuple(t.shape)}")
            cond = cond + self.sigma_emb(sigma_bucket.long())
        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.final_norm(h))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BoundField:
    """Adapter binding a dual-stream head's extra inputs (neg_condition,
    sigma_bucket) so it presents the plain head(x_t, t, condition) interface
    the samplers in cfm.py expect. Same pattern as integration._CFGField.

    sigma_bucket: int constant for the whole call (inference feeds the head
    real, un-noised feedback -> bucket 0), or a per-row LongTensor.
    """

    def __init__(self, head: FlowHead, neg_condition: torch.Tensor | None, sigma_bucket=0):
        self.head = head
        self.cfg = head.cfg  # samplers/patches read .cfg.d_latent
        self.neg = neg_condition
        self.sigma_bucket = sigma_bucket

    def __call__(self, x_t, t, condition):
        sb = None
        if self.cfg.sigma_buckets > 0:
            sb = (
                self.sigma_bucket
                if torch.is_tensor(self.sigma_bucket)
                else torch.full_like(t, int(self.sigma_bucket), dtype=torch.long)
            )
        return self.head(x_t, t, condition, neg_condition=self.neg, sigma_bucket=sb)
