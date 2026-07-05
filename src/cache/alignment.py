"""Frame-alignment guard (CLAUDE.md hard constraint 3).

Cached hidden states [B, T, d_model] and acoustic latents [B, T, d_latent]
must share the same T. d_model comes from the checkpoint config and d_latent
from the sigma-VAE config at runtime -- never hardcoded. Every cached pair
must pass through assert_frame_aligned before being written to disk.
"""

import torch


def assert_frame_aligned(hidden: torch.Tensor, latent: torch.Tensor) -> None:
    if hidden.ndim != 3 or latent.ndim != 3:
        raise ValueError(f"expected [B, T, d] tensors, got {hidden.shape} and {latent.shape}")
    if hidden.shape[:2] != latent.shape[:2]:
        raise ValueError(
            f"frame misalignment: hidden {tuple(hidden.shape)} vs latent {tuple(latent.shape)} "
            "-- this is the April 7 failure signature, do not proceed"
        )
    if not torch.isfinite(hidden).all() or not torch.isfinite(latent).all():
        raise ValueError("non-finite values in cached pair -- refusing to write")
