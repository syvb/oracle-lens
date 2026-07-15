"""Whitened fraction-of-variance-explained (PLAN.md §2).

The project's single FVE definition — training gates, teacher health checks,
GRPO rewards, and the §8.1 eval table must all import from here. Inputs are
assumed ALREADY whitened (see activations/whitening.py); the whitened corpus
mean is ~0, so the denominator is ||x||^2 (PLAN.md §2).

Two aggregations are reported because the paper doesn't disambiguate:
- per-example FVE, averaged ("mean")   — headline-comparable number
- pooled FVE over the whole set ("pooled") — 1 - sum(err^2)/sum(||x||^2)
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def fve_per_example(target_w: torch.Tensor, recon_w: torch.Tensor) -> torch.Tensor:
    """[B, d], [B, d] -> [B] whitened FVE per example."""
    if target_w.shape != recon_w.shape or target_w.ndim != 2:
        raise ValueError(
            f"bad shapes: target {tuple(target_w.shape)}, recon {tuple(recon_w.shape)}"
        )
    t = target_w.to(torch.float32)
    r = recon_w.to(torch.float32)
    err = (t - r).square().sum(dim=-1)
    denom = t.square().sum(dim=-1).clamp_min(_EPS)
    return 1.0 - err / denom


def fve_summary(target_w: torch.Tensor, recon_w: torch.Tensor) -> dict[str, float]:
    per_ex = fve_per_example(target_w, recon_w)
    t = target_w.to(torch.float32)
    r = recon_w.to(torch.float32)
    pooled = 1.0 - (t - r).square().sum() / t.square().sum().clamp_min(_EPS)
    return {"fve_mean": per_ex.mean().item(), "fve_pooled": pooled.item(), "n": t.shape[0]}
