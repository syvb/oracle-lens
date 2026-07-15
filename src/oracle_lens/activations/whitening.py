"""Whitening transform for residual-stream activations (PLAN.md §2, §4).

Every reconstruction loss, teacher decomposition, reward, and FVE number in
this project lives in whitened coordinates: x_w = Sigma^{-1/2} (x - mu), with
ridge Sigma <- Sigma + lambda*I, lambda = ridge_frac * tr(Sigma)/d. This module
is the single owner of that transform; nothing else may whiten differently.

The subject model is ~4k-dimensional and the fit uses >=550k vectors, so
moments are accumulated streaming in float64 and the eigendecomposition is done
once in float64, then stored in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

DEFAULT_RIDGE_FRAC = 1e-2  # [choice] PLAN.md §2; sweep one order each way if teacher FVE is bad


class RunningMoments:
    """Streaming mean/covariance accumulator (float64)."""

    def __init__(self, dim: int, device: torch.device | str = "cpu") -> None:
        self.dim = dim
        self.count = 0
        self._sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self._outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, batch: torch.Tensor) -> None:
        if batch.ndim != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"expected [n, {self.dim}], got {tuple(batch.shape)}")
        b = batch.to(self._sum.device, torch.float64)
        self.count += b.shape[0]
        self._sum += b.sum(dim=0)
        self._outer += b.T @ b

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("no data accumulated")
        return self._sum / self.count

    def covariance(self) -> torch.Tensor:
        if self.count < 2:
            raise ValueError("need at least 2 samples")
        mu = self.mean()
        # E[xx^T] - mu mu^T, with the unbiased n/(n-1) correction.
        cov = self._outer / self.count - torch.outer(mu, mu)
        cov *= self.count / (self.count - 1)
        return (cov + cov.T) / 2  # enforce exact symmetry


@dataclass
class WhiteningTransform:
    """x_w = (x - mu) @ w  and  x = x_w @ w_inv + mu (w, w_inv symmetric)."""

    mu: torch.Tensor  # [d] float32
    w: torch.Tensor  # [d, d] float32, Sigma_ridged^{-1/2}
    w_inv: torch.Tensor  # [d, d] float32, Sigma_ridged^{1/2}
    ridge_frac: float
    ridge_lambda: float
    n_samples: int
    condition_number: float  # of Sigma_ridged

    @property
    def dim(self) -> int:
        return self.mu.shape[0]

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w.to(x.device, torch.float32)
        mu = self.mu.to(x.device, torch.float32)
        return (x.to(torch.float32) - mu) @ w

    def unwhiten(self, x_w: torch.Tensor) -> torch.Tensor:
        w_inv = self.w_inv.to(x_w.device, torch.float32)
        mu = self.mu.to(x_w.device, torch.float32)
        return x_w.to(torch.float32) @ w_inv + mu

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "mu": self.mu,
                "w": self.w,
                "w_inv": self.w_inv,
                "ridge_frac": self.ridge_frac,
                "ridge_lambda": self.ridge_lambda,
                "n_samples": self.n_samples,
                "condition_number": self.condition_number,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "WhiteningTransform":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**d)


def fit_whitening(
    batches: Iterable[torch.Tensor] | torch.Tensor,
    *,
    ridge_frac: float = DEFAULT_RIDGE_FRAC,
    device: torch.device | str = "cpu",
) -> WhiteningTransform:
    """Fit mu, Sigma^{+-1/2} from activation batches (streaming, float64)."""
    if isinstance(batches, torch.Tensor):
        batches = [batches]
    moments: RunningMoments | None = None
    for batch in batches:
        if moments is None:
            moments = RunningMoments(batch.shape[-1], device=device)
        moments.update(batch.reshape(-1, batch.shape[-1]))
    if moments is None:
        raise ValueError("no batches provided")

    d = moments.dim
    cov = moments.covariance()
    lam = ridge_frac * torch.trace(cov).item() / d
    cov_r = cov + lam * torch.eye(d, dtype=torch.float64, device=cov.device)

    evals, evecs = torch.linalg.eigh(cov_r)
    if evals[0] <= 0:
        raise RuntimeError(
            f"non-positive eigenvalue {evals[0].item():.3e} after ridge; increase ridge_frac"
        )
    w = (evecs * evals.rsqrt()) @ evecs.T
    w_inv = (evecs * evals.sqrt()) @ evecs.T

    return WhiteningTransform(
        mu=moments.mean().to(torch.float32).cpu(),
        w=w.to(torch.float32).cpu(),
        w_inv=w_inv.to(torch.float32).cpu(),
        ridge_frac=ridge_frac,
        ridge_lambda=lam,
        n_samples=moments.count,
        condition_number=(evals[-1] / evals[0]).item(),
    )


def validate_whitening(
    transform: WhiteningTransform, holdout: torch.Tensor
) -> dict[str, float]:
    """M1 sanity check (PLAN.md §4.2): whitened held-out data should have
    ~zero mean and ~unit variance per direction (slightly below 1 overall,
    since the ridge shrinks true directions)."""
    xw = transform.whiten(holdout)
    var = xw.var(dim=0)
    return {
        "mean_abs_mean": xw.mean(dim=0).abs().mean().item(),
        "var_mean": var.mean().item(),
        "var_min": var.min().item(),
        "var_max": var.max().item(),
        "condition_number": transform.condition_number,
    }
