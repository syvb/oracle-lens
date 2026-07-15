"""Batched non-negative least squares (shared by teacher NN-OMP, oracle decode,
and every eval baseline — PLAN.md §6.2, §7, §8.1).

Solves, independently per batch element b:

    min_{c >= 0}  || A_b^T c - x_b ||^2      A_b: [K, d] atoms, x_b: [d]

via FISTA (accelerated projected gradient) on the normal equations. K is small
(<= 32 atoms) so the per-iteration cost is a [B, K, K] bmm — negligible next to
the dictionary-correlation matmuls around it. Zero-padded atom rows are safe:
a zero atom has zero gradient and keeps coefficient 0, so variable atom counts
are handled by padding.
"""

from __future__ import annotations

import torch


def nnls_batched(
    atoms: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iters: int = 600,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Return non-negative coefficients [B, K] for atoms [B, K, d], targets [B, d].

    Deterministic; runs in float32 internally regardless of input dtype.
    """
    if atoms.ndim != 3 or targets.ndim != 2 or atoms.shape[0] != targets.shape[0]:
        raise ValueError(f"bad shapes: atoms {tuple(atoms.shape)}, targets {tuple(targets.shape)}")
    a = atoms.to(torch.float32)
    x = targets.to(torch.float32)

    gram = a @ a.mT  # [B, K, K]
    b = (a @ x.unsqueeze(-1)).squeeze(-1)  # [B, K]

    # Lipschitz constant of the gradient = lambda_max(gram); float64 eigh on
    # K x K for stability. Guard fully-padded batches with a floor.
    lip = torch.linalg.eigvalsh(gram.to(torch.float64))[..., -1].to(torch.float32)
    lip = lip.clamp_min(1e-12).unsqueeze(-1)  # [B, 1]

    c = torch.zeros_like(b)
    y = c
    t = 1.0
    for i in range(max_iters):
        grad = (gram @ y.unsqueeze(-1)).squeeze(-1) - b
        c_next = (y - grad / lip).clamp_min(0.0)
        t_next = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        y = c_next + ((t - 1.0) / t_next) * (c_next - c)
        converged = i % 25 == 24 and (c_next - c).abs().max().item() < tol
        c, t = c_next, t_next
        if converged:
            break
    return c


def reconstruct(atoms: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """[B, K, d], [B, K] -> [B, d]."""
    return (coeffs.unsqueeze(1) @ atoms.to(coeffs.dtype)).squeeze(1)


def nnls_refit_fve(
    atoms: torch.Tensor, targets_w: torch.Tensor, **nnls_kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """The standard decode path [paper]: NNLS-refit coefficients of the given
    (whitened, unit-norm) atoms against whitened targets, return
    (coeffs [B, K], whitened FVE [B]). Shared by oracle decode and every
    §8.1 baseline so the metric can't fork."""
    from oracle_lens.eval.fve import fve_per_example

    coeffs = nnls_batched(atoms, targets_w, **nnls_kwargs)
    return coeffs, fve_per_example(targets_w, reconstruct(atoms, coeffs))
