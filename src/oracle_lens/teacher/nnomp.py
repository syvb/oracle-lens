"""Non-negative orthogonal matching pursuit — the teacher decomposition of
PLAN.md §6.2. No open-source analog exists (this stage is the novel part of
the oracle-lens method), so it is written here from scratch as batched torch.

Given a dictionary of unit-normalized whitened phrase directions and a batch
of whitened target activations, greedily: pick the allowed atom with the
largest positive correlation to the residual, NNLS-refit all selected
coefficients (oracle_lens.nnls), subtract, repeat. Stop at ``max_atoms``
[paper] or when the marginal whitened-FVE gain falls below ``min_gain``
[choice]; an atom that fails the gain test is dropped, not kept.

The paper's "restrict each decomposition to a random half of the dictionary"
step is implemented storage-free with a splitmix64 hash of (example_seed,
atom_index) — each atom is allowed with p=1/2 independently, which realizes
the paper's intent (prevent the distilled oracle from memorizing one fixed
ranking) without materializing [n_examples, dict_size] masks. Fully
deterministic given the seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from oracle_lens.eval.fve import fve_per_example
from oracle_lens.nnls import nnls_batched, reconstruct

MAX_ATOMS_DEFAULT = 16  # [paper]
MIN_GAIN_DEFAULT = 5e-3  # [choice] marginal-FVE stop, PLAN.md §6.2

_SM64_GAMMA = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
_SM64_MUL1 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_SM64_MUL2 = -7723592293110705685  # 0x94D049BB133111EB


def _lsr(z: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift on int64 (torch's >> is arithmetic)."""
    return (z >> k) & ((1 << (64 - k)) - 1)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    """Vectorized splitmix64 finalizer on int64 (wrapping arithmetic)."""
    z = x + _SM64_GAMMA
    z = (z ^ _lsr(z, 30)) * _SM64_MUL1
    z = (z ^ _lsr(z, 27)) * _SM64_MUL2
    return z ^ _lsr(z, 31)


def half_dictionary_mask(
    seeds: torch.Tensor, atom_lo: int, atom_hi: int
) -> torch.Tensor:
    """Bool [len(seeds), atom_hi - atom_lo]: atom allowed for this example?

    Deterministic in (seed, absolute atom index); each atom allowed with
    p=1/2. Seeds must be int64.
    """
    atoms = torch.arange(atom_lo, atom_hi, dtype=torch.int64, device=seeds.device)
    h = splitmix64(seeds.unsqueeze(1) ^ splitmix64(atoms).unsqueeze(0))
    return (h & 1).to(torch.bool)


@dataclass
class OmpResult:
    """Per example: selected atom indices (-1 padded), NNLS coefficients
    (0 padded), cumulative whitened FVE after each kept atom (last value
    padded), and the number of atoms kept."""

    indices: torch.Tensor  # [B, max_atoms] int64
    coeffs: torch.Tensor  # [B, max_atoms] float32
    fve_path: torch.Tensor  # [B, max_atoms] float32
    n_selected: torch.Tensor  # [B] int64

    @property
    def final_fve(self) -> torch.Tensor:
        return self.fve_path.gather(
            1, (self.n_selected - 1).clamp_min(0).unsqueeze(1)
        ).squeeze(1) * (self.n_selected > 0)


def _chunked_masked_argmax(
    residual: torch.Tensor,
    dictionary: torch.Tensor,
    seeds: torch.Tensor | None,
    selected: torch.Tensor,
    n_selected: torch.Tensor,
    corr_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best positive-correlation allowed atom per example: (values, indices).

    Chunks the dictionary axis so [B, M] never fully materializes; the
    half-dictionary mask is regenerated per chunk from the seeds.
    """
    bsz = residual.shape[0]
    best_val = torch.full((bsz,), float("-inf"), device=residual.device)
    best_idx = torch.full((bsz,), -1, dtype=torch.int64, device=residual.device)
    for lo in range(0, dictionary.shape[0], corr_chunk):
        hi = min(lo + corr_chunk, dictionary.shape[0])
        corr = residual @ dictionary[lo:hi].T  # [B, hi-lo]
        if seeds is not None:
            corr = corr.masked_fill(~half_dictionary_mask(seeds, lo, hi), float("-inf"))
        # Mask out atoms this example has already selected.
        in_chunk = (selected >= lo) & (selected < hi)
        if in_chunk.any():
            local = (selected - lo).clamp(0, hi - lo - 1)
            rows = torch.nonzero(in_chunk, as_tuple=True)
            corr[rows[0], local[rows]] = float("-inf")
        val, idx = corr.max(dim=1)
        better = val > best_val
        best_val = torch.where(better, val, best_val)
        best_idx = torch.where(better, idx + lo, best_idx)
    return best_val, best_idx


def nn_omp(
    dictionary: torch.Tensor,
    targets: torch.Tensor,
    *,
    seeds: torch.Tensor | None = None,
    max_atoms: int = MAX_ATOMS_DEFAULT,
    min_gain: float = MIN_GAIN_DEFAULT,
    corr_chunk: int = 65536,
    nnls_iters: int = 600,
) -> OmpResult:
    """Batched NN-OMP of whitened targets [B, d] against a unit-normalized
    dictionary [M, d]. ``seeds`` (int64 [B]) enables the per-example
    half-dictionary restriction; None uses the full dictionary."""
    if dictionary.ndim != 2 or targets.ndim != 2 or dictionary.shape[1] != targets.shape[1]:
        raise ValueError(
            f"bad shapes: dictionary {tuple(dictionary.shape)}, targets {tuple(targets.shape)}"
        )
    dev = targets.device
    dic = dictionary.to(dev, torch.float32)
    x = targets.to(torch.float32)
    bsz = x.shape[0]

    indices = torch.full((bsz, max_atoms), -1, dtype=torch.int64, device=dev)
    coeffs = torch.zeros(bsz, max_atoms, device=dev)
    fve_path = torch.zeros(bsz, max_atoms, device=dev)
    n_selected = torch.zeros(bsz, dtype=torch.int64, device=dev)
    active = torch.ones(bsz, dtype=torch.bool, device=dev)
    residual = x.clone()
    prev_fve = torch.zeros(bsz, device=dev)

    for k in range(max_atoms):
        best_val, best_idx = _chunked_masked_argmax(
            residual, dic, seeds, indices, n_selected, corr_chunk
        )
        # No allowed atom positively correlates -> those examples are done.
        active &= best_val > 0
        if not active.any():
            break

        rows = torch.nonzero(active, as_tuple=True)[0]
        trial_idx = indices[rows].clone()
        trial_idx[:, k] = best_idx[rows]

        atoms = dic[trial_idx.clamp_min(0)] * (trial_idx >= 0).unsqueeze(-1)  # [b, K, d]
        c = nnls_batched(atoms, x[rows], max_iters=nnls_iters)
        recon = reconstruct(atoms, c)
        new_fve = fve_per_example(x[rows], recon)

        kept = new_fve - prev_fve[rows] >= min_gain
        keep_rows = rows[kept]
        if keep_rows.numel():
            indices[keep_rows, k] = best_idx[keep_rows]
            coeffs[keep_rows, :] = 0.0
            coeffs[keep_rows, : k + 1] = c[kept][:, : k + 1]
            fve_path[keep_rows, k] = new_fve[kept]
            n_selected[keep_rows] = k + 1
            prev_fve[keep_rows] = new_fve[kept]
            residual[keep_rows] = x[keep_rows] - recon[kept]
        # Examples whose trial atom added < min_gain stop, dropping the atom.
        active[rows[~kept]] = False
        if not active.any():
            break

    # Pad fve_path beyond n_selected with the final value (plot-friendly).
    steps = torch.arange(max_atoms, device=dev).unsqueeze(0)
    fve_path = torch.where(
        steps < n_selected.unsqueeze(1).clamp_min(1),
        fve_path,
        prev_fve.unsqueeze(1),
    )
    return OmpResult(indices=indices, coeffs=coeffs, fve_path=fve_path, n_selected=n_selected)
