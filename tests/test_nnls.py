import numpy as np
import pytest
import torch
from scipy.optimize import nnls as scipy_nnls

from oracle_lens.eval.fve import fve_per_example, fve_summary
from oracle_lens.nnls import nnls_batched, reconstruct


@pytest.mark.parametrize("seed", range(4))
def test_matches_scipy(seed):
    g = torch.Generator().manual_seed(seed)
    bsz, k, d = 32, 12, 48
    atoms = torch.randn(bsz, k, d, generator=g)
    atoms = atoms / atoms.norm(dim=-1, keepdim=True)
    targets = torch.randn(bsz, d, generator=g)

    ours = nnls_batched(atoms, targets, max_iters=3000, tol=1e-10)
    for i in range(bsz):
        ref, _ = scipy_nnls(atoms[i].numpy().T.astype(np.float64), targets[i].numpy().astype(np.float64))
        # Coefficients can be non-unique when atoms correlate; compare the
        # objective, which is what NN-OMP consumes.
        obj_ours = (atoms[i].T @ ours[i] - targets[i]).square().sum().item()
        obj_ref = float(((atoms[i].numpy().T @ ref - targets[i].numpy()) ** 2).sum())
        assert obj_ours <= obj_ref + 1e-4
        assert (ours[i] >= 0).all()


def test_zero_padded_atoms_get_zero_coefficients():
    g = torch.Generator().manual_seed(0)
    atoms = torch.randn(4, 6, 16, generator=g)
    atoms[:, 4:] = 0.0
    c = nnls_batched(atoms, torch.randn(4, 16, generator=g))
    assert (c[:, 4:] == 0).all()


def test_exact_recovery_of_nonneg_combination():
    g = torch.Generator().manual_seed(1)
    atoms = torch.randn(8, 5, 64, generator=g)
    true_c = torch.rand(8, 5, generator=g) * 2.0
    targets = reconstruct(atoms, true_c)
    c = nnls_batched(atoms, targets, max_iters=5000, tol=1e-12)
    assert torch.allclose(c, true_c, atol=1e-3)
    assert fve_per_example(targets, reconstruct(atoms, c)).min() > 0.999


def test_fve_definitions():
    t = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    assert torch.allclose(fve_per_example(t, t), torch.ones(2))
    zeros = torch.zeros_like(t)
    assert torch.allclose(fve_per_example(t, zeros), torch.zeros(2))
    half = fve_summary(t, t * 0.5)
    # per-example FVE of 0.5-scaled recon is 1 - 0.25 = 0.75 for every row
    assert abs(half["fve_mean"] - 0.75) < 1e-6
    assert abs(half["fve_pooled"] - 0.75) < 1e-6
    assert half["n"] == 2
