import torch

from oracle_lens.teacher.nnomp import (
    half_dictionary_mask,
    nn_omp,
    splitmix64,
)


def _py_splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def test_splitmix64_matches_reference():
    xs = torch.tensor([0, 1, 42, -1, 2**62, -(2**62)], dtype=torch.int64)
    got = splitmix64(xs)
    for x, g in zip(xs.tolist(), got.tolist()):
        assert g & ((1 << 64) - 1) == _py_splitmix64(x & ((1 << 64) - 1))


def test_half_mask_deterministic_and_balanced():
    seeds = torch.arange(100, dtype=torch.int64)
    m1 = half_dictionary_mask(seeds, 0, 10000)
    m2 = half_dictionary_mask(seeds, 0, 10000)
    assert torch.equal(m1, m2)
    # chunked generation must agree with one-shot
    chunked = torch.cat(
        [half_dictionary_mask(seeds, lo, lo + 2500) for lo in range(0, 10000, 2500)], dim=1
    )
    assert torch.equal(m1, chunked)
    density = m1.float().mean().item()
    assert 0.48 < density < 0.52
    # different seeds -> different masks
    assert not torch.equal(m1[0], m1[1])


def _sparse_problem(seed=0, m=200, d=64, k_true=3, bsz=16):
    g = torch.Generator().manual_seed(seed)
    dic = torch.randn(m, d, generator=g)
    dic = dic / dic.norm(dim=-1, keepdim=True)
    true_idx = torch.stack([torch.randperm(m, generator=g)[:k_true] for _ in range(bsz)])
    true_c = torch.rand(bsz, k_true, generator=g) + 0.5
    targets = torch.einsum("bk,bkd->bd", true_c, dic[true_idx])
    return dic, targets, true_idx


def test_recovers_sparse_ground_truth():
    dic, targets, true_idx = _sparse_problem()
    res = nn_omp(dic, targets, max_atoms=8, min_gain=1e-3, corr_chunk=64)
    assert res.final_fve.min() > 0.98
    for b in range(targets.shape[0]):
        chosen = set(res.indices[b, : res.n_selected[b]].tolist())
        assert set(true_idx[b].tolist()) <= chosen


def test_respects_half_dictionary_mask():
    dic, targets, _ = _sparse_problem(seed=1)
    seeds = torch.arange(targets.shape[0], dtype=torch.int64) + 7
    res = nn_omp(dic, targets, seeds=seeds, max_atoms=8, min_gain=1e-3, corr_chunk=97)
    allowed = half_dictionary_mask(seeds, 0, dic.shape[0])
    for b in range(targets.shape[0]):
        for idx in res.indices[b, : res.n_selected[b]].tolist():
            assert allowed[b, idx]


def test_min_gain_stops_early_and_orthogonal_target_selects_nothing():
    dic = torch.eye(8)[:4]  # atoms span dims 0-3 only
    target = torch.zeros(1, 8)
    target[0, 7] = 1.0  # orthogonal to every atom
    res = nn_omp(dic, target, max_atoms=4)
    assert res.n_selected[0] == 0
    assert res.final_fve[0] == 0.0

    # A target explained by one atom: greedy must stop after it.
    target2 = torch.zeros(1, 8)
    target2[0, 1] = 2.0
    res2 = nn_omp(dic, target2, max_atoms=4, min_gain=5e-3)
    assert res2.n_selected[0] == 1
    assert res2.final_fve[0] > 0.999
    assert res2.indices[0, 0] == 1


def test_deterministic_across_chunk_sizes():
    dic, targets, _ = _sparse_problem(seed=4)
    seeds = torch.full((targets.shape[0],), 123, dtype=torch.int64)
    a = nn_omp(dic, targets, seeds=seeds, max_atoms=6, corr_chunk=13)
    b = nn_omp(dic, targets, seeds=seeds, max_atoms=6, corr_chunk=200)
    assert torch.equal(a.indices, b.indices)
    assert torch.allclose(a.coeffs, b.coeffs, atol=1e-5)
