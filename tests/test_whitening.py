import torch

from oracle_lens.activations.whitening import (
    WhiteningTransform,
    fit_whitening,
    validate_whitening,
)


def _correlated_data(n: int, d: int, seed: int = 0) -> torch.Tensor:
    # One fixed population transform; `seed` varies only the samples drawn
    # from it, so different seeds give i.i.d. fit/holdout sets.
    g_pop = torch.Generator().manual_seed(1234)
    scale = torch.linspace(0.1, 10.0, d)
    mix = torch.randn(d, d, generator=g_pop) / d**0.5
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * scale @ mix + 3.0


def test_whitened_holdout_is_isotropic():
    d = 32
    fit = _correlated_data(20000, d, seed=0)
    holdout = _correlated_data(5000, d, seed=1)
    t = fit_whitening(fit, ridge_frac=1e-4)
    stats = validate_whitening(t, holdout)
    assert stats["mean_abs_mean"] < 0.05
    assert 0.9 < stats["var_mean"] < 1.1
    assert stats["var_min"] > 0.7
    assert stats["var_max"] < 1.4


def test_round_trip():
    d = 16
    x = _correlated_data(5000, d)
    t = fit_whitening(x)
    back = t.unwhiten(t.whiten(x[:100]))
    assert torch.allclose(back, x[:100], atol=1e-3, rtol=1e-3)


def test_streaming_matches_full():
    x = _correlated_data(9000, 8)
    full = fit_whitening(x)
    streamed = fit_whitening(iter(x.chunk(7)))
    assert full.n_samples == streamed.n_samples == 9000
    assert torch.allclose(full.w, streamed.w, atol=1e-5)
    assert torch.allclose(full.mu, streamed.mu, atol=1e-6)


def test_ridge_bounds_condition_number():
    # Rank-deficient data: without ridge Sigma is singular; with ridge the
    # transform must exist and have finite conditioning.
    g = torch.Generator().manual_seed(2)
    low_rank = torch.randn(4000, 4, generator=g) @ torch.randn(4, 12, generator=g)
    t = fit_whitening(low_rank, ridge_frac=1e-2)
    assert t.condition_number < 1e4
    assert torch.isfinite(t.w).all()


def test_save_load_round_trip(tmp_path):
    t = fit_whitening(_correlated_data(3000, 8))
    p = tmp_path / "whitening.pt"
    t.save(p)
    t2 = WhiteningTransform.load(p)
    x = torch.randn(10, 8)
    assert torch.allclose(t.whiten(x), t2.whiten(x))
    assert t2.n_samples == 3000
