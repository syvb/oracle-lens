import numpy as np
import torch

from oracle_lens.activations.store import ActivationStore
from oracle_lens.config import Config
from oracle_lens.corpus.positions import sample_positions_for_response


def _is_delim(token_id: int) -> bool:
    return token_id % 10 == 0


def test_positions_keyed_deterministic():
    cfg = Config()
    resp = list(range(1, 200))
    a = sample_positions_for_response("c1", 50, resp, _is_delim, cfg)
    b = sample_positions_for_response("c1", 50, resp, _is_delim, cfg)
    assert a == b
    c = sample_positions_for_response("c2", 50, resp, _is_delim, cfg)
    assert a != c


def test_positions_fields_consistent():
    cfg = Config()
    n_prompt = 50
    resp = list(range(1, 150))
    rows = sample_positions_for_response("c1", n_prompt, resp, _is_delim, cfg)
    assert rows  # non-empty
    total = n_prompt + len(resp)
    for r in rows:
        assert n_prompt <= r["pos"] < total
        assert r["token_id"] == resp[r["pos"] - n_prompt]
        assert r["is_delimiter"] == _is_delim(r["token_id"])
        assert r["n_following"] == total - r["pos"] - 1
    # sorted, unique
    ps = [r["pos"] for r in rows]
    assert ps == sorted(set(ps))


def test_positions_include_delimiters():
    cfg = Config()
    resp = [7] * 100 + [10] + [7] * 100  # single delimiter token
    rows = sample_positions_for_response("c9", 10, resp, _is_delim, cfg)
    assert any(r["is_delimiter"] for r in rows)


def test_positions_short_response():
    cfg = Config()
    rows = sample_positions_for_response("c1", 5, [3, 4], _is_delim, cfg)
    assert len(rows) == 2
    assert sample_positions_for_response("c1", 5, [], _is_delim, cfg) == []


def test_store_rejects_fp16_overflow(tmp_path):
    import pytest

    store = ActivationStore.create(tmp_path / "acts", 4, 8, meta={})
    bad = torch.ones(1, 8)
    bad[0, 3] = 70000.0  # beyond fp16 range
    with pytest.raises(ValueError, match="fp16"):
        store.write_rows(np.array([0]), bad)
    bad[0, 3] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        store.write_rows(np.array([0]), bad)


def test_store_round_trip(tmp_path):
    store = ActivationStore.create(tmp_path / "acts", 10, 8, meta={"layer_index": 2})
    vecs = torch.randn(4, 8)
    store.write_rows(np.array([1, 3, 5, 7]), vecs)
    store.flush()

    ro = ActivationStore.open(tmp_path / "acts")
    got = ro.read_rows([1, 3, 5, 7])
    assert torch.allclose(got, vecs, atol=1e-2)  # fp16 storage
    assert ro.read_rows([0]).abs().sum() == 0
    batches = list(ro.iter_batches(batch_size=4))
    assert [lo for lo, _ in batches] == [0, 4, 8]
    assert ActivationStore.meta(tmp_path / "acts")["layer_index"] == 2
