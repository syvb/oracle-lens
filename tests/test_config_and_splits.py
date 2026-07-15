import pytest
import yaml

from oracle_lens.config import Config, dump_config, load_config
from oracle_lens.corpus.splits import SPLIT_NAMES, split_for_conversation


def test_config_round_trip(tmp_path):
    cfg = Config()
    p = tmp_path / "c.yaml"
    p.write_text(dump_config(cfg))
    assert load_config(p) == cfg


def test_config_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"run_name": "x", "model": {"layer_index": 20}}))
    cfg = load_config(p)
    assert cfg.run_name == "x"
    assert cfg.model.layer_index == 20
    assert cfg.model.name == "Qwen/Qwen3-8B"  # untouched default
    assert cfg.dictionary.lengths == (2, 4, 8, 16, 32)  # tuple survives


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"model": {"layre_index": 20}}))
    with pytest.raises(KeyError, match="layre_index"):
        load_config(p)


def test_config_list_becomes_tuple(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"dictionary": {"lengths": [2, 4]}}))
    assert load_config(p).dictionary.lengths == (2, 4)


def test_splits_deterministic_and_proportional():
    cfg = Config()
    ids = [f"conv-{i}" for i in range(20000)]
    labels = [split_for_conversation(cfg, c) for c in ids]
    assert labels == [split_for_conversation(cfg, c) for c in ids]  # deterministic

    counts = {name: labels.count(name) for name in SPLIT_NAMES}
    s = cfg.splits
    total = s.reconstructor + s.dictionary + s.teacher + s.rl + s.eval
    for name, target in zip(
        SPLIT_NAMES, (s.reconstructor, s.dictionary, s.teacher, s.rl, s.eval)
    ):
        expected = len(ids) * target / total
        assert abs(counts[name] - expected) < 4 * (expected**0.5) + 30, (name, counts)


def test_splits_change_with_seed():
    cfg_a, cfg_b = Config(), Config()
    cfg_b.splits.seed = 99
    ids = [f"conv-{i}" for i in range(500)]
    a = [split_for_conversation(cfg_a, c) for c in ids]
    b = [split_for_conversation(cfg_b, c) for c in ids]
    assert a != b
