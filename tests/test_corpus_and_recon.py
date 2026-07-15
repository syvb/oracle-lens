"""Corpus verify gate, reconstructor pairs, and the truncated critic model —
all against the tiny local tokenizer/model."""

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from oracle_lens.corpus.generate import CORPUS_SCHEMA, write_corpus_sidecar
from oracle_lens.corpus.verify import verify_token_identity
from oracle_lens.rendering import (
    load_subject_tokenizer,
    render_first_turn,
    template_fingerprint,
)


def _write_tiny_corpus(cfg, tokenizer, path, n=8, corrupt_row=None):
    rows = {k: [] for k in CORPUS_SCHEMA.names}
    for i in range(n):
        prompt = f"tell me about thing number {i}"
        prompt_ids = render_first_turn(tokenizer, prompt, cfg)
        response_ids = tokenizer(
            f"It was raining today. The quick brown fox number {i}.\nDone.",
            add_special_tokens=False,
        )["input_ids"]
        if corrupt_row == i:
            prompt_ids = prompt_ids + [prompt_ids[-1]]
        rows["conversation_id"].append(f"conv-{i}")
        rows["prompt"].append(prompt)
        rows["prompt_ids"].append(prompt_ids)
        rows["response_ids"].append(response_ids)
        rows["finish_reason"].append("stop")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows, schema=CORPUS_SCHEMA), path)
    write_corpus_sidecar(path, cfg, tokenizer, n)
    return path


def test_verify_passes_on_faithful_corpus(cfg, tiny_tokenizer, tmp_path):
    path = _write_tiny_corpus(cfg, tiny_tokenizer, tmp_path / "corpus.parquet")
    report = verify_token_identity(cfg, path, n_samples=8)
    assert report.passed
    assert report.response_roundtrip_failures == 0


def test_verify_catches_prompt_drift(cfg, tiny_tokenizer, tmp_path):
    path = _write_tiny_corpus(
        cfg, tiny_tokenizer, tmp_path / "corpus.parquet", corrupt_row=3
    )
    report = verify_token_identity(cfg, path, n_samples=8)
    assert not report.passed
    assert report.prompt_mismatches == 1


def test_fingerprint_detects_drift(cfg, tiny_tokenizer, tmp_path):
    import pytest

    from oracle_lens.corpus.generate import assert_fingerprint_matches

    path = _write_tiny_corpus(cfg, tiny_tokenizer, tmp_path / "corpus.parquet")
    assert_fingerprint_matches(path, cfg, tiny_tokenizer)  # no drift -> ok
    cfg.generation.temperature = 0.9  # sampling drift
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        assert_fingerprint_matches(path, cfg, tiny_tokenizer)


def test_fingerprint_stable_across_loads(cfg, tiny_tokenizer_dir):
    a = load_subject_tokenizer(str(tiny_tokenizer_dir))
    b = load_subject_tokenizer(str(tiny_tokenizer_dir))
    from oracle_lens.config import Config

    cfg2 = Config()
    cfg2.model.name = cfg.model.name
    assert template_fingerprint(a, cfg) == template_fingerprint(b, cfg2)


def test_recon_pairs_keyed_and_bucketed(cfg, tiny_tokenizer, tmp_path):
    from oracle_lens.corpus.positions import build_positions
    from oracle_lens.corpus.splits import assign_splits
    from oracle_lens.reconstructor.data import build_pairs

    corpus = _write_tiny_corpus(cfg, tiny_tokenizer, tmp_path / "corpus.parquet", n=30)
    positions = tmp_path / "positions.parquet"
    cfg.model.name = str(tiny_tokenizer.name_or_path)
    build_positions(cfg, corpus, positions)
    assign_splits(cfg, positions)

    # every split label present in this tiny corpus's positions?  not
    # guaranteed — pick one that is.
    split = pq.read_table(positions, columns=["split"])["split"].to_pylist()[0]
    pairs_a = build_pairs(cfg, corpus, positions, tiny_tokenizer, split=split)
    pairs_b = build_pairs(cfg, corpus, positions, tiny_tokenizer, split=split)
    assert [(p.store_row, p.phrase, p.n_tokens) for p in pairs_a] == [
        (p.store_row, p.phrase, p.n_tokens) for p in pairs_b
    ]
    assert pairs_a and all(1 <= p.n_tokens <= 32 for p in pairs_a)

    fixed = build_pairs(cfg, corpus, positions, tiny_tokenizer, split=split, fixed_n=2)
    assert fixed and all(p.n_tokens == 2 for p in fixed)
    # a fixed-N phrase decodes exactly the 2 tokens after pos
    from oracle_lens.reconstructor.data import load_corpus_ids

    full = load_corpus_ids(corpus)
    table = pq.read_table(positions)
    p0 = fixed[0]
    conv = table["conversation_id"][p0.store_row].as_py()
    pos = table["pos"][p0.store_row].as_py()
    assert p0.phrase == tiny_tokenizer.decode(full[conv][pos + 1 : pos + 3])


def test_truncated_critic_forward(tiny_model_dir, tiny_tokenizer):
    from oracle_lens.reconstructor.data import collate_recon
    from oracle_lens.reconstructor.model import (
        extract_at,
        load_reconstructor,
        recon_loss,
        unit,
    )

    model = load_reconstructor(tiny_model_dir, layer_index=2, dtype=torch.float32)
    assert model.config.num_hidden_layers == 3  # blocks 0..2 inclusive

    items = [
        {
            "input_ids": tiny_tokenizer("<text>hello world</text> <summary>")["input_ids"],
            "store_row": 0,
        },
        {
            "input_ids": tiny_tokenizer("<text>a much longer phrase here</text> <summary>")[
                "input_ids"
            ],
            "store_row": 1,
        },
    ]
    batch = collate_recon(items, tiny_tokenizer.pad_token_id)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    pred = extract_at(out.values, batch["last_index"])
    assert pred.shape == (2, model.config.hidden_size)

    target = torch.randn(2, model.config.hidden_size)
    loss = recon_loss(pred, target)
    assert 0 <= loss.item() <= 4 + 1e-5  # 2(1-cos) in [0, 4]
    assert torch.allclose(
        recon_loss(target, target), torch.tensor(0.0), atol=1e-10
    )
    assert torch.allclose(unit(target).norm(dim=-1), torch.ones(2), atol=1e-5)
