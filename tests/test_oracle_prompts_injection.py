import math

import pytest
import torch

from oracle_lens.oracle.data import (
    OracleExample,
    OracleSftDataset,
    collate_oracle,
)
from oracle_lens.oracle.injection import (
    InjectionState,
    register_injection_hook,
    resolve_alpha,
    scale_for_injection,
)
from oracle_lens.oracle.prompts import (
    canonical_actor_template,
    format_target,
    oracle_token_meta,
    parse_phrases,
    render_oracle_prompt,
)


def test_canonical_template_keeps_injection_slot(cfg):
    t = canonical_actor_template(cfg)
    assert "{injection_char}" in t
    assert "{k}" not in t and "{n}" not in t


def test_token_meta_and_prompt_have_one_marker(cfg, tiny_tokenizer):
    meta = oracle_token_meta(tiny_tokenizer, cfg)
    ids = render_oracle_prompt(tiny_tokenizer, cfg, meta, k=8, n=4)
    assert ids.count(meta.injection_token_id) == 1
    p = ids.index(meta.injection_token_id)
    assert ids[p - 1] == meta.injection_left_neighbor_id
    assert ids[p + 1] == meta.injection_right_neighbor_id
    # K and N are actually stated in the prompt
    text = tiny_tokenizer.decode(ids)
    assert "8" in text and "4" in text


def test_format_parse_round_trip():
    phrases = ["the currency of", "Italy is the", "euro and it"]
    assert parse_phrases(format_target(phrases)) == phrases
    assert parse_phrases("no tags at all") == []
    assert parse_phrases("<explanation>\n\n</explanation>") == []


def test_alpha_resolution(cfg):
    assert resolve_alpha(cfg) == pytest.approx(math.sqrt(cfg.model.d_model))
    cfg.oracle.alpha = 123.0
    assert resolve_alpha(cfg) == 123.0
    v = torch.randn(5, cfg.model.d_model)
    scaled = scale_for_injection(v, 50.0)
    assert torch.allclose(scaled.norm(dim=-1), torch.full((5,), 50.0), atol=1e-4)


def test_injection_hook_replaces_marker_row(cfg, tiny_tokenizer, tiny_model_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    meta = oracle_token_meta(tiny_tokenizer, cfg)
    state = InjectionState()
    register_injection_hook(model, meta, state)

    ids_a = render_oracle_prompt(tiny_tokenizer, cfg, meta, k=4, n=2)
    ids_b = render_oracle_prompt(tiny_tokenizer, cfg, meta, k=6, n=8)
    max_len = max(len(ids_a), len(ids_b))
    batch = torch.full((2, max_len), tiny_tokenizer.pad_token_id, dtype=torch.long)
    batch[0, : len(ids_a)] = torch.tensor(ids_a)
    batch[1, : len(ids_b)] = torch.tensor(ids_b)

    vectors = torch.randn(2, model.config.hidden_size) * 7
    state.vectors = vectors
    embeds = model.get_input_embeddings()(batch)
    for i, ids in enumerate((ids_a, ids_b)):
        p = ids.index(meta.injection_token_id)
        assert torch.allclose(embeds[i, p], vectors[i].to(embeds.dtype))
        assert not torch.allclose(embeds[i, p - 1], vectors[i].to(embeds.dtype))

    # count mismatch must crash loudly (nla invariant)
    state.vectors = torch.randn(3, model.config.hidden_size)
    with pytest.raises(RuntimeError, match="injection sites"):
        model.get_input_embeddings()(batch)

    # hook is a no-op when no vectors are staged
    state.vectors = None
    plain = model.get_input_embeddings()(batch)
    assert plain.shape == embeds.shape


def test_oracle_dataset_and_collate(cfg, tiny_tokenizer):
    meta = oracle_token_meta(tiny_tokenizer, cfg)
    examples = [
        OracleExample(store_row=3, k=2, n=4, phrases=["alpha beta gamma delta", "one two three four"]),
        OracleExample(store_row=9, k=1, n=2, phrases=["hello world"]),
    ]
    ds = OracleSftDataset(examples, tiny_tokenizer, cfg, meta)
    batch = collate_oracle([ds[0], ds[1]], tiny_tokenizer.pad_token_id)

    assert batch["store_rows"].tolist() == [3, 9]
    for i in range(2):
        item = ds[i]
        n_prompt = len(item["prompt_ids"])
        n_resp = len(item["response_ids"])
        assert (batch["labels"][i, :n_prompt] == -100).all()
        assert (
            batch["labels"][i, n_prompt : n_prompt + n_resp]
            == batch["input_ids"][i, n_prompt : n_prompt + n_resp]
        ).all()
        assert (batch["labels"][i, n_prompt + n_resp :] == -100).all()
        assert batch["attention_mask"][i].sum() == n_prompt + n_resp
        # target parses back to the phrases
        target_text = tiny_tokenizer.decode(item["response_ids"], skip_special_tokens=True)
        assert parse_phrases(target_text) == examples[i].phrases
