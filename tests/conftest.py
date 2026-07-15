"""Shared fixtures: a tiny byte-level BPE tokenizer with a chat template and a
tiny random Qwen3 model — everything is built locally so the suite runs
offline on CPU."""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from oracle_lens.config import Config

_CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_TRAIN_TEXT = [
    "The quick brown fox jumps over the lazy dog. It was raining today.\n",
    "Activation: <concept>x</concept> List exactly 8 phrases of exactly 4 tokens.",
    "<text>hello world this is a phrase</text> <summary> the currency of Italy",
    "One phrase per line. What the model producing this activation is about",
    "to say or is considering. explanation danger recognition assessment 0123456789",
    "<explanation>\nsome phrases here\n</explanation> Phrase: paris france capital",
]


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=800,
        special_tokens=["<|pad|>", "<|im_start|>", "<|im_end|>", "㊗"],
        show_progress=False,
    )
    tok.train_from_iterator(_TRAIN_TEXT * 30, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<|pad|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    fast.chat_template = _CHAT_TEMPLATE
    out = tmp_path_factory.mktemp("tiny_tokenizer")
    fast.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_tokenizer_dir)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tiny_tokenizer):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    out = tmp_path_factory.mktemp("tiny_model")
    model.save_pretrained(out)
    return out


@pytest.fixture()
def cfg(tiny_tokenizer_dir) -> Config:
    c = Config()
    c.model.name = str(tiny_tokenizer_dir)
    c.model.d_model = 32
    c.model.layer_index = 2
    return c


@pytest.fixture(autouse=True)
def _isolated_injection_cache(tmp_path, monkeypatch):
    """Keep nla's injection-token cache out of the vendored tree during tests."""
    import nla.datagen.injection_tokens as it

    monkeypatch.setattr(it, "_CACHE_PATH", tmp_path / "injection_token_cache.yaml")
