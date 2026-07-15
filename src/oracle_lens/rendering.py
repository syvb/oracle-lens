"""THE single owner of subject-model chat rendering and sampling settings.

PLAN.md §3's critical consistency warning: the chat template, system prompt
(none), sampling settings, and thinking-mode flag used at GENERATION time must
exactly match ACTIVATION-COLLECTION time, or the on-policy assumption silently
breaks. Both stages therefore import from here and nowhere else, and the
sidecar records `template_fingerprint()` so any drift is detected loudly at
collection time.
"""

from __future__ import annotations

import hashlib
from typing import Any

from transformers import AutoTokenizer

from oracle_lens.config import Config


def load_subject_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def render_first_turn(tokenizer: Any, user_message: str, cfg: Config) -> list[int]:
    """Token IDs of the full generation prompt for a single first user turn:
    no system prompt, generation prompt appended, thinking mode per config."""
    kwargs: dict[str, Any] = {}
    # Qwen3's template takes enable_thinking; other templates would reject the
    # kwarg, so only pass it when the template knows it.
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template:
        kwargs["enable_thinking"] = cfg.model.enable_thinking
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,  # transformers >=5 defaults to True
        **kwargs,
    )
    return list(ids)


def sampling_params(cfg: Config) -> dict[str, Any]:
    """Sampling settings shared by generation and any regeneration checks."""
    return {
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "max_tokens": cfg.generation.max_new_tokens,
    }


def template_fingerprint(tokenizer: Any, cfg: Config) -> str:
    """Hash of everything that must be identical between generation and
    collection. Stored in the corpus sidecar; asserted before collection."""
    parts = [
        cfg.model.name,
        repr(tokenizer.chat_template),
        repr(sorted(tokenizer.get_vocab().items())[:100]),  # cheap vocab probe
        str(tokenizer.vocab_size),
        f"enable_thinking={cfg.model.enable_thinking}",
        "system_prompt=None",
        repr(sorted(sampling_params(cfg).items())),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()
