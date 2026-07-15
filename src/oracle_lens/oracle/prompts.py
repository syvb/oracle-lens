"""Stage 4 prompt machinery (PLAN.md §7.1).

Reuses the vendored nla marker-token stack: auto-picked single-token CJK
marker (find_injection_token), the NLATokenMeta contract, and <explanation>
tags for robust parsing of generated phrase lists. The oracle prompt is fixed
and boring, with K and N explicit [paper].

One deliberate divergence: neighbor IDs are computed HERE via the exact same
render path the oracle prompts use (render_oracle_prompt), not via
nla.schema.compute_canonical_neighbors — the vendored helper was written for
transformers 4.x (apply_chat_template returned a list; >=5 returns a dict by
default), and deriving neighbors from our own render guarantees meta/prompt
consistency by construction.
"""

from __future__ import annotations

from typing import Any

from nla.datagen.injection_tokens import find_injection_token
from nla.schema import NLATokenMeta, extract_explanation, wrap_explanation

from oracle_lens.config import Config

__all__ = ["oracle_token_meta", "render_oracle_prompt", "format_target", "parse_phrases"]


def canonical_actor_template(cfg: Config) -> str:
    """The oracle template with representative K/N filled in, {injection_char}
    left open (the sidecar-template shape nla expects). K and N are plain
    digits far from the marker, so neighbors don't depend on them."""
    return cfg.oracle.prompt_template.format(k=8, n=4, injection_char="{injection_char}")


def oracle_token_meta(tokenizer: Any, cfg: Config) -> NLATokenMeta:
    inj_char, inj_id = find_injection_token(tokenizer)
    meta = NLATokenMeta(
        injection_char=inj_char,
        injection_token_id=inj_id,
        injection_left_neighbor_id=-1,  # filled below from a real render
        injection_right_neighbor_id=-1,
    )
    ids = render_oracle_prompt(tokenizer, cfg, meta, k=8, n=4)
    matches = [i for i, tid in enumerate(ids) if tid == inj_id]
    assert len(matches) == 1, (
        f"injection token {inj_char!r} (id {inj_id}) appears {len(matches)}x in the "
        f"canonical oracle prompt (expected exactly 1): {cfg.oracle.prompt_template!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1, "injection token at sequence edge"
    meta.injection_left_neighbor_id = ids[p - 1]
    meta.injection_right_neighbor_id = ids[p + 1]
    return meta


def render_oracle_prompt(
    tokenizer: Any, cfg: Config, meta: NLATokenMeta, *, k: int, n: int
) -> list[int]:
    """Token IDs of the injection prompt (user turn + generation prompt)."""
    content = cfg.oracle.prompt_template.format(
        injection_char=meta.injection_char, k=k, n=n
    )
    kwargs: dict[str, Any] = {}
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template:
        kwargs["enable_thinking"] = cfg.model.enable_thinking
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,  # transformers >=5 defaults to True
            **kwargs,
        )
    )


def format_target(phrases: list[str]) -> str:
    """SFT target: teacher phrases, teacher order, newline-separated, inside
    <explanation> tags (nla's parse-safe convention)."""
    return wrap_explanation("\n".join(phrases))


def parse_phrases(generated: str) -> list[str]:
    """Inverse of format_target, for decode/eval/reward."""
    payload = extract_explanation(generated)
    if payload is None:
        return []
    return [line.strip() for line in payload.splitlines() if line.strip()]
