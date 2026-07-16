"""Stage 0d: position sampling over assistant spans (PLAN.md §3).

~10 uniform positions per response plus a few DELIMITER positions (sentence-
final periods, newlines, end-of-turn) — the paper found delimiters carry
qualitatively different, commentary-like content, and §8.2's delimiter scan
depends on having them flagged.

Reproducibility follows the vendored nla convention: RNG keyed on
(seed, conversation_id), so the same conversation yields the same positions
regardless of ordering, sharding, or chunk size.

A row's `pos` is an absolute index into prompt_ids + response_ids; the target
activation lives at `pos`, and the phrase starting there is tokens
pos+1 .. pos+N (`n_following` bounds N).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from oracle_lens.config import Config
from oracle_lens.rendering import load_subject_tokenizer

POSITIONS_SCHEMA = pa.schema(
    [
        ("conversation_id", pa.string()),
        ("pos", pa.int32()),  # absolute index into prompt_ids + response_ids
        ("token_id", pa.int32()),
        ("is_delimiter", pa.bool_()),
        ("n_prompt_tokens", pa.int32()),
        ("n_following", pa.int32()),  # response tokens available after pos
    ]
)

# [choice] Broader than the plan's "periods, newlines, end-of-turn" wording:
# any sentence-final punctuation plus every special token counts as a
# delimiter. Report this deviation (PLAN.md §11).
_DELIMITER_TAILS = (".", "\n", "!", "?", ":", ";")


class DelimiterOracle:
    """Caches per-token-id delimiter-ness (decode is slow; vocab is finite)."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._cache: dict[int, bool] = {
            tid: True for tid in tokenizer.all_special_ids  # e.g. <|im_end|>
        }

    def __call__(self, token_id: int) -> bool:
        hit = self._cache.get(token_id)
        if hit is None:
            text = self._tokenizer.decode([token_id])
            hit = text.rstrip(" ").endswith(_DELIMITER_TAILS)
            self._cache[token_id] = hit
        return hit


def _rng_for(seed: int, conversation_id: str) -> random.Random:
    return random.Random(
        hashlib.sha256(f"{seed}|{conversation_id}".encode()).digest()
    )


def sample_positions_for_response(
    conversation_id: str,
    n_prompt_tokens: int,
    response_ids: list[int],
    is_delim: DelimiterOracle | Any,
    cfg: Config,
) -> list[dict[str, Any]]:
    """Pure per-conversation sampler (unit-tested; the file-level driver just
    maps it over the corpus)."""
    pcfg = cfg.positions
    n_resp = len(response_ids)
    if n_resp == 0:
        return []
    rng = _rng_for(pcfg.seed, conversation_id)

    span = list(range(n_prompt_tokens, n_prompt_tokens + n_resp))
    chosen = set(rng.sample(span, k=min(pcfg.per_response, n_resp)))
    delim_positions = [p for p in span if is_delim(response_ids[p - n_prompt_tokens])]
    if delim_positions:
        chosen.update(
            rng.sample(delim_positions, k=min(pcfg.delimiter_extra, len(delim_positions)))
        )

    total = n_prompt_tokens + n_resp
    return [
        {
            "conversation_id": conversation_id,
            "pos": p,
            "token_id": response_ids[p - n_prompt_tokens],
            "is_delimiter": is_delim(response_ids[p - n_prompt_tokens]),
            "n_prompt_tokens": n_prompt_tokens,
            "n_following": total - (p + 1),
        }
        for p in sorted(chosen)
    ]


def build_positions(cfg: Config, corpus_path: Path, out_path: Path) -> int:
    tokenizer = load_subject_tokenizer(cfg.model.name)
    is_delim = DelimiterOracle(tokenizer)
    rows: list[dict[str, Any]] = []
    pf = pq.ParquetFile(corpus_path)
    for batch in pf.iter_batches(
        batch_size=1024, columns=["conversation_id", "prompt_ids", "response_ids"]
    ):
        d = batch.to_pydict()
        for conv_id, p_ids, r_ids in zip(
            d["conversation_id"], d["prompt_ids"], d["response_ids"]
        ):
            rows.extend(
                sample_positions_for_response(conv_id, len(p_ids), r_ids, is_delim, cfg)
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=POSITIONS_SCHEMA)
    pq.write_table(table, out_path)
    n_delim = sum(r["is_delimiter"] for r in rows)
    print(f"positions: {len(rows)} sampled ({n_delim} delimiter-flagged) -> {out_path}")
    return len(rows)
