"""Stage 2 data: (phrase, preceding-activation) pairs (PLAN.md §5).

Mechanically generated: at a sampled position, the target is the (whitened,
unit-normalized) activation there and the phrase is the N tokens that FOLLOW
it, N ~ U(n_min, n_max) keyed on (seed, conversation, pos). Zero labeling
cost — if held-out error hasn't plateaued, generate more.

Phrases are handed to the reconstructor as DECODED TEXT (the reconstructor
must learn arbitrary text -> direction, not just corpus token IDs).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch


@dataclass
class ReconPair:
    store_row: int  # row into the activation store (= positions.parquet row)
    phrase: str
    n_tokens: int
    is_delimiter: bool


def load_corpus_ids(corpus_path: Path) -> dict[str, list[int]]:
    table = pq.read_table(corpus_path, columns=["conversation_id", "prompt_ids", "response_ids"])
    return {
        c: list(p) + list(r)
        for c, p, r in zip(
            table["conversation_id"].to_pylist(),
            table["prompt_ids"].to_pylist(),
            table["response_ids"].to_pylist(),
        )
    }


def build_pairs(
    cfg: Any,
    corpus_path: Path,
    positions_path: Path,
    tokenizer: Any,
    *,
    split: str,
    fixed_n: int | None = None,
    limit: int | None = None,
) -> list[ReconPair]:
    """fixed_n=None samples N ~ U(n_min, min(n_max, available)) per example
    [paper]; a fixed N builds per-length eval buckets."""
    rcfg = cfg.reconstructor
    full_ids = load_corpus_ids(corpus_path)
    table = pq.read_table(
        positions_path,
        columns=["conversation_id", "pos", "n_following", "is_delimiter", "split"],
    )
    pairs: list[ReconPair] = []
    for row in range(table.num_rows):
        if table["split"][row].as_py() != split:
            continue
        n_following = table["n_following"][row].as_py()
        if n_following < (fixed_n or rcfg.n_min):
            continue
        conv_id = table["conversation_id"][row].as_py()
        pos = table["pos"][row].as_py()
        if fixed_n is None:
            rng = random.Random(
                hashlib.sha256(f"{rcfg.seed}|{conv_id}|{pos}".encode()).digest()
            )
            n = rng.randint(rcfg.n_min, min(rcfg.n_max, n_following))
        else:
            n = fixed_n
        phrase_ids = full_ids[conv_id][pos + 1 : pos + 1 + n]
        phrase = tokenizer.decode(phrase_ids)
        if not phrase.strip():
            continue
        pairs.append(
            ReconPair(
                store_row=row,
                phrase=phrase,
                n_tokens=n,
                is_delimiter=table["is_delimiter"][row].as_py(),
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


class ReconDataset(torch.utils.data.Dataset):
    """Tokenized reconstructor prompts; targets are looked up from the
    activation store by the training loop (store rows ride along)."""

    def __init__(self, pairs: list[ReconPair], tokenizer: Any, template: str) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.template = template

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        pair = self.pairs[i]
        ids = self.tokenizer(
            self.template.format(phrase=pair.phrase), add_special_tokens=False
        )["input_ids"]
        return {"input_ids": ids, "store_row": pair.store_row}


def collate_recon(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    last_index = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = 1
        last_index[i] = n - 1  # suffix-anchored extraction (nla convention)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "last_index": last_index,
        "store_rows": torch.tensor([b["store_row"] for b in batch], dtype=torch.long),
    }
