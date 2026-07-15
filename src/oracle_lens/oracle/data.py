"""Stage 4 SFT data: teacher decompositions -> (injected prompt, phrase list)
examples (PLAN.md §7.1-7.2).

Per example: K ~ U(k_min, k_max) keyed, capped by how many atoms the teacher
kept; the prompt states the ACTUAL K and the teacher's N; the target is the
teacher's first K phrases in teacher order. The activation rides along as a
store row; the training loop whitens + alpha-scales it per batch.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from nla.schema import NLATokenMeta

from oracle_lens.config import Config
from oracle_lens.oracle.prompts import format_target, render_oracle_prompt


@dataclass
class OracleExample:
    store_row: int
    k: int
    n: int
    phrases: list[str]


def build_oracle_examples(
    cfg: Config, teacher_path: Path, *, limit: int | None = None, min_phrases: int = 1
) -> list[OracleExample]:
    ocfg = cfg.oracle
    table = pq.read_table(
        teacher_path,
        columns=["store_row", "conversation_id", "pos", "n_tokens", "phrases"],
    )
    examples: list[OracleExample] = []
    for i in range(table.num_rows):
        phrases = table["phrases"][i].as_py()
        if len(phrases) < min_phrases:
            continue
        rng = random.Random(
            hashlib.sha256(
                f"{ocfg.seed}|k|{table['conversation_id'][i].as_py()}|{table['pos'][i].as_py()}".encode()
            ).digest()
        )
        k = min(rng.randint(ocfg.k_min, ocfg.k_max), len(phrases))
        examples.append(
            OracleExample(
                store_row=table["store_row"][i].as_py(),
                k=k,
                n=table["n_tokens"][i].as_py(),
                phrases=phrases[:k],
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


class OracleSftDataset(torch.utils.data.Dataset):
    """Prompt (with marker) + target tokens; labels mask the prompt."""

    def __init__(
        self, examples: list[OracleExample], tokenizer: Any, cfg: Config, meta: NLATokenMeta
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.meta = meta

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.examples[i]
        prompt_ids = render_oracle_prompt(
            self.tokenizer, self.cfg, self.meta, k=ex.k, n=ex.n
        )
        target = format_target(ex.phrases)
        response_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "store_row": ex.store_row,
        }


def collate_oracle(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(b["prompt_ids"]) + len(b["response_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["prompt_ids"] + b["response_ids"]
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
        labels[i, len(b["prompt_ids"]) : len(seq)] = torch.tensor(
            b["response_ids"], dtype=torch.long
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "store_rows": torch.tensor([b["store_row"] for b in batch], dtype=torch.long),
    }
