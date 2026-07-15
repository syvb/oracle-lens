"""Stage 0e: the five disjoint-by-conversation splits (PLAN.md §3).

Split assignment is a pure function of (seed, conversation_id): the id hashes
to a uniform float which falls into buckets sized proportionally to the
target counts. Conversation-level (never position-level) so no conversation
leaks across splits — the same invariant the vendored nla repo enforces
document-level.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from oracle_lens.config import Config

SPLIT_NAMES = ("reconstructor", "dictionary", "teacher", "rl", "eval")


def split_for_conversation(cfg: Config, conversation_id: str) -> str:
    s = cfg.splits
    targets = [s.reconstructor, s.dictionary, s.teacher, s.rl, s.eval]
    total = sum(targets)
    digest = hashlib.sha256(f"{s.seed}|{conversation_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    acc = 0.0
    for name, t in zip(SPLIT_NAMES, targets):
        acc += t / total
        if u < acc:
            return name
    return SPLIT_NAMES[-1]


def assign_splits(cfg: Config, positions_path: Path) -> dict[str, int]:
    """Add/overwrite the `split` column on positions.parquet; returns counts."""
    table = pq.read_table(positions_path)
    conv_ids = table["conversation_id"].to_pylist()
    labels = [split_for_conversation(cfg, c) for c in conv_ids]
    if "split" in table.column_names:
        table = table.drop_columns(["split"])
    table = table.append_column("split", pa.array(labels, pa.string()))
    pq.write_table(table, positions_path)
    counts = {name: 0 for name in SPLIT_NAMES}
    for label in labels:
        counts[label] += 1
    print(f"splits: {counts}")
    return counts
