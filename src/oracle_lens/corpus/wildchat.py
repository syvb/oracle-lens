"""Stage 0a: WildChat first-user-turn extraction (PLAN.md §3).

WildChat's assistant responses were written by GPT-3.5/4 and are discarded;
we keep only the FIRST user turn of each conversation, English-filtered and
deduplicated, as prompts for on-policy regeneration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from oracle_lens.config import Config

PROMPTS_SCHEMA = pa.schema(
    [("conversation_id", pa.string()), ("prompt", pa.string())]
)


def _first_user_turn(conversation: list[dict]) -> str | None:
    for turn in conversation:
        if turn.get("role") == "user":
            content = turn.get("content")
            return content if isinstance(content, str) else None
    return None


def build_prompts(cfg: Config, out_path: Path, *, dataset_name: str | None = None) -> int:
    """Write prompts.parquet; returns row count. Deterministic in cfg seeds."""
    gen = cfg.generation
    ds = load_dataset(dataset_name or gen.corpus, split="train")

    seen: set[str] = set()
    rows_id: list[str] = []
    rows_prompt: list[str] = []
    n_scanned = 0
    for i, ex in enumerate(ds):
        n_scanned += 1
        if gen.english_only and ex.get("language") not in (None, "English"):
            continue
        prompt = _first_user_turn(ex.get("conversation") or [])
        if not prompt or not prompt.strip():
            continue
        # Character cap ~4x the token cap; the hard token cap happens at render.
        if len(prompt) > 4 * gen.max_prompt_tokens:
            continue
        dedup_key = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        conv_id = ex.get("conversation_hash") or f"{gen.corpus}:train:{i}"
        rows_id.append(str(conv_id))
        rows_prompt.append(prompt)
        if len(rows_id) >= gen.n_conversations:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {"conversation_id": rows_id, "prompt": rows_prompt}, schema=PROMPTS_SCHEMA
    )
    pq.write_table(table, out_path)
    print(
        f"prompts: kept {len(rows_id)} of {n_scanned} scanned "
        f"(english_only={gen.english_only}, dedup dropped {n_scanned - len(rows_id)} incl. filters)"
    )
    return len(rows_id)
