"""Validate and merge parallel M0 generation shards into the canonical corpus."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from oracle_lens.config import Config
from oracle_lens.corpus.generate import (
    CORPUS_SCHEMA,
    corpus_shard_path,
    corpus_sidecar_path,
    shard_for_conversation,
    write_corpus_sidecar,
)
from oracle_lens.rendering import load_subject_tokenizer, render_first_turn, template_fingerprint


def merge_corpus_shards(
    cfg: Config,
    prompts_path: Path,
    corpus_path: Path,
    *,
    num_shards: int,
) -> int:
    tokenizer = load_subject_tokenizer(cfg.model.name)
    expected_fingerprint = template_fingerprint(tokenizer, cfg)
    tables: list[pa.Table] = []
    seen: set[str] = set()

    for shard_index in range(num_shards):
        path = corpus_shard_path(corpus_path, shard_index, num_shards)
        sidecar = corpus_sidecar_path(path)
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(f"missing shard or sidecar: {path}, {sidecar}")
        meta = yaml.safe_load(sidecar.read_text())
        if meta.get("template_fingerprint") != expected_fingerprint:
            raise RuntimeError(f"fingerprint drift in shard {shard_index}: {path}")
        table = pq.read_table(path)
        if table.schema != CORPUS_SCHEMA:
            raise RuntimeError(f"schema mismatch in shard {shard_index}: {table.schema}")
        ids = table["conversation_id"].to_pylist()
        if len(ids) != meta.get("n_rows"):
            raise RuntimeError(f"sidecar row count mismatch in shard {shard_index}")
        wrong = [cid for cid in ids if shard_for_conversation(cid, num_shards) != shard_index]
        if wrong:
            raise RuntimeError(f"{len(wrong)} rows assigned to wrong shard {shard_index}")
        dup = seen.intersection(ids)
        if dup:
            raise RuntimeError(f"duplicate conversation IDs across shards: {sorted(dup)[:3]}")
        seen.update(ids)
        tables.append(table)

    merged = pa.concat_tables(tables)
    by_id = {
        cid: i for i, cid in enumerate(merged["conversation_id"].to_pylist())
    }
    prompts = pq.read_table(prompts_path, columns=["conversation_id", "prompt"])
    expected_order: list[str] = []
    for cid, prompt in zip(
        prompts["conversation_id"].to_pylist(), prompts["prompt"].to_pylist()
    ):
        if len(render_first_turn(tokenizer, prompt, cfg)) <= cfg.generation.max_prompt_tokens:
            expected_order.append(cid)
    if seen != set(expected_order):
        missing = set(expected_order) - seen
        extra = seen - set(expected_order)
        raise RuntimeError(
            f"shard coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    merged = merged.take(pa.array([by_id[cid] for cid in expected_order], type=pa.int64()))
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, corpus_path)
    write_corpus_sidecar(corpus_path, cfg, tokenizer, merged.num_rows)
    print(f"corpus: merged {num_shards} shards, {merged.num_rows} transcripts -> {corpus_path}")
    return merged.num_rows
