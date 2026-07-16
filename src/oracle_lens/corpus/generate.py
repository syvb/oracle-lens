"""Stage 0b: on-policy regeneration with vLLM (PLAN.md §3).

Feeds pre-rendered token IDs to vLLM and saves prompt+response TOKEN IDS
directly — the plan's "safest path" for the on-policy consistency requirement:
no re-tokenization ever sits between generation and activation collection.

vLLM is imported lazily: it is not a locked dependency (see pyproject.toml).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from oracle_lens.config import Config
from oracle_lens.rendering import (
    load_subject_tokenizer,
    render_first_turn,
    sampling_params,
    template_fingerprint,
)

CORPUS_SCHEMA = pa.schema(
    [
        ("conversation_id", pa.string()),
        ("prompt", pa.string()),
        ("prompt_ids", pa.list_(pa.int32())),
        ("response_ids", pa.list_(pa.int32())),
        ("finish_reason", pa.string()),
    ]
)


def shard_for_conversation(conversation_id: str, num_shards: int) -> int:
    """Stable prompt partition independent of source ordering or box count."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    digest = hashlib.sha256(f"generate-shard|{conversation_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def seed_for_conversation(base_seed: int, conversation_id: str) -> int:
    """Per-request seed so a prompt's sampling does not depend on its shard/order."""
    digest = hashlib.sha256(f"generate-seed|{base_seed}|{conversation_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def retry_seed_for_conversation(
    base_seed: int, conversation_id: str, attempt: int
) -> int:
    """Deterministic rejection-resampling seed for rare thinking-tag leaks."""
    digest = hashlib.sha256(
        f"generate-seed|{base_seed}|{conversation_id}|nonthinking-retry-{attempt}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def has_thinking_tag(token_ids: list[int], tokenizer: Any) -> bool:
    """Detect Qwen chat-template thinking delimiters by their token IDs."""
    tag_ids = {
        *tokenizer.encode("<think>", add_special_tokens=False),
        *tokenizer.encode("</think>", add_special_tokens=False),
    }
    return bool(tag_ids.intersection(token_ids))


def corpus_shard_path(corpus_path: Path, shard_index: int, num_shards: int) -> Path:
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index={shard_index} outside [0, {num_shards})")
    return corpus_path.with_name(
        f"{corpus_path.stem}.shard-{shard_index:05d}-of-{num_shards:05d}{corpus_path.suffix}"
    )


def corpus_sidecar_path(corpus_path: Path) -> Path:
    return corpus_path.with_name(corpus_path.name + ".meta.yaml")


def write_corpus_sidecar(corpus_path: Path, cfg: Config, tokenizer, n_rows: int) -> None:
    meta = {
        "kind": "oracle_lens_corpus",
        "model": cfg.model.name,
        "enable_thinking": cfg.model.enable_thinking,
        "sampling": sampling_params(cfg),
        "template_fingerprint": template_fingerprint(tokenizer, cfg),
        "n_rows": n_rows,
        "seed": cfg.generation.seed,
    }
    corpus_sidecar_path(corpus_path).write_text(yaml.safe_dump(meta, sort_keys=True))


def assert_fingerprint_matches(corpus_path: Path, cfg: Config, tokenizer) -> None:
    """Called by activation collection before touching the corpus (§3 warning)."""
    meta = yaml.safe_load(corpus_sidecar_path(corpus_path).read_text())
    live = template_fingerprint(tokenizer, cfg)
    if meta["template_fingerprint"] != live:
        raise RuntimeError(
            "template/sampling fingerprint mismatch between generation time and "
            "now — chat template, tokenizer, thinking flag, or sampling settings "
            "drifted. Collecting activations would break the on-policy "
            f"assumption. sidecar={meta['template_fingerprint']} live={live}"
        )


def generate_corpus(
    cfg: Config,
    prompts_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> int:
    from vllm import LLM, SamplingParams, TokensPrompt  # lazy: GPU-node only
    from transformers import PreTrainedTokenizerBase

    # vLLM 0.10.x (the CUDA-12.8 wheel used on driver-570 hosts) reads this
    # legacy Transformers property. Transformers 5 removed it; its old value
    # is exactly the still-present `all_special_tokens` list.
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda self: self.all_special_tokens
        )

    tokenizer = load_subject_tokenizer(cfg.model.name)
    prompts = pq.read_table(prompts_path).to_pydict()
    ids = prompts["conversation_id"]
    texts = prompts["prompt"]
    if limit is not None:
        ids, texts = ids[:limit], texts[:limit]
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be provided together")
    if shard_index is not None and num_shards is not None:
        if not 0 <= shard_index < num_shards:
            raise ValueError(f"shard_index={shard_index} outside [0, {num_shards})")
        selected = [
            i for i, conversation_id in enumerate(ids)
            if shard_for_conversation(conversation_id, num_shards) == shard_index
        ]
        ids = [ids[i] for i in selected]
        texts = [texts[i] for i in selected]
        out_path = corpus_shard_path(out_path, shard_index, num_shards)

    rendered: list[list[int]] = []
    keep: list[int] = []
    for i, text in enumerate(texts):
        tok_ids = render_first_turn(tokenizer, text, cfg)
        if len(tok_ids) <= cfg.generation.max_prompt_tokens:
            keep.append(i)
            rendered.append(tok_ids)

    llm = LLM(model=cfg.model.name, dtype="bfloat16", seed=cfg.generation.seed)
    params = [
        SamplingParams(
            **sampling_params(cfg),
            seed=seed_for_conversation(cfg.generation.seed, ids[i]),
        )
        for i in keep
    ]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=r) for r in rendered], params
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "conversation_id": [ids[i] for i in keep],
            "prompt": [texts[i] for i in keep],
            "prompt_ids": rendered,
            "response_ids": [list(o.outputs[0].token_ids) for o in outputs],
            "finish_reason": [str(o.outputs[0].finish_reason) for o in outputs],
        },
        schema=CORPUS_SCHEMA,
    )
    pq.write_table(table, out_path)
    write_corpus_sidecar(out_path, cfg, tokenizer, len(keep))
    print(f"corpus: generated {len(keep)} transcripts -> {out_path}")
    return len(keep)


def repair_thinking_leaks(
    cfg: Config,
    corpus_path: Path,
    diagnostics_path: Path,
    *,
    max_attempts: int = 8,
) -> int:
    """Rejection-resample rare thinking-tag leaks in a non-thinking corpus.

    Qwen's non-thinking template supplies an empty thinking block, but a tiny
    fraction of sampled responses can nevertheless emit a new closing tag
    after an internal reasoning trace. Preserve those original rows, then
    deterministically resample only them until the response is tag-free.
    """
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM, SamplingParams, TokensPrompt

    if cfg.model.enable_thinking:
        raise RuntimeError("repair-thinking is only valid when enable_thinking=false")
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda self: self.all_special_tokens
        )

    tokenizer = load_subject_tokenizer(cfg.model.name)
    table = pq.read_table(corpus_path)
    rows = table.to_pydict()
    original_bad = [
        i
        for i, response_ids in enumerate(rows["response_ids"])
        if has_thinking_tag(response_ids, tokenizer)
    ]
    if not original_bad:
        print("repair-thinking: no thinking-tag leaks found")
        return 0

    llm = LLM(model=cfg.model.name, dtype="bfloat16", seed=cfg.generation.seed)
    remaining = list(original_bad)
    retry_used: dict[int, int] = {}
    repaired_response: dict[int, list[int]] = {}
    repaired_finish: dict[int, str] = {}
    for attempt in range(1, max_attempts + 1):
        if not remaining:
            break
        params = [
            SamplingParams(
                **sampling_params(cfg),
                seed=retry_seed_for_conversation(
                    cfg.generation.seed, rows["conversation_id"][i], attempt
                ),
            )
            for i in remaining
        ]
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=rows["prompt_ids"][i]) for i in remaining],
            params,
        )
        next_remaining: list[int] = []
        for i, output in zip(remaining, outputs, strict=True):
            response_ids = list(output.outputs[0].token_ids)
            if has_thinking_tag(response_ids, tokenizer):
                next_remaining.append(i)
            else:
                retry_used[i] = attempt
                repaired_response[i] = response_ids
                repaired_finish[i] = str(output.outputs[0].finish_reason)
        remaining = next_remaining
    if remaining:
        ids = [rows["conversation_id"][i] for i in remaining]
        raise RuntimeError(
            f"{len(remaining)} thinking-tag leaks remain after {max_attempts} "
            f"attempts: {ids[:3]}"
        )

    diagnostic_rows = {
        "conversation_id": [],
        "original_response_ids": [],
        "repaired_response_ids": [],
        "retry_attempt": [],
    }
    for i in original_bad:
        diagnostic_rows["conversation_id"].append(rows["conversation_id"][i])
        diagnostic_rows["original_response_ids"].append(rows["response_ids"][i])
        diagnostic_rows["repaired_response_ids"].append(repaired_response[i])
        diagnostic_rows["retry_attempt"].append(retry_used[i])
        rows["response_ids"][i] = repaired_response[i]
        rows["finish_reason"][i] = repaired_finish[i]

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(diagnostic_rows), diagnostics_path)
    pq.write_table(pa.Table.from_pydict(rows, schema=CORPUS_SCHEMA), corpus_path)
    meta_path = corpus_sidecar_path(corpus_path)
    meta = yaml.safe_load(meta_path.read_text())
    meta["thinking_leak_remediation"] = {
        "kind": "deterministic_rejection_resampling",
        "n_rows": len(original_bad),
        "max_attempts": max(retry_used.values()),
        "diagnostics": str(diagnostics_path.name),
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=True))
    print(
        f"repair-thinking: repaired {len(original_bad)} rows; originals and replacements "
        f"-> {diagnostics_path}"
    )
    return len(original_bad)
