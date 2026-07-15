"""Stage 0b: on-policy regeneration with vLLM (PLAN.md §3).

Feeds pre-rendered token IDs to vLLM and saves prompt+response TOKEN IDS
directly — the plan's "safest path" for the on-policy consistency requirement:
no re-tokenization ever sits between generation and activation collection.

vLLM is imported lazily: it is not a locked dependency (see pyproject.toml).
"""

from __future__ import annotations

from pathlib import Path

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


def generate_corpus(cfg: Config, prompts_path: Path, out_path: Path, *, limit: int | None = None) -> int:
    from vllm import LLM, SamplingParams, TokensPrompt  # lazy: GPU-node only

    tokenizer = load_subject_tokenizer(cfg.model.name)
    prompts = pq.read_table(prompts_path).to_pydict()
    ids = prompts["conversation_id"]
    texts = prompts["prompt"]
    if limit is not None:
        ids, texts = ids[:limit], texts[:limit]

    rendered: list[list[int]] = []
    keep: list[int] = []
    for i, text in enumerate(texts):
        tok_ids = render_first_turn(tokenizer, text, cfg)
        if len(tok_ids) <= cfg.generation.max_prompt_tokens:
            keep.append(i)
            rendered.append(tok_ids)

    llm = LLM(model=cfg.model.name, dtype="bfloat16", seed=cfg.generation.seed)
    params = SamplingParams(**sampling_params(cfg), seed=cfg.generation.seed)
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
