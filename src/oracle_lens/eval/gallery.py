"""§8.2 qualitative probe gallery runner (PLAN.md).

For each probe: generate the subject model's response (rendering.py settings),
capture layer-l* activations at the probe's read positions, decode each
through the oracle, and emit a markdown gallery with auto-checked expect-
keyword hits and a manual pass/fail checkbox per probe. Plus the delimiter
scan: at delimiter vs. matched ordinary positions of held-out transcripts,
oracle readouts side-by-side with SAMPLED CONTINUATIONS from the same state —
the paper's signature phenomenon is commentary at delimiters that the
continuations don't contain.

The gallery, not the FVE number, decides whether the tool is useful (§8.2).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml

from oracle_lens.activations.collect import TokenIdExtractor
from oracle_lens.config import Config
from oracle_lens.corpus.positions import DelimiterOracle
from oracle_lens.corpus.splits import split_for_conversation
from oracle_lens.oracle.decode import OracleDecoder
from oracle_lens.rendering import render_first_turn, sampling_params
from oracle_lens.runs import RunPaths

PROBES_PATH = Path(__file__).parent / "probes" / "probes.yaml"


def load_probes() -> list[dict]:
    return yaml.safe_load(PROBES_PATH.read_text())


def _read_positions(
    probe: dict, n_prompt: int, response_ids: list[int], is_delim: DelimiterOracle
) -> list[int]:
    mode = probe["read_at"]
    total = n_prompt + len(response_ids)
    if mode == "pre_answer":
        return [p for p in (n_prompt - 1, n_prompt, n_prompt + 1) if p < total]
    if mode == "line_break":
        pos = [
            n_prompt + i
            for i, t in enumerate(response_ids)
            if "\n" in probe["_tokenizer"].decode([t])
        ]
        return pos[:4] or [total - 2]
    if mode == "delimiters":
        pos = [n_prompt + i for i, t in enumerate(response_ids) if is_delim(t)]
        return pos[:4] or [total - 2]
    raise ValueError(f"unknown read_at: {mode}")


@torch.no_grad()
def _generate_response(subject, tokenizer, prompt_ids: list[int], cfg: Config) -> list[int]:
    params = sampling_params(cfg)
    out = subject.generate(
        input_ids=torch.tensor([prompt_ids]).to(subject.device),
        do_sample=True,
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_new_tokens=params["max_tokens"],
        pad_token_id=tokenizer.pad_token_id,
    )
    return out[0, len(prompt_ids):].tolist()


@torch.no_grad()
def _sample_continuations(
    subject, tokenizer, prefix_ids: list[int], *, n: int = 3, length: int = 16
) -> list[str]:
    out = subject.generate(
        input_ids=torch.tensor([prefix_ids] * n).to(subject.device),
        do_sample=True,
        temperature=1.0,
        max_new_tokens=length,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [tokenizer.decode(o[len(prefix_ids):]) for o in out]


def _readout_block(
    decoder: OracleDecoder, act: torch.Tensor, ns=(4, 8), k: int = 8
) -> tuple[list[str], list[str]]:
    """Returns (markdown lines, raw phrases). Auto-checks match against the
    RAW phrases only — the formatted lines contain coefficient digits that
    would trivially self-match numeric expect-keywords."""
    lines, phrases = [], []
    for n in ns:
        res = decoder.decode(act, n=n, k=k)
        ranked = ", ".join(f"`{p}` ({c:.2f})" for p, c in res.ranked[:k])
        lines.append(f"  - N={n} (FVE {res.fve:.2f}): {ranked}")
        phrases.extend(res.phrases)
    return lines, phrases


def run_gallery(
    cfg: Config,
    run: RunPaths,
    *,
    checkpoint: Path | None = None,
    device: str = "cuda",
    n_delimiter_transcripts: int = 10,
) -> Path:
    checkpoint = checkpoint or run.oracle_sft_dir / "checkpoint"
    decoder = OracleDecoder(cfg, checkpoint, device=device)
    tokenizer = decoder.tokenizer
    extractor = TokenIdExtractor(model_name=cfg.model.name)
    subject = extractor.model
    is_delim = DelimiterOracle(tokenizer)

    lines = ["# Oracle-lens probe gallery", ""]
    hits = 0
    probes = load_probes()
    for probe in probes:
        probe["_tokenizer"] = tokenizer
        prompt_ids = render_first_turn(tokenizer, probe["prompt"], cfg)
        response_ids = _generate_response(subject, tokenizer, prompt_ids, cfg)
        full = prompt_ids + list(response_ids)
        read_at = _read_positions(probe, len(prompt_ids), response_ids, is_delim)
        acts = extractor.extract_at_positions([full], [read_at], cfg.model.layer_index)[0]

        lines += [
            f"## {probe['id']}  ({probe['category']})",
            "",
            f"**Prompt:** {probe['prompt']!r}",
            f"**Response:** {tokenizer.decode(response_ids, skip_special_tokens=True)!r}",
            "",
        ]
        readout_phrases: list[str] = []
        for pos, act in zip(read_at, acts):
            tok = tokenizer.decode([full[pos]])
            lines.append(f"- position {pos} (token {tok!r}):")
            block, phrases = _readout_block(decoder, act)
            lines += block
            readout_phrases += phrases
        expected = probe.get("expect") or []
        joined = " ".join(readout_phrases).lower()
        found = [
            e for e in expected
            if re.search(rf"\b{re.escape(e.lower())}\b", joined)
        ]
        if expected:
            hits += bool(found)
            lines.append(f"- auto-check: expected {expected}, found {found or 'none'}")
        lines += ["- [ ] PASS (manual review)", ""]

    # Delimiter scan over held-out transcripts (§8.2) — EVAL split only:
    # other splits' delimiter positions may literally be training examples.
    lines += ["# Delimiter scan", ""]
    corpus = pq.read_table(run.corpus, columns=["conversation_id", "prompt_ids", "response_ids"])
    eval_rows = [
        i for i in range(corpus.num_rows)
        if split_for_conversation(cfg, corpus["conversation_id"][i].as_py()) == "eval"
    ]
    order = sorted(
        eval_rows,
        key=lambda i: hashlib.sha256(
            f"gallery|{corpus['conversation_id'][i].as_py()}".encode()
        ).hexdigest(),
    )[:n_delimiter_transcripts]
    for i in order:
        p_ids = corpus["prompt_ids"][i].as_py()
        r_ids = corpus["response_ids"][i].as_py()
        full = list(p_ids) + list(r_ids)
        delims = [len(p_ids) + j for j, t in enumerate(r_ids) if is_delim(t)][:3]
        ordinary = [len(p_ids) + j for j, t in enumerate(r_ids) if not is_delim(t)][
            len(r_ids) // 3 :: max(1, len(r_ids) // 3)
        ][:3]
        pos_list = delims + ordinary
        if not pos_list:
            continue
        acts = extractor.extract_at_positions([full], [pos_list], cfg.model.layer_index)[0]
        lines += [f"## transcript {corpus['conversation_id'][i].as_py()}", ""]
        for pos, act in zip(pos_list, acts):
            kind = "DELIMITER" if pos in delims else "ordinary"
            tok = tokenizer.decode([full[pos]])
            conts = _sample_continuations(subject, tokenizer, full[: pos + 1])
            lines.append(f"- {kind} position {pos} (token {tok!r}):")
            lines += _readout_block(decoder, act)[0]
            lines += [f"  - sampled continuation: `{c}`" for c in conts]
        lines.append("")

    if any(p.get("expect") for p in probes):
        lines.insert(2, f"Auto-check keyword hits: {hits}/{sum(bool(p.get('expect')) for p in probes)}")
    run.eval_dir.mkdir(parents=True, exist_ok=True)
    out = run.eval_dir / "gallery.md"
    out.write_text("\n".join(lines))
    print(f"gallery ({len(probes)} probes + {n_delimiter_transcripts} transcripts) -> {out}")
    return out
