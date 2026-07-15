"""Stage 1: hooked-prefill activation collection (PLAN.md §4).

Reuses the vendored nla HFExtractor (model loading, layer-hook capture,
right-padding discipline) but feeds EXACT stored token IDs instead of text —
re-tokenizing rendered transcripts is precisely the silent on-policy breakage
§3 warns about. The corpus sidecar fingerprint is asserted before anything
runs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from nla.arch_adapters import resolve_text_config
from nla.datagen.extractors import HFExtractor

from oracle_lens.activations.store import ActivationStore
from oracle_lens.config import Config
from oracle_lens.corpus.generate import assert_fingerprint_matches
from oracle_lens.rendering import load_subject_tokenizer


class TokenIdExtractor(HFExtractor):
    """HFExtractor that accepts token IDs + positions directly."""

    @torch.no_grad()
    def extract_at_positions(
        self,
        sequences: list[list[int]],
        positions: list[list[int]],
        layer_index: int,
    ) -> list[torch.Tensor]:
        """For each sequence, return the layer-`layer_index` residual stream at
        the requested positions: list of [n_positions_i, d_model] fp32 CPU."""
        handle = self._register_hook(layer_index)
        try:
            return self._extract_ids_impl(sequences, positions)
        finally:
            handle.remove()

    def _extract_ids_impl(
        self, sequences: list[list[int]], positions: list[list[int]]
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        device = self.model.get_input_embeddings().weight.device
        pad_id = self.tokenizer.pad_token_id
        for start in range(0, len(sequences), self.batch_size):
            sub = sequences[start : start + self.batch_size]
            sub_pos = positions[start : start + self.batch_size]
            max_len = max(len(s) for s in sub)
            input_ids = torch.full((len(sub), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros(len(sub), max_len, dtype=torch.long)
            for i, s in enumerate(sub):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                attention_mask[i, : len(s)] = 1

            self._captured = None
            self.model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            )
            assert self._captured is not None, "layer hook did not fire"
            hidden = self._captured.float().cpu()
            for i, pos in enumerate(sub_pos):
                assert all(p < len(sub[i]) for p in pos), (
                    f"position out of range: {pos} vs seq len {len(sub[i])}"
                )
                results.append(hidden[i, pos].clone())
        return results


def _iter_length_sorted_chunks(
    items: list[tuple[int, list[int], list[int]]], chunk: int
) -> Iterable[list[tuple[int, list[int], list[int]]]]:
    """Sort by sequence length within large windows so batches pad efficiently;
    row indices ride along so results scatter back to store order."""
    for lo in range(0, len(items), chunk):
        window = sorted(items[lo : lo + chunk], key=lambda t: len(t[1]))
        yield window


def collect_activations(cfg: Config, corpus_path: Path, positions_path: Path, out_dir: Path) -> None:
    tokenizer = load_subject_tokenizer(cfg.model.name)
    assert_fingerprint_matches(corpus_path, cfg, tokenizer)

    corpus = pq.read_table(corpus_path, columns=["conversation_id", "prompt_ids", "response_ids"])
    full_ids: dict[str, list[int]] = {}
    for conv_id, p_ids, r_ids in zip(
        corpus["conversation_id"].to_pylist(),
        corpus["prompt_ids"].to_pylist(),
        corpus["response_ids"].to_pylist(),
    ):
        full_ids[conv_id] = list(p_ids) + list(r_ids)

    positions = pq.read_table(positions_path, columns=["conversation_id", "pos"])
    by_conv: dict[str, list[tuple[int, int]]] = defaultdict(list)  # conv -> [(row, pos)]
    for row, (conv_id, pos) in enumerate(
        zip(positions["conversation_id"].to_pylist(), positions["pos"].to_pylist())
    ):
        by_conv[conv_id].append((row, pos))

    extractor = TokenIdExtractor(model_name=cfg.model.name)
    model_cfg = resolve_text_config(extractor.model.config)
    if model_cfg.hidden_size != cfg.model.d_model:
        raise RuntimeError(
            f"config.json hidden_size={model_cfg.hidden_size} != cfg.model.d_model="
            f"{cfg.model.d_model}; update the config (PLAN.md §2)"
        )
    if not (0 <= cfg.model.layer_index < model_cfg.num_hidden_layers):
        raise RuntimeError(
            f"layer_index={cfg.model.layer_index} out of range for "
            f"{model_cfg.num_hidden_layers}-layer model"
        )

    store = ActivationStore.create(
        out_dir,
        n_rows=positions.num_rows,
        d_model=cfg.model.d_model,
        meta={
            "model": cfg.model.name,
            "layer_index": cfg.model.layer_index,
            "positions_path": str(positions_path),
        },
    )

    items = [
        (rows_pos[0][0], full_ids[conv_id], [p for _, p in rows_pos])
        for conv_id, rows_pos in by_conv.items()
    ]
    # store rows for each conversation: positions rows, in the order they appear
    row_lists = {conv_id: [r for r, _ in rows_pos] for conv_id, rows_pos in by_conv.items()}
    conv_order = list(by_conv)

    done = 0
    triples = [
        (i, full_ids[conv_id], [p for _, p in by_conv[conv_id]])
        for i, conv_id in enumerate(conv_order)
    ]
    del items
    for window in _iter_length_sorted_chunks(triples, chunk=4096):
        seqs = [t[1] for t in window]
        pos_lists = [t[2] for t in window]
        vecs = extractor.extract_at_positions(seqs, pos_lists, cfg.model.layer_index)
        for (conv_i, _, _), v in zip(window, vecs):
            rows = np.array(row_lists[conv_order[conv_i]])
            store.write_rows(rows, v)
        done += len(window)
        print(f"collected {done}/{len(triples)} conversations", flush=True)
    store.flush()
    print(f"activations -> {out_dir}")
