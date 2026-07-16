"""Stage 3a: phrase dictionary (PLAN.md §6.1).

From the dictionary split's start positions, take the 2/4/8/16/32-token
continuations, dedup within each length, and run every phrase through the
trained reconstructor. The dictionary exists only to manufacture supervised
targets — it does not limit what the final oracle can say.

Layout: dictionary/N{n}/{phrases.parquet, directions.f16.bin, meta.yaml}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from oracle_lens.config import Config
from oracle_lens.reconstructor.data import load_corpus_ids
from oracle_lens.reconstructor.model import encode_phrases, load_reconstructor
from oracle_lens.rendering import load_subject_tokenizer
from oracle_lens.runs import RunPaths

DIRECTIONS_FILE = "directions.f16.bin"
PHRASES_FILE = "phrases.parquet"
META_FILE = "meta.yaml"


class DictionaryBucket:
    """One phrase length's dictionary: phrases[i] <-> directions[i]."""

    def __init__(self, root: Path) -> None:
        meta = yaml.safe_load((root / META_FILE).read_text())
        self.n_tokens: int = meta["n_tokens"]
        self.size: int = meta["size"]
        self.d_model: int = meta["d_model"]
        self.phrases: list[str] = (
            pq.read_table(root / PHRASES_FILE, columns=["phrase"])["phrase"].to_pylist()
        )
        self._dirs = np.memmap(
            root / DIRECTIONS_FILE, dtype=np.float16, mode="r",
            shape=(self.size, self.d_model),
        )

    def directions(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.from_numpy(np.asarray(self._dirs)).to(device, torch.float32)


def bucket_dir(dictionary_root: Path, n: int) -> Path:
    return dictionary_root / f"N{n}"


def build_dictionary(
    cfg: Config, run: RunPaths, *, checkpoint: Path | None = None, device: str = "cuda"
) -> None:
    checkpoint = checkpoint or run.reconstructor_dir / "checkpoint"
    run.require(checkpoint, run.corpus, run.positions)
    tokenizer = load_subject_tokenizer(cfg.model.name)
    model = load_reconstructor(checkpoint).to(device)
    full_ids = load_corpus_ids(run.corpus)

    table = pq.read_table(
        run.positions, columns=["conversation_id", "pos", "n_following", "split"]
    )
    rows = [
        (table["conversation_id"][i].as_py(), table["pos"][i].as_py(), table["n_following"][i].as_py())
        for i in range(table.num_rows)
        if table["split"][i].as_py() == "dictionary"
    ]

    for n in cfg.dictionary.lengths:
        seen: dict[str, None] = {}
        for conv_id, pos, n_following in rows:
            if n_following < n:
                continue
            phrase = tokenizer.decode(full_ids[conv_id][pos + 1 : pos + 1 + n])
            if phrase.strip():
                seen.setdefault(phrase, None)
        phrases = list(seen)
        if not phrases:
            print(f"dictionary N={n}: no phrases (corpus too small?) — skipping bucket")
            continue
        out = bucket_dir(run.dictionary_dir, n)
        out.mkdir(parents=True, exist_ok=True)

        dirs = encode_phrases(
            model, tokenizer, phrases, cfg.reconstructor.prompt_template,
            batch_size=cfg.dictionary.encode_batch_size, device=device,
        )
        mm = np.memmap(
            out / DIRECTIONS_FILE, dtype=np.float16, mode="w+",
            shape=(len(phrases), cfg.model.d_model),
        )
        mm[:] = dirs.numpy().astype(np.float16)
        mm.flush()
        pq.write_table(
            pa.Table.from_pydict({"phrase": phrases}), out / PHRASES_FILE
        )
        (out / META_FILE).write_text(
            yaml.safe_dump(
                {
                    "n_tokens": n,
                    "size": len(phrases),
                    "d_model": cfg.model.d_model,
                    "reconstructor": str(checkpoint),
                },
                sort_keys=True,
            )
        )
        print(f"dictionary N={n}: {len(phrases)} unique phrases -> {out}")
