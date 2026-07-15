"""Flat fp16 memmap of activations, row-aligned with positions.parquet.

Row i of the store is the activation for row i of positions.parquet — the
positions table IS the index (split labels, delimiter flags, phrase bounds all
live there). 1M x 4096 x fp16 ~ 8 GB (PLAN.md §4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

VECTORS_FILE = "vectors.f16.bin"
META_FILE = "meta.yaml"


class ActivationStore:
    def __init__(self, root: Path, n_rows: int, d_model: int, mode: str) -> None:
        self.root = root
        self.n_rows = n_rows
        self.d_model = d_model
        self._mm = np.memmap(
            root / VECTORS_FILE, dtype=np.float16, mode=mode, shape=(n_rows, d_model)
        )

    @classmethod
    def create(cls, root: Path, n_rows: int, d_model: int, meta: dict) -> "ActivationStore":
        root.mkdir(parents=True, exist_ok=True)
        (root / META_FILE).write_text(
            yaml.safe_dump({"n_rows": n_rows, "d_model": d_model, **meta}, sort_keys=True)
        )
        return cls(root, n_rows, d_model, mode="w+")

    @classmethod
    def open(cls, root: Path) -> "ActivationStore":
        meta = cls.meta(root)
        return cls(root, meta["n_rows"], meta["d_model"], mode="r")

    @staticmethod
    def meta(root: Path) -> dict:
        return yaml.safe_load((root / META_FILE).read_text())

    def write_rows(self, row_indices: np.ndarray, vectors: torch.Tensor) -> None:
        self._mm[row_indices] = vectors.to(torch.float16).cpu().numpy()

    def read_rows(self, row_indices: np.ndarray | list[int]) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self._mm[row_indices])).to(torch.float32)

    def iter_batches(self, batch_size: int = 8192):
        for lo in range(0, self.n_rows, batch_size):
            yield lo, self.read_rows(np.arange(lo, min(lo + batch_size, self.n_rows)))

    def flush(self) -> None:
        self._mm.flush()
