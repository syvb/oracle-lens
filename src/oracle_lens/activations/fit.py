"""Stage 1b: fit + validate the whitening transform (PLAN.md §4.2).

mu, Sigma are estimated on the reconstructor-train + teacher splits (>=550k
vectors >> d), validated on the eval split, and persisted next to the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from oracle_lens.activations.store import ActivationStore
from oracle_lens.activations.whitening import fit_whitening, validate_whitening
from oracle_lens.config import Config

FIT_SPLITS = ("reconstructor", "teacher")
VALIDATE_SPLIT = "eval"
_VALIDATE_CAP = 50_000


def fit_whitening_for_run(
    cfg: Config, positions_path: Path, store_dir: Path, out_path: Path
) -> dict[str, float]:
    splits = np.array(pq.read_table(positions_path, columns=["split"])["split"].to_pylist())
    store = ActivationStore.open(store_dir)
    if len(splits) != store.n_rows:
        raise RuntimeError(
            f"positions ({len(splits)}) and activation store ({store.n_rows}) disagree"
        )

    fit_mask = np.isin(splits, FIT_SPLITS)

    def fit_batches():
        for lo, batch in store.iter_batches():
            m = fit_mask[lo : lo + batch.shape[0]]
            if m.any():
                yield batch[np.flatnonzero(m)]

    transform = fit_whitening(fit_batches(), ridge_frac=cfg.whitening.ridge_frac)
    transform.save(out_path)

    val_rows = np.flatnonzero(splits == VALIDATE_SPLIT)[:_VALIDATE_CAP]
    stats = validate_whitening(transform, store.read_rows(val_rows))
    stats["n_fit"] = int(fit_mask.sum())
    print(f"whitening fit on {stats['n_fit']} vectors -> {out_path}")
    print(f"  held-out: {stats}")
    if not (0.8 < stats["var_mean"] < 1.2):
        print(
            "  WARNING: held-out whitened variance far from 1 — whitening is "
            "suspect (M1 gate; see PLAN.md §10.4)"
        )
    return stats
