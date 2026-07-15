"""Stage 4 evaluation & M4 gate (PLAN.md §7.2).

Held-out (eval split) activations: sample (K, N) as in training, greedy-decode
K phrases, NNLS-refit against the true activation, whitened FVE. Gates:
oracle FVE >= 70% of teacher FVE [choice]; format validity >= 95%
(right K, right N +- 1 token).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from oracle_lens.activations.store import ActivationStore
from oracle_lens.config import Config
from oracle_lens.oracle.decode import OracleDecoder
from oracle_lens.runs import RunPaths


def _sample_kn(cfg: Config, conv_id: str, pos: int) -> tuple[int, int]:
    rng = random.Random(
        hashlib.sha256(f"{cfg.oracle.seed}|eval|{conv_id}|{pos}".encode()).digest()
    )
    k = rng.randint(cfg.oracle.k_min, cfg.oracle.k_max)
    n = rng.choice(list(cfg.dictionary.lengths))
    return k, n


def _phrase_len_ok(tokenizer, phrase: str, n: int) -> bool:
    n_tok = len(tokenizer(phrase, add_special_tokens=False)["input_ids"])
    return abs(n_tok - n) <= 1


def evaluate_oracle(
    cfg: Config,
    run: RunPaths,
    *,
    checkpoint: Path | None = None,
    n_samples: int = 2000,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    checkpoint = checkpoint or run.oracle_sft_dir / "checkpoint"
    run.require(checkpoint, run.positions, run.activations_dir)
    decoder = OracleDecoder(cfg, checkpoint, device=device)
    store = ActivationStore.open(run.activations_dir)

    table = pq.read_table(
        run.positions, columns=["conversation_id", "pos", "is_delimiter", "split"]
    )
    rows = [i for i in range(table.num_rows) if table["split"][i].as_py() == "eval"]
    rows = sorted(
        rows,
        key=lambda i: hashlib.sha256(
            f"oracle-eval|{table['conversation_id'][i].as_py()}|{table['pos'][i].as_py()}".encode()
        ).hexdigest(),
    )[:n_samples]

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in rows:
        kn = _sample_kn(cfg, table["conversation_id"][i].as_py(), table["pos"][i].as_py())
        grouped[kn].append(i)

    fves, valid_flags = [], []
    by_k: dict[int, list[float]] = defaultdict(list)
    by_n: dict[int, list[float]] = defaultdict(list)
    by_class: dict[str, list[float]] = defaultdict(list)
    for (k, n), members in sorted(grouped.items()):
        for lo in range(0, len(members), batch_size):
            chunk = members[lo : lo + batch_size]
            acts = store.read_rows(chunk)
            results = decoder.decode_batch(acts, n=n, k=k)
            for i, res in zip(chunk, results):
                fves.append(res.fve)
                by_k[k].append(res.fve)
                by_n[n].append(res.fve)
                cls = "delimiter" if table["is_delimiter"][i].as_py() else "ordinary"
                by_class[cls].append(res.fve)
                valid_flags.append(
                    len(res.phrases) == k
                    and all(_phrase_len_ok(decoder.tokenizer, p, n) for p in res.phrases)
                )
        print(f"oracle-eval (K={k}, N={n}): {len(members)} examples done", flush=True)

    teacher_gate = json.loads((run.teacher.parent / "gate.json").read_text())
    fve_mean = float(np.mean(fves))
    report = {
        "n_samples": len(fves),
        "fve_mean": fve_mean,
        "fve_by_k": {k: float(np.mean(v)) for k, v in sorted(by_k.items())},
        "fve_by_n": {n: float(np.mean(v)) for n, v in sorted(by_n.items())},
        "fve_by_position_class": {c: float(np.mean(v)) for c, v in by_class.items()},
        "format_validity": float(np.mean(valid_flags)),
        "teacher_fve_mean": teacher_gate["fve_mean_overall"],
        "teacher_frac": fve_mean / max(teacher_gate["fve_mean_overall"], 1e-9),
        "gate": {},
    }
    report["gate"] = {
        "teacher_frac_ok": report["teacher_frac"] >= cfg.oracle.gate_min_teacher_frac,
        "format_validity_ok": report["format_validity"] >= cfg.oracle.gate_min_format_valid,
    }
    out = checkpoint.parent / "eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
