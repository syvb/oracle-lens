"""Stage 3b: teacher NN-OMP decompositions (PLAN.md §6.2) + the M3 gate.

Per teacher-split activation: sample ONE phrase length (uniform over the five
buckets, keyed RNG), restrict to a seeded random half of that bucket
(teacher/nnomp.py's hash masks), run NN-OMP in whitened space, record the
ordered phrases, coefficients, and cumulative FVE.

The teacher's mean whitened FVE is the supervised ceiling for oracle SFT and
the single best health check of the pipeline so far (gate: >= ~15% [choice]).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from oracle_lens.activations.store import ActivationStore
from oracle_lens.activations.whitening import WhiteningTransform
from oracle_lens.config import Config
from oracle_lens.runs import RunPaths
from oracle_lens.teacher.dictionary import DictionaryBucket, bucket_dir
from oracle_lens.teacher.nnomp import nn_omp

TEACHER_SCHEMA = pa.schema(
    [
        ("store_row", pa.int64()),
        ("conversation_id", pa.string()),
        ("pos", pa.int32()),
        ("is_delimiter", pa.bool_()),
        ("n_tokens", pa.int32()),
        ("phrases", pa.list_(pa.string())),
        ("coeffs", pa.list_(pa.float32())),
        ("fve_path", pa.list_(pa.float32())),
        ("final_fve", pa.float32()),
    ]
)


def _seed64(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def run_teacher(cfg: Config, run: RunPaths, *, device: str = "cuda") -> dict:
    tcfg = cfg.teacher
    run.require(run.positions, run.activations_dir, run.whitening, run.dictionary_dir)
    store = ActivationStore.open(run.activations_dir)
    whitening = WhiteningTransform.load(run.whitening)

    table = pq.read_table(
        run.positions, columns=["conversation_id", "pos", "is_delimiter", "split"]
    )
    teacher_rows = [
        i for i in range(table.num_rows) if table["split"][i].as_py() == "teacher"
    ]

    # Assign each example a length bucket, uniform over lengths [paper], keyed.
    lengths = list(cfg.dictionary.lengths)
    by_n: dict[int, list[int]] = {n: [] for n in lengths}
    for i in teacher_rows:
        key = _seed64(
            tcfg.seed, "length", table["conversation_id"][i].as_py(), table["pos"][i].as_py()
        )
        by_n[lengths[key % len(lengths)]].append(i)

    out_path = run.teacher
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, TEACHER_SCHEMA)
    fve_all: list[float] = []
    stats: dict = {"by_n": {}, "by_position_class": {"delimiter": [], "ordinary": []}}

    for n in lengths:
        rows = by_n[n]
        if not rows:
            continue
        bucket = DictionaryBucket(bucket_dir(run.dictionary_dir, n))
        dictionary = bucket.directions(device)
        n_fve: list[float] = []
        for lo in range(0, len(rows), tcfg.batch_size):
            chunk = rows[lo : lo + tcfg.batch_size]
            targets = whitening.whiten(store.read_rows(chunk)).to(device)
            seeds = torch.tensor(
                [
                    _seed64(
                        tcfg.seed, "half", table["conversation_id"][i].as_py(),
                        table["pos"][i].as_py(),
                    )
                    for i in chunk
                ],
                dtype=torch.int64,
                device=device,
            )
            res = nn_omp(
                dictionary, targets, seeds=seeds,
                max_atoms=tcfg.max_atoms, min_gain=tcfg.min_gain,
            )
            records = {k: [] for k in TEACHER_SCHEMA.names}
            for j, i in enumerate(chunk):
                k = int(res.n_selected[j])
                idxs = res.indices[j, :k].tolist()
                is_delim = table["is_delimiter"][i].as_py()
                fve = float(res.final_fve[j])
                records["store_row"].append(i)
                records["conversation_id"].append(table["conversation_id"][i].as_py())
                records["pos"].append(table["pos"][i].as_py())
                records["is_delimiter"].append(is_delim)
                records["n_tokens"].append(n)
                records["phrases"].append([bucket.phrases[x] for x in idxs])
                records["coeffs"].append(res.coeffs[j, :k].tolist())
                records["fve_path"].append(res.fve_path[j, :k].tolist())
                records["final_fve"].append(fve)
                n_fve.append(fve)
                stats["by_position_class"][
                    "delimiter" if is_delim else "ordinary"
                ].append(fve)
            writer.write_table(pa.Table.from_pydict(records, schema=TEACHER_SCHEMA))
            print(f"teacher N={n}: {min(lo + tcfg.batch_size, len(rows))}/{len(rows)}", flush=True)
        stats["by_n"][n] = {
            "n_examples": len(n_fve),
            "fve_mean": float(np.mean(n_fve)),
        }
        fve_all.extend(n_fve)
    writer.close()

    gate = {
        "fve_mean_overall": float(np.mean(fve_all)),
        "fve_mean_delimiter": float(np.mean(stats["by_position_class"]["delimiter"]))
        if stats["by_position_class"]["delimiter"] else None,
        "fve_mean_ordinary": float(np.mean(stats["by_position_class"]["ordinary"]))
        if stats["by_position_class"]["ordinary"] else None,
        "gate_min_fve": tcfg.gate_min_fve,
        "gate_ok": float(np.mean(fve_all)) >= tcfg.gate_min_fve,
        "by_n": stats["by_n"],
    }
    (out_path.parent / "gate.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps({k: v for k, v in gate.items() if k != "by_n"}, indent=2))
    if not gate["gate_ok"]:
        print(
            "M3 GATE FAILED — in order: more reconstructor data, bigger "
            "dictionary, revisit ridge lambda and l* (PLAN.md §6.2)"
        )
    return gate
