"""§8.1 quantitative table (PLAN.md).

All rows share the eval split, the (K, N) sampler from oracle/evaluate.py, and
the single NNLS+FVE decode path (nnls_refit_fve) so numbers are comparable:

  random dictionary phrases (matched K, N)   -> floor
  true-continuation phrases (K = 1..4)       -> "trivial prediction" baseline
  teacher NN-OMP                             -> supervised ceiling (from gate.json)
  oracle SFT / RL                            -> deliverables (from their eval.json)
  J-lens top-k tokens as length-1 phrases    -> single-token method comparison
"""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pyarrow.parquet as pq
import torch

from oracle_lens.activations.store import ActivationStore
from oracle_lens.activations.whitening import WhiteningTransform
from oracle_lens.config import Config
from oracle_lens.nnls import nnls_refit_fve
from oracle_lens.oracle.evaluate import _sample_kn
from oracle_lens.reconstructor.data import load_corpus_ids
from oracle_lens.reconstructor.model import encode_phrases, load_reconstructor
from oracle_lens.rendering import load_subject_tokenizer
from oracle_lens.runs import RunPaths
from oracle_lens.teacher.dictionary import DictionaryBucket, bucket_dir


def _eval_rows(table, n_samples: int) -> list[int]:
    rows = [i for i in range(table.num_rows) if table["split"][i].as_py() == "eval"]
    return sorted(
        rows,
        key=lambda i: hashlib.sha256(
            f"baselines|{table['conversation_id'][i].as_py()}|{table['pos'][i].as_py()}".encode()
        ).hexdigest(),
    )[:n_samples]


def teacher_on_eval_baseline(
    cfg: Config, run: RunPaths, table, rows, targets_w: torch.Tensor, *, device: str
) -> float:
    """Teacher NN-OMP recomputed ON THE EVAL ROWS (same N sampler, same
    half-dictionary convention) so the §8.1 supervised-ceiling row lives on
    the same split as every other row — gate.json's number is the teacher
    split and only distributionally comparable."""
    from collections import defaultdict as dd

    from oracle_lens.teacher.decompose import _seed64
    from oracle_lens.teacher.nnomp import nn_omp

    by_n: dict[int, list[int]] = dd(list)  # bucket -> positions into `rows`
    for j, i in enumerate(rows):
        conv, pos = table["conversation_id"][i].as_py(), table["pos"][i].as_py()
        _, n = _sample_kn(cfg, conv, pos)
        by_n[n].append(j)

    fves: list[float] = []
    for n, members in sorted(by_n.items()):
        dictionary = DictionaryBucket(bucket_dir(run.dictionary_dir, n)).directions(device)
        seeds = torch.tensor(
            [
                _seed64(
                    cfg.teacher.seed, "half",
                    table["conversation_id"][rows[j]].as_py(),
                    table["pos"][rows[j]].as_py(),
                )
                for j in members
            ],
            dtype=torch.int64,
            device=device,
        )
        res = nn_omp(
            dictionary,
            targets_w[members].to(device),
            seeds=seeds,
            max_atoms=cfg.teacher.max_atoms,
            min_gain=cfg.teacher.min_gain,
        )
        fves.extend(res.final_fve.cpu().tolist())
    return float(np.mean(fves))


def random_dictionary_baseline(
    cfg: Config, run: RunPaths, table, rows, targets_w: torch.Tensor
) -> float:
    """Matched-(K, N) random dictionary atoms + NNLS = the floor."""
    buckets = {n: DictionaryBucket(bucket_dir(run.dictionary_dir, n)) for n in cfg.dictionary.lengths}
    dirs = {n: b.directions() for n, b in buckets.items()}
    fves = []
    for j, i in enumerate(rows):
        conv, pos = table["conversation_id"][i].as_py(), table["pos"][i].as_py()
        k, n = _sample_kn(cfg, conv, pos)
        rng = random.Random(hashlib.sha256(f"randdict|{conv}|{pos}".encode()).digest())
        idxs = [rng.randrange(buckets[n].size) for _ in range(k)]
        atoms = dirs[n][idxs].unsqueeze(0)
        _, fve = nnls_refit_fve(atoms, targets_w[j : j + 1])
        fves.append(float(fve[0]))
    return float(np.mean(fves))


def true_continuation_baseline(
    cfg: Config,
    run: RunPaths,
    table,
    rows,
    targets_w: torch.Tensor,
    reconstructor,
    tokenizer,
    *,
    device: str,
) -> dict[int, float]:
    """K consecutive non-overlapping N-token chunks of the TRUE continuation,
    K = 1..4. The oracle must beat this clearly at delimiter positions."""
    full_ids = load_corpus_ids(run.corpus)
    out: dict[int, float] = {}
    for k in (1, 2, 3, 4):
        phrases_per_row: list[list[str]] = []
        keep: list[int] = []
        for j, i in enumerate(rows):
            conv, pos = table["conversation_id"][i].as_py(), table["pos"][i].as_py()
            _, n = _sample_kn(cfg, conv, pos)
            if table["n_following"][i].as_py() < k * n:
                continue
            ids = full_ids[conv]
            phrases = [
                tokenizer.decode(ids[pos + 1 + c * n : pos + 1 + (c + 1) * n])
                for c in range(k)
            ]
            phrases_per_row.append(phrases)
            keep.append(j)
        if not keep:
            continue
        flat = [p for ps in phrases_per_row for p in ps]
        dirs = encode_phrases(
            reconstructor, tokenizer, flat, cfg.reconstructor.prompt_template, device=device
        )
        atoms = dirs.reshape(len(keep), k, -1)
        _, fve = nnls_refit_fve(atoms, targets_w[keep])
        out[k] = float(fve.mean())
    return out


def jlens_baseline(
    cfg: Config,
    run: RunPaths,
    targets_raw: torch.Tensor,
    targets_w: torch.Tensor,
    reconstructor,
    tokenizer,
    *,
    top_k: int = 8,
    device: str = "cuda",
) -> float:
    """J-lens top-k tokens, treated as k length-1 phrases through the same
    reconstructor+NNLS path — the single-token method comparison."""
    import jlens
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16
    ).to(device)
    model = jlens.from_hf(hf, tokenizer)
    lens = jlens.JacobianLens.load(run.layer_check_dir / "lens.pt")
    logits = model.unembed(
        lens.transport(targets_raw.to(device, torch.float32), cfg.model.layer_index)
    )
    top_ids = logits.topk(top_k, dim=-1).indices.cpu()
    del hf, model

    fves = []
    for j in range(targets_w.shape[0]):
        phrases = [tokenizer.decode([t]) for t in top_ids[j].tolist()]
        dirs = encode_phrases(
            reconstructor, tokenizer, phrases, cfg.reconstructor.prompt_template, device=device
        )
        _, fve = nnls_refit_fve(dirs.unsqueeze(0), targets_w[j : j + 1])
        fves.append(float(fve[0]))
    return float(np.mean(fves))


def run_baselines(
    cfg: Config,
    run: RunPaths,
    *,
    n_samples: int = 1000,
    skip_jlens: bool = False,
    device: str = "cuda",
) -> dict:
    run.require(run.positions, run.activations_dir, run.whitening, run.dictionary_dir)
    table = pq.read_table(
        run.positions, columns=["conversation_id", "pos", "n_following", "split"]
    )
    rows = _eval_rows(table, n_samples)
    store = ActivationStore.open(run.activations_dir)
    whitening = WhiteningTransform.load(run.whitening)
    targets_raw = store.read_rows(rows)
    targets_w = whitening.whiten(targets_raw)
    tokenizer = load_subject_tokenizer(cfg.model.name)
    reconstructor = load_reconstructor(run.reconstructor_dir / "checkpoint").to(device)

    results: dict = {"n_samples": len(rows)}
    results["random_dictionary_fve"] = random_dictionary_baseline(
        cfg, run, table, rows, targets_w
    )
    results["true_continuation_fve_by_k"] = true_continuation_baseline(
        cfg, run, table, rows, targets_w, reconstructor, tokenizer, device=device
    )
    results["teacher_eval_split_fve"] = teacher_on_eval_baseline(
        cfg, run, table, rows, targets_w, device=device
    )
    if not skip_jlens and (run.layer_check_dir / "lens.pt").exists():
        results["jlens_top8_fve"] = jlens_baseline(
            cfg, run, targets_raw, targets_w, reconstructor, tokenizer, device=device
        )

    # Pull the already-computed rows for context (teacher-split convention).
    teacher_gate = run.teacher.parent / "gate.json"
    if teacher_gate.exists():
        results["teacher_fve_teacher_split"] = json.loads(teacher_gate.read_text())[
            "fve_mean_overall"
        ]
    for name, d in (("oracle_sft", run.oracle_sft_dir), ("oracle_rl", run.oracle_rl_dir)):
        rpt = d / "eval.json"
        if rpt.exists():
            results[f"{name}_fve"] = json.loads(rpt.read_text())["fve_mean"]

    run.eval_dir.mkdir(parents=True, exist_ok=True)
    out = run.eval_dir / "table.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results
