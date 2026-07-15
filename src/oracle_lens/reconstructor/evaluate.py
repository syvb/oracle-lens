"""Stage 2 evaluation & M2 gate (PLAN.md §5).

- held-out whitened cosine, by phrase length N (expect monotone degradation)
- continuation-FVE probe: the TRUE next-N phrase through the reconstructor,
  single-coefficient NNLS refit (closed form for one unit atom:
  c = max(0, <dir, x>)), whitened FVE — the "PastLens-style floor"
- controls: shuffled phrase pairing, and the best constant direction
  (mean of unit targets); both must be dramatically worse
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from oracle_lens.activations.store import ActivationStore
from oracle_lens.activations.whitening import WhiteningTransform
from oracle_lens.config import Config
from oracle_lens.eval.fve import fve_per_example
from oracle_lens.reconstructor.data import build_pairs
from oracle_lens.reconstructor.model import encode_phrases, load_reconstructor, unit
from oracle_lens.rendering import load_subject_tokenizer
from oracle_lens.runs import RunPaths

EVAL_NS = (1, 2, 4, 8, 16, 32)


def _continuation_fve(dirs: torch.Tensor, targets_w: torch.Tensor) -> torch.Tensor:
    coef = (dirs * targets_w).sum(-1).clamp_min(0.0)  # 1-atom NNLS, closed form
    return fve_per_example(targets_w, coef.unsqueeze(-1) * dirs)


def evaluate_reconstructor(
    cfg: Config,
    run: RunPaths,
    *,
    checkpoint: Path | None = None,
    n_per_bucket: int = 2000,
    device: str = "cuda",
) -> dict:
    checkpoint = checkpoint or run.reconstructor_dir / "checkpoint"
    run.require(checkpoint, run.whitening, run.activations_dir)
    tokenizer = load_subject_tokenizer(cfg.model.name)
    model = load_reconstructor(checkpoint).to(device)
    store = ActivationStore.open(run.activations_dir)
    whitening = WhiteningTransform.load(run.whitening)
    template = cfg.reconstructor.prompt_template

    report: dict = {"by_n": {}, "controls": {}, "gate": {}}
    all_cos: list[float] = []
    for n in EVAL_NS:
        pairs = build_pairs(
            cfg, run.corpus, run.positions, tokenizer,
            split="eval", fixed_n=n, limit=n_per_bucket,
        )
        if not pairs:
            continue
        dirs = encode_phrases(
            model, tokenizer, [p.phrase for p in pairs], template, device=device
        )
        targets_w = whitening.whiten(
            store.read_rows([p.store_row for p in pairs])
        )
        tw_unit = unit(targets_w)
        cos = (dirs * tw_unit).sum(-1)
        cont_fve = _continuation_fve(dirs, targets_w)
        delim = torch.tensor([p.is_delimiter for p in pairs])
        report["by_n"][n] = {
            "n_examples": len(pairs),
            "cosine_mean": cos.mean().item(),
            "continuation_fve_mean": cont_fve.mean().item(),
            "continuation_fve_delim": cont_fve[delim].mean().item() if delim.any() else None,
            "continuation_fve_nondelim": cont_fve[~delim].mean().item() if (~delim).any() else None,
        }
        all_cos.append(cos.mean().item())

        if n == 4:  # controls on one representative bucket
            shuffled_cos = (dirs.roll(1, dims=0) * tw_unit).sum(-1).mean().item()
            mean_dir = unit(tw_unit.mean(dim=0, keepdim=True))
            mean_cos = (mean_dir * tw_unit).sum(-1).mean().item()
            report["controls"] = {
                "shuffled_pairing_cosine": shuffled_cos,
                "mean_direction_cosine": mean_cos,
                "true_pairing_cosine": cos.mean().item(),
            }

    # M2 gate (PLAN.md §5): (a) clear separation from controls,
    # (c) continuation FVE meaningfully above zero at N <= 8.
    # (b) — cosine still improving vs. plateaued — is read off metrics.jsonl.
    ctrl = report["controls"]
    sep = ctrl["true_pairing_cosine"] - max(
        ctrl["shuffled_pairing_cosine"], ctrl["mean_direction_cosine"]
    )
    small_n_fve = [report["by_n"][n]["continuation_fve_mean"] for n in (1, 2, 4, 8) if n in report["by_n"]]
    report["gate"] = {
        "control_separation": sep,
        "control_separation_ok": sep > 0.1,
        "small_n_continuation_fve_min": min(small_n_fve) if small_n_fve else None,
        "small_n_continuation_fve_ok": bool(small_n_fve) and min(small_n_fve) > 0.02,
        "cosine_mean_overall": sum(all_cos) / len(all_cos) if all_cos else None,
    }

    out = run.reconstructor_dir / "eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gate"], indent=2))
    print(f"reconstructor eval -> {out}")
    return report
