"""Stage 4 oracle SFT + the alpha mini-sweep (PLAN.md §7.1-7.2).

The oracle is a full fine-tune of the subject model that perceives an
activation through one injected pseudo-token (oracle/injection.py) and lists
K phrases of N tokens. Targets are the teacher decompositions; the SFT
checkpoint is the primary de-risking milestone.

Launch like recon-train:
    accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
        -m oracle_lens.cli oracle-sft --config configs/qwen3-8b.yaml
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from oracle_lens.activations.store import ActivationStore
from oracle_lens.activations.whitening import WhiteningTransform
from oracle_lens.config import Config
from oracle_lens.oracle.data import (
    OracleExample,
    OracleSftDataset,
    build_oracle_examples,
    collate_oracle,
)
from oracle_lens.oracle.injection import (
    InjectionState,
    register_injection_hook,
    resolve_alpha,
    scale_for_injection,
)
from oracle_lens.oracle.prompts import oracle_token_meta
from oracle_lens.rendering import load_subject_tokenizer
from oracle_lens.runs import RunPaths
from oracle_lens.training_utils import (
    MetricsLogger,
    cosine_lr_lambda,
    make_optimizer,
    set_seed,
)

META_FILE = "oracle_lens_oracle.yaml"
_HOLDOUT_FRAC = 50  # 1/50 = 2% of teacher examples held out for sweep/plateau CE


def _is_holdout(ex: OracleExample) -> bool:
    return int(hashlib.sha256(f"holdout|{ex.store_row}".encode()).hexdigest(), 16) % _HOLDOUT_FRAC == 0


def _batch_vectors(store, whitening, rows, alpha, device, dtype):
    w = whitening.whiten(store.read_rows(rows.cpu().numpy()))
    return scale_for_injection(w, alpha).to(device, dtype)


@torch.no_grad()
def _holdout_ce(model, batches, store, whitening, alpha, state, device) -> float:
    model.eval()
    losses = []
    for batch in batches:
        state.vectors = _batch_vectors(
            store, whitening, batch["store_rows"], alpha, device, torch.bfloat16
        )
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        losses.append(out.loss.item())
    state.vectors = None
    model.train()
    return sum(losses) / len(losses)


def train_oracle_sft(
    cfg: Config,
    run: RunPaths,
    *,
    device_batch_size: int = 4,
    max_steps: int | None = None,
    alpha: float | None = None,
    out_dir: Path | None = None,
    eval_every: int = 200,
) -> Path:
    ocfg = cfg.oracle
    run.require(run.teacher, run.activations_dir, run.whitening)
    set_seed(ocfg.seed)
    alpha = alpha if alpha is not None else resolve_alpha(cfg)
    out_dir = out_dir or run.oracle_sft_dir / "checkpoint"

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=max(
            1, ocfg.batch_size // (device_batch_size * max(1, torch.cuda.device_count()))
        ),
    )
    tokenizer = load_subject_tokenizer(cfg.model.name)
    meta = oracle_token_meta(tokenizer, cfg)
    store = ActivationStore.open(run.activations_dir)
    whitening = WhiteningTransform.load(run.whitening)

    examples = build_oracle_examples(cfg, run.teacher)
    train_ex = [e for e in examples if not _is_holdout(e)]
    holdout_ex = [e for e in examples if _is_holdout(e)][:512]
    loader = DataLoader(
        OracleSftDataset(train_ex, tokenizer, cfg, meta),
        batch_size=device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_oracle(b, tokenizer.pad_token_id),
        num_workers=4,
        drop_last=True,
    )
    holdout_batches = list(
        DataLoader(
            OracleSftDataset(holdout_ex, tokenizer, cfg, meta),
            batch_size=device_batch_size,
            collate_fn=lambda b: collate_oracle(b, tokenizer.pad_token_id),
        )
    )

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    optimizer = make_optimizer(model, ocfg.lr)
    total_steps = max_steps or (len(loader) * ocfg.epochs) // accelerator.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lr_lambda(total_steps, ocfg.warmup_steps)
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    # Hook AFTER prepare: FSDP wraps modules but the embedding module instance
    # (and thus the hook) survives; scan runs inside the hook on live ids.
    state = InjectionState()
    register_injection_hook(accelerator.unwrap_model(model), meta, state)

    metrics = MetricsLogger(
        (out_dir.parent / "metrics.jsonl"),
        enabled=accelerator.is_main_process,
        wandb_run_name=f"{cfg.run_name}-{out_dir.parent.name}-{out_dir.name}"
        if out_dir.name != "checkpoint"
        else f"{cfg.run_name}-oracle-sft",
        config=cfg,
    )
    if accelerator.is_main_process:
        print(
            f"oracle-sft: {len(train_ex)} examples ({len(holdout_ex)} holdout), "
            f"{total_steps} steps, alpha={alpha:.1f}"
        )

    step = 0
    done = False
    for _epoch in range(ocfg.epochs):
        if done:
            break
        for batch in loader:
            with accelerator.accumulate(model):
                state.vectors = _batch_vectors(
                    store, whitening, batch["store_rows"], alpha,
                    accelerator.device, torch.bfloat16,
                )
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % 20 == 0:
                    metrics.log(step=step, loss=out.loss.item(), lr=scheduler.get_last_lr()[0])
                if step % eval_every == 0 and accelerator.is_main_process:
                    ce = _holdout_ce(
                        accelerator.unwrap_model(model), holdout_batches, store,
                        whitening, alpha, state, accelerator.device,
                    )
                    metrics.log(step=step, holdout_ce=ce)
                    print(f"step {step}: holdout CE {ce:.4f}")
                if step >= total_steps:
                    done = True
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            out_dir, state_dict=accelerator.get_state_dict(model)
        )
        tokenizer.save_pretrained(out_dir)
        final_ce = _holdout_ce(
            accelerator.unwrap_model(model), holdout_batches, store, whitening,
            alpha, state, accelerator.device,
        )
        (out_dir / META_FILE).write_text(
            yaml.safe_dump(
                {
                    "kind": "oracle_lens_oracle",
                    "base_model": cfg.model.name,
                    "alpha": float(alpha),
                    "prompt_template": cfg.oracle.prompt_template,
                    "whitening_path": str(run.whitening),
                    "reconstructor": str(run.reconstructor_dir / "checkpoint"),
                    "holdout_ce": final_ce,
                },
                sort_keys=True,
            )
        )
        print(f"oracle SFT -> {out_dir} (holdout CE {final_ce:.4f})")
    metrics.finish()
    return out_dir


def alpha_sweep(
    cfg: Config, run: RunPaths, *, steps: int = 2000, device_batch_size: int = 4
) -> dict:
    """PLAN.md §7.1: short SFT probes at each alpha multiplier; pick by
    held-out teacher-imitation CE. Alpha is documented as touchy — run this
    before committing to the main run."""
    base = math.sqrt(cfg.model.d_model)  # sweep multiplies the ambient scale
    results: dict[str, float] = {}
    for mult in cfg.oracle.alpha_sweep:
        alpha = base * mult
        out = run.oracle_sft_dir / "sweep" / f"alpha_{mult:g}"
        train_oracle_sft(
            cfg, run, device_batch_size=device_batch_size, max_steps=steps,
            alpha=alpha, out_dir=out,
        )
        results[f"{mult:g}"] = yaml.safe_load((out / META_FILE).read_text())["holdout_ce"]
    best = min(results, key=results.get)  # type: ignore[arg-type]
    report = {"base_alpha": base, "holdout_ce_by_multiplier": results, "best_multiplier": best}
    (run.oracle_sft_dir / "alpha_sweep.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"set oracle.alpha: {base * float(best):.1f} in the config, then run the full SFT")
    return report
