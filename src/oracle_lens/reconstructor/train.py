"""Stage 2 trainer (PLAN.md §5). Launch on the GPU node with e.g.

    accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
        -m oracle_lens.cli recon-train --config configs/qwen3-8b.yaml

Single-GPU / debug runs work with plain `oracle-lens recon-train`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from oracle_lens.activations.store import ActivationStore
from oracle_lens.activations.whitening import WhiteningTransform
from oracle_lens.config import Config
from oracle_lens.reconstructor.data import ReconDataset, build_pairs, collate_recon
from oracle_lens.reconstructor.model import (
    extract_at,
    load_reconstructor,
    recon_loss,
    save_recon_meta,
    unit,
)
from oracle_lens.rendering import load_subject_tokenizer
from oracle_lens.runs import RunPaths
from oracle_lens.training_utils import (
    MetricsLogger,
    cosine_lr_lambda,
    make_optimizer,
    set_seed,
)


@torch.no_grad()
def _quick_eval(model, batches, store, whitening, device) -> float:
    model.eval()
    cos = []
    for batch in batches:
        res = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        pred = unit(extract_at(res.values, batch["last_index"]))
        target = unit(whitening.whiten(store.read_rows(batch["store_rows"].numpy())).to(device))
        cos.append((pred.float() * target).sum(-1).mean().item())
    model.train()
    return sum(cos) / len(cos)


def train_reconstructor(
    cfg: Config,
    run: RunPaths,
    *,
    device_batch_size: int = 8,
    max_steps: int | None = None,
    eval_every: int = 200,
) -> Path:
    rcfg = cfg.reconstructor
    run.require(run.corpus, run.positions, run.whitening, run.activations_dir)
    set_seed(rcfg.seed)

    # World size from the launcher env, NOT torch.cuda.device_count(): a plain
    # single-process run on an 8-GPU box must still accumulate to batch_size.
    world_size = int(os.environ.get("WORLD_SIZE") or 1)
    grad_accum = max(1, rcfg.batch_size // (device_batch_size * world_size))
    accelerator = Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=grad_accum
    )
    tokenizer = load_subject_tokenizer(cfg.model.name)
    store = ActivationStore.open(run.activations_dir)
    whitening = WhiteningTransform.load(run.whitening)

    pairs = build_pairs(
        cfg, run.corpus, run.positions, tokenizer, split="reconstructor"
    )
    dataset = ReconDataset(pairs, tokenizer, rcfg.prompt_template)
    loader = DataLoader(
        dataset,
        batch_size=device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_recon(b, tokenizer.pad_token_id),
        num_workers=4,
        drop_last=True,
    )
    eval_pairs = build_pairs(
        cfg, run.corpus, run.positions, tokenizer, split="eval", limit=1024
    )
    eval_loader = DataLoader(
        ReconDataset(eval_pairs, tokenizer, rcfg.prompt_template),
        batch_size=device_batch_size,
        collate_fn=lambda b: collate_recon(b, tokenizer.pad_token_id),
    )
    eval_batches = list(eval_loader)

    # fp32 master weights are load-bearing: AdamW steps at lr ~1e-5 sit below
    # the bf16 ulp of Qwen-scale weights, so a bf16-loaded model silently
    # freezes its backbone (only the 10x-lr head trains). bf16 compute comes
    # from autocast via Accelerator(mixed_precision="bf16"); the checkpoint is
    # cast back to bf16 on save.
    model = load_reconstructor(
        cfg.model.name, layer_index=cfg.model.layer_index, dtype=torch.float32
    )
    model.gradient_checkpointing_enable()
    model.train()  # from_pretrained yields eval mode; checkpointing is gated on it
    optimizer = make_optimizer(model, rcfg.lr, rcfg.head_lr_mult)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    # The scheduler is deliberately NOT accelerator-prepared: prepared
    # schedulers advance num_processes ticks per optimizer step, which breaks
    # max_steps/warmup accounting. We keep everything in OPTIMIZER-step units
    # and step manually once per sync (len(loader) is per-process post-prepare).
    total_steps = max_steps or max(1, (len(loader) * rcfg.epochs) // grad_accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lr_lambda(total_steps, rcfg.warmup_steps)
    )
    metrics = MetricsLogger(
        run.reconstructor_dir / "metrics.jsonl",
        enabled=accelerator.is_main_process,
        wandb_run_name=f"{cfg.run_name}-recon-train",
        config=cfg,
    )
    if accelerator.is_main_process:
        print(f"recon-train: {len(pairs)} pairs, {total_steps} optim steps")

    step = 0
    done = False
    for _epoch in range(rcfg.epochs):
        if done:
            break
        for batch in loader:
            with accelerator.accumulate(model):
                res = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                pred = extract_at(res.values, batch["last_index"])
                target = whitening.whiten(
                    store.read_rows(batch["store_rows"].cpu().numpy())
                ).to(pred.device)
                loss = recon_loss(pred, target)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                scheduler.step()
                step += 1
                if step % 20 == 0:
                    metrics.log(step=step, loss=loss.item(), lr=scheduler.get_last_lr()[0])
                if step % eval_every == 0:
                    # ALL ranks run the eval forward — the model is FSDP-
                    # sharded, so a rank-0-only forward would deadlock on the
                    # weight all-gather collectives. Only rank 0 logs.
                    cos = _quick_eval(
                        model, eval_batches, store, whitening, accelerator.device
                    )
                    if accelerator.is_main_process:
                        metrics.log(step=step, eval_cosine=cos)
                        print(f"step {step}: eval cosine {cos:.4f}")
                if step >= total_steps:
                    done = True
                    break

    accelerator.wait_for_everyone()
    out_dir = run.reconstructor_dir / "checkpoint"
    # get_state_dict is a collective under FSDP — every rank must enter it;
    # only rank 0 touches the filesystem.
    state = accelerator.get_state_dict(model)
    state = {  # save in bf16: downstream loads bf16, and fp32 doubles the artifact
        k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
        for k, v in state.items()
    }
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(out_dir, state_dict=state)
        tokenizer.save_pretrained(out_dir)
        save_recon_meta(out_dir, cfg, run.whitening)
        print(f"reconstructor -> {out_dir}")
    metrics.finish()
    return out_dir
