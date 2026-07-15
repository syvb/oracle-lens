"""Stage 2 trainer (PLAN.md §5). Launch on the GPU node with e.g.

    accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
        -m oracle_lens.cli recon-train --config configs/qwen3-8b.yaml

Single-GPU / debug runs work with plain `oracle-lens recon-train`.
"""

from __future__ import annotations

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

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=max(
            1, rcfg.batch_size // (device_batch_size * max(1, torch.cuda.device_count()))
        ),
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

    model = load_reconstructor(cfg.model.name, layer_index=cfg.model.layer_index)
    model.gradient_checkpointing_enable()
    optimizer = make_optimizer(model, rcfg.lr, rcfg.head_lr_mult)
    total_steps = max_steps or (len(loader) * rcfg.epochs) // accelerator.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lr_lambda(total_steps, rcfg.warmup_steps)
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    metrics = MetricsLogger(
        run.reconstructor_dir / "metrics.jsonl", enabled=accelerator.is_main_process
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
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % 20 == 0:
                    metrics.log(step=step, loss=loss.item(), lr=scheduler.get_last_lr()[0])
                if step % eval_every == 0 and accelerator.is_main_process:
                    cos = _quick_eval(
                        accelerator.unwrap_model(model), eval_batches, store, whitening,
                        accelerator.device,
                    )
                    metrics.log(step=step, eval_cosine=cos)
                    print(f"step {step}: eval cosine {cos:.4f}")
                if step >= total_steps:
                    done = True
                    break

    accelerator.wait_for_everyone()
    out_dir = run.reconstructor_dir / "checkpoint"
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            out_dir, state_dict=accelerator.get_state_dict(model)
        )
        tokenizer.save_pretrained(out_dir)
        save_recon_meta(out_dir, cfg, run.whitening)
        print(f"reconstructor -> {out_dir}")
    return out_dir
