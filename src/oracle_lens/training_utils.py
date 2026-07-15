"""Small shared pieces for the two single-node trainers (reconstructor SFT,
oracle SFT). Distribution strategy (FSDP/DDP/single-GPU) is owned by
`accelerate launch` + configs/accelerate/*.yaml — this code is agnostic."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def cosine_lr_lambda(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return f


def make_optimizer(model: torch.nn.Module, lr: float, head_lr_mult: float = 1.0):
    """AdamW with the value head (if any) on a higher LR (PLAN.md §5)."""
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if "value_head" in name else backbone).append(p)
    groups = [{"params": backbone, "lr": lr}]
    if head:
        groups.append({"params": head, "lr": lr * head_lr_mult})
    return torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))


class MetricsLogger:
    """Append-only metrics.jsonl per run, mirrored to wandb when configured.

    wandb activates when a run name is given AND WANDB_API_KEY is set (export
    it from ~/.wandb_key on the box; never pass it in argv). Project/entity
    come from WANDB_PROJECT / WANDB_ENTITY, defaulting to oracle-lens /
    octahedral-systems. Local JSONL is always written regardless, so runs
    remain inspectable without network.
    """

    def __init__(
        self,
        path: Path,
        enabled: bool = True,
        *,
        wandb_run_name: str | None = None,
        config: Any = None,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self._wandb = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
        if enabled and wandb_run_name:
            if os.environ.get("WANDB_API_KEY"):
                from dataclasses import asdict, is_dataclass

                import wandb

                self._wandb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "oracle-lens"),
                    entity=os.environ.get("WANDB_ENTITY", "octahedral-systems"),
                    name=wandb_run_name,
                    config=asdict(config) if is_dataclass(config) else config,
                )
            else:
                print(
                    "WARNING: WANDB_API_KEY not set — training run is NOT "
                    "tracked in wandb (ENV.md requires tracking on GPU boxes; "
                    "export WANDB_API_KEY=$(cat ~/.wandb_key) before launch)",
                    flush=True,
                )

    def log(self, **fields: Any) -> None:
        if not self.enabled:
            return
        fields.setdefault("t", time.time())
        with self.path.open("a") as f:
            f.write(json.dumps(fields) + "\n")
        if self._wandb is not None:
            step = fields.get("step")
            self._wandb.log(
                {k: v for k, v in fields.items() if k not in ("t", "step")}, step=step
            )

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
