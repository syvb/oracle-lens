"""Stage 2 model: phrase -> whitened direction, via the vendored
nla.models.NLACriticModel (truncated backbone through block l*, final-LN
stripped, Linear(d, d) value head). We add nothing structural — only the
whitened-space objective and the encode API downstream stages use.

Loss = MSE between unit-normalized whitened prediction and unit-normalized
whitened target = 2(1 - cos) in the whitened metric (PLAN.md §5). The head's
raw output is interpreted as whitened coordinates directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from nla.models import NLACriticModel

META_FILE = "oracle_lens_recon.yaml"


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(x.dtype)


def load_reconstructor(
    base_or_checkpoint: str | Path, *, layer_index: int | None = None, dtype=torch.bfloat16
) -> NLACriticModel:
    """Fresh init: pass the base model name + layer_index (truncates to blocks
    0..layer_index inclusive). Trained checkpoint: pass its dir, no layer_index
    (config.json already carries the truncation)."""
    return NLACriticModel.from_pretrained(
        str(base_or_checkpoint), nla_num_layers=layer_index, torch_dtype=dtype
    )


def recon_loss(pred: torch.Tensor, target_w: torch.Tensor) -> torch.Tensor:
    """pred: head output at the extraction position [B, d] (whitened coords);
    target_w: whitened raw activations [B, d] (normalized here)."""
    return (unit(pred).float() - unit(target_w).float()).square().sum(dim=-1).mean()


def extract_at(values: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
    """values: [B, T, d] -> [B, d] at each sample's suffix-anchor position."""
    b = torch.arange(values.shape[0], device=values.device)
    return values[b, last_index.to(values.device)]


@torch.no_grad()
def encode_phrases(
    model: NLACriticModel,
    tokenizer: Any,
    phrases: list[str],
    template: str,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Unit-normalized whitened directions [len(phrases), d] — the API used by
    dictionary building, oracle decode, and every baseline."""
    from oracle_lens.reconstructor.data import collate_recon

    model.eval()
    out: list[torch.Tensor] = []
    for lo in range(0, len(phrases), batch_size):
        chunk = phrases[lo : lo + batch_size]
        items = [
            {
                "input_ids": tokenizer(template.format(phrase=p), add_special_tokens=False)[
                    "input_ids"
                ],
                "store_row": 0,
            }
            for p in chunk
        ]
        batch = collate_recon(items, tokenizer.pad_token_id)
        res = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        pred = extract_at(res.values, batch["last_index"])
        out.append(unit(pred).float().cpu())
    return torch.cat(out)


def save_recon_meta(out_dir: Path, cfg: Any, whitening_path: Path) -> None:
    (Path(out_dir) / META_FILE).write_text(
        yaml.safe_dump(
            {
                "kind": "oracle_lens_reconstructor",
                "base_model": cfg.model.name,
                "layer_index": cfg.model.layer_index,
                "prompt_template": cfg.reconstructor.prompt_template,
                "whitening_path": str(whitening_path),
                "output_space": "whitened_unit",
            },
            sort_keys=True,
        )
    )


def load_recon_meta(checkpoint_dir: Path) -> dict:
    return yaml.safe_load((Path(checkpoint_dir) / META_FILE).read_text())
