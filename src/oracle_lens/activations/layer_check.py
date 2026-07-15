"""Stage 1c: J-lens layer sanity check (PLAN.md §4.3) — de-risks l*.

Fits a Jacobian lens for the subject model with the vendored jlens package and
renders slice visualizations for a handful of prompts. A human then eyeballs
whether layer l* sits inside the band where readouts are interpretable, and
moves l* if not. The fitted lens is saved and reused by the §8.1 J-lens
baseline (eval/baselines.py).
"""

from __future__ import annotations

from pathlib import Path

import torch

from oracle_lens.config import Config

LENS_FILE = "lens.pt"

_DEFAULT_PROMPTS = 200  # jlens README: ~100 suffices; quality saturates fast


def run_layer_check(
    cfg: Config,
    out_dir: Path,
    *,
    n_fit_prompts: int = _DEFAULT_PROMPTS,
    slice_prompts: list[str] | None = None,
    dim_batch: int = 16,
) -> Path:
    import jlens
    from jlens.examples import EXAMPLES, load_wikitext_prompts
    from jlens.vis import build_page, compute_slice
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = jlens.from_hf(hf_model, tokenizer)

    lens_path = out_dir / LENS_FILE
    if lens_path.exists():
        lens = jlens.JacobianLens.load(lens_path)
        print(f"layer-check: reusing fitted lens at {lens_path}")
    else:
        prompts = load_wikitext_prompts(n_fit_prompts)
        lens = jlens.fit(
            model,
            prompts,
            dim_batch=dim_batch,
            checkpoint_path=str(out_dir / "lens_ckpt.pt"),
        )
        lens.save(lens_path)

    for i, prompt in enumerate(slice_prompts or EXAMPLES[:4]):
        data = compute_slice(model, lens, prompt, mask_display=True)  # Qwen: mask punct
        html, _, _ = build_page(
            data,
            prompt,
            title=f"{cfg.model.name} slice {i} (l* candidate = {cfg.model.layer_index})",
            description="M1 layer check: confirm l* sits in the interpretable band",
            mode="embed",
        )
        (out_dir / f"slice_{i}.html").write_text(html)
    print(
        f"layer-check: slice pages in {out_dir} — eyeball that layer "
        f"{cfg.model.layer_index} readouts are interpretable (PLAN.md §4.3)"
    )
    return lens_path
