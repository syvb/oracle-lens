# oracle-lens

Reconstruction of the **oracle lens** interpretability method (from the
appendix *"Extending the Jacobian lens to multi-token concepts"* of the
[workspace paper](https://transformer-circuits.pub/2026/workspace/)) for
**Qwen/Qwen3-8B**. Full experiment plan, definitions, gates, and provenance
rules: **[PLAN.md](PLAN.md)** — read it first; this README is only the
runbook.

Given a residual-stream activation, the trained oracle proposes K free-form
phrases of N tokens whose reconstructor-directions linearly explain the
activation (non-negative least squares, whitened space):

```python
decoder = OracleDecoder(cfg, run.oracle_sft_dir / "checkpoint")
result = decoder.decode(activation, n=8, k=8)
result.ranked   # [(phrase, coefficient), ...]
result.fve      # whitened fraction of variance explained
```

## Layout

- `src/oracle_lens/` — the pipeline (one CLI subcommand per stage, below)
- `third_party/` — vendored scaffolding at pinned SHAs (`nla`, `jlens`);
  see `third_party/README.md` for what is reused and the do-not-modify rule
- `configs/qwen3-8b.yaml` — every [paper]/[choice] knob; each run pins a copy
- `runs/<run_name>/` — all artifacts (gitignored); layout in `runs.py`
- `tests/` — CPU-only, offline; tiny local tokenizer + tiny random Qwen3

## Setup

Dev box (CPU, tests only): `uv sync --group dev && uv run pytest`

GPU node (8x H100/A100-80GB):

```bash
uv sync --group dev
uv pip install vllm         # stage 0 only; brings its own CUDA torch — fine
```

## Runbook (one milestone per line, gates in parentheses)

```bash
O="uv run oracle-lens"                              # add --config/--runs-root as needed
$O prompts && $O generate && $O verify              # M0 (token-identity gate; verify exits 1 on fail)
$O positions && $O split
$O collect && $O whiten && $O layer-check           # M1 (conditioning; eyeball slice_*.html for l*)
accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
    -m oracle_lens.cli recon-train                  # M2 train
$O recon-eval                                       # M2 gate (controls separation, continuation-FVE)
$O dictionary && $O teacher                         # M3 (teacher FVE >= 0.15)
$O alpha-sweep                                      # pick alpha, set oracle.alpha in config
accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
    -m oracle_lens.cli oracle-sft                   # M4 train
$O oracle-eval                                      # M4 gate (>= 70% of teacher FVE, format >= 95%)
$O baselines && $O gallery                          # §8.1 table + §8.2 gallery (the real decision)
```

M5 (optional GRPO, gated on the M4 gallery): the trainer is the vendored nla
repo's Miles + SGLang stack, run in its own environment; this repo owns only
the reward. Runbook + reward: `src/oracle_lens/oracle/grpo.py`.

## Where results land

`runs/<name>/reconstructor/eval.json` (M2) · `runs/<name>/teacher/gate.json`
(M3) · `runs/<name>/oracle_sft/eval.json` (M4) · `runs/<name>/eval/table.json`
(§8.1) · `runs/<name>/eval/gallery.md` (§8.2, hand-score the checkboxes).
