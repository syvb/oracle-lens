# EXECUTION.md — running the oracle-lens experiment on Vast.ai

**Audience:** an agent on this dev box (`/home/debian/oracle-lens`) executing
the experiment end to end. Three documents divide the work:

- **PLAN.md** — the science: stages, definitions, gates, expectations. Read fully first.
- **README.md** — the runbook: which command implements which milestone.
- **This file** — operations: renting GPUs, moving artifacts, budgets, recovery.
- **`~/ENV.md`** — the machine/Vast reference. Its **Gotchas (§5, §10)** are
  binding; the worst ones are restated inline where they bite.

The pipeline is sequential (M0 → M4); each stage reads the previous stage's
artifacts. You will rent a fresh box per stage-group, pull artifacts from HF,
run, push artifacts to HF, destroy the box. Nothing irreplaceable ever lives
only on a rented box.

---

## 0. Non-negotiable ground rules

1. **Instance ownership (ENV.md §3, verbatim rule applies).** Only ever
   operate on instances you created for this task. Immediately after every
   `vastai create instance`, append a line to
   `/home/debian/oracle-lens/ops/vast_ledger.tsv` (NOT under `runs/`, which is
   gitignored — the ledger must be committable):
   `contract_id  label  gpu  $/hr  bw_$per_TB_down/up  created_at  purpose`.
   Destroy only IDs from this ledger; mark them destroyed with a timestamp.
   **Always destroy boxes when their stage's artifacts are safely on HF** —
   verify the HF upload (list files + sizes) BEFORE destroying.
2. **wandb on every training run.** `export WANDB_API_KEY=$(cat /root/.wandb_key)`
   in the launch environment (never in argv). The trainers auto-init wandb
   (project `oracle-lens`, entity `octahedral-systems`) and print a loud
   WARNING if the key is missing — treat that warning as a launch failure.
3. **Secrets:** copy token *files* to boxes (`umask 077`), read via `$(cat ...)`
   into env vars; never `echo $TOKEN`. ENV.md's own token-copy recipe expands
   `$(cat ~/.hf_token)` inside the local ssh argv — that exact pattern is
   blessed there and acceptable; what's forbidden is echoing token values or
   passing them as long-lived flags (e.g. wandb keys on training command
   lines — the env var is picked up automatically).
4. **Budget:** target ≤ **$120** through M4 including retries; hard stop and
   report to the user if the ledger + forecast exceeds **$250**. Keep the
   ledger current — it is the budget instrument.
5. **Human checkpoints — stop and ask, do not proceed:**
   - after `layer-check` (a human eyeballs the slice pages before M2),
   - on any failed gate (M2/M3/M4) *after* running the plan's prescribed
     first remedy once,
   - after the M4 gallery (the M5 go/no-go is the user's call).

---

## 1. Stage map, budgets, artifacts

GPU classes (shop per stage, §2): **[infer]** = any ≥48 GB Ampere-or-newer
inference box; **[cheap-infer]** = ≥24 GB Ampere (3090/4090/A5000);
**[train-8B]** = full 8B fine-tune: 1×141 GB (H200), or 2–4×80 GB (H100/A100
SXM) with the FSDP config; **[train-6B]** = the truncated reconstructor
(~5.9 B params): 1×H200 comfortable, 1×H100-80GB tight-but-OK
(bump `--device-batch-size` down if OOM).

| # | Commands | Box | Est. GPU time | Est. cost | Artifact → HF | Gate |
|---|---|---|---|---|---|---|
| 0a | `prompts`, `generate` (smoke `--limit 500`, then full), `verify` | [infer] | 3–5 h | $2–5 | `corpus/` (prompts, corpus.parquet + sidecar) | verify exits 0 |
| 0b | `positions`, `split` | same box | minutes | — | positions.parquet | split counts sane |
| 1a | `collect`, `whiten` | [infer] (same box as 0 is fine) | 1.5–3 h | $1–3 | `activations/` (~8 GB), whitening.pt | var_mean ∈ (0.8, 1.2) |
| 1b | `layer-check` | [infer], 80 GB preferred (jlens fit does backward) | ~1 h | $1–3 | `layer_check/` (lens.pt + slice html) | **HUMAN: confirm l\*** |
| 2 | `recon-train` (smoke `--max-steps 20`, then full), `recon-eval` | [train-6B] | 4–7 h | $12–25 | `reconstructor/checkpoint` (~12 GB), eval.json | §5 gate in eval.json |
| 3 | `dictionary`, `teacher` | [infer] 48–80 GB | 3–6 h | $3–8 | `dictionary/` (~4 GB), `teacher/` | teacher FVE ≥ 0.15 |
| 4a | `alpha-sweep` (4 arms × 2000 steps) | [train-8B]; arms parallelize on separate boxes | 4×1.5–2.5 h | $15–30 | alpha_sweep.json | pick α, commit config |
| 4b | `oracle-sft` (smoke, then full), `oracle-eval` | [train-8B] | 4–7 h | $15–25 | `oracle_sft/checkpoint` (~16 GB), eval.json | ≥70% teacher FVE, ≥95% format |
| 4c | `baselines`, `gallery` | **80 GB** — `gallery` holds THREE models resident (subject 8B + oracle 8B + truncated reconstructor ≈ 44 GB weights before KV cache); cheapest correct play is running 4c on the still-alive 4b box before destroying it. Only `baselines --skip-jlens` fits a 24–48 GB box. | 2–4 h | $2–8 | `eval/` (table.json, gallery.md) | **HUMAN: gallery review** |

Totals: ~$50–100 GPU + bandwidth ≈ well inside the $120 target; wall-clock
2–4 days mostly serialized (α-sweep in parallel saves ~6 h). Time estimates
assume Hopper/Ada-class throughput; on a 48 GB Ampere (A6000) multiply the
0a/1a numbers by 2–3× — since $/result is then similar, prefer H100-class for
those two stages when hourly prices are close.

**Artifact home (create once, §3):** private HF dataset repo
**`syvb/oracle-lens-qwen3-8b-artifacts`** mirroring `runs/qwen3-8b-v1/`
subtree-per-stage, plus model repos **`syvb/oracle-lens-qwen3-8b-recon`** and
**`syvb/oracle-lens-qwen3-8b-oracle-sft`**. Keep `.meta.yaml` sidecars next to
their parquets when copying (HF convention in ENV.md §6).

---

## 2. Shopping for boxes — requirements, not models

Don't fix on a GPU model; pick the best offer meeting the stage's floor:

- **VRAM floors:** [cheap-infer] 24 GB · [infer] 48 GB · [train-6B] 96 GB
  single-card (or 2×80 FSDP) · [train-8B] 141 GB single-card (or 2–4×80 FSDP).
  Single-card H200 for training avoids all FSDP moving parts and per ENV.md §7
  fits an 8B full FT + Adam; multi-card 80 GB boxes are often cheaper per hour
  — use `configs/accelerate/fsdp_8gpu.yaml` with `num_processes` edited to
  match, and smoke-test 20 steps before trusting it.
- **Compute floor: Ampere (sm_86)** — bf16 everywhere; no Turing. Avoid B200
  (needs a cu128+ stack; fragile per ENV.md §3) unless nothing else exists.
- **vLLM (stage 0) needs compute capability ≥ 8.0** — same Ampere floor.
- **Search & rank by TOTAL cost** — ENV.md §3's raw-JSON ranking snippet,
  sorting on `(inet_down_cost + inet_up_cost, dph_total)`. Bandwidth traps,
  all three from ENV.md, all binding: rank BOTH directions; missing field ≠
  free (verify on the live contract: `internet_*_cost_per_tb`); advertised
  speed can be fiction.
- **Speed-test egress FIRST** on every fresh box (ENV.md §3): the
  `curl ... resolve/main/model-....safetensors` probe; **reject < 150 MB/s**
  (destroy, re-rent). Every stage here pulls 16–30 GB.
- **Disk:** 0/1: 150 GB · 2: 200 GB · 3: 150 GB · 4a/b: 250 GB · 4c: 150 GB.
- **Reliability > 0.99** for anything ≥ 2 h. Geolocation unrestricted.
- **Image:** stock `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` everywhere
  (we bring our own venv; the image's conda torch is unused). The image choice
  matters only for M5 (ENV.md §8) — not part of this execution.

Create + connect + boot gotchas (stopped-after-pull → `vastai start`; pubkey
denied → `recycle`, try proxy host; `scp -P` capital): ENV.md §3 verbatim.

---

## 3. One-time prep (before renting anything)

1. **WildChat gating:** `allenai/WildChat-1M` requires accepted terms. From the
   dev box, verify:
   `.venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('allenai/WildChat-1M','README.md',repo_type='dataset',token=open('/home/debian/.hf_token').read().strip())"`
   — a 403 means: tell the user to accept the terms for `syvb` in a browser;
   stop until resolved.
2. **Create artifact repos** (dev box, HfApi with the syvb token):
   `create_repo("syvb/oracle-lens-qwen3-8b-artifacts", repo_type="dataset", private=True, exist_ok=True)` and the two model repos likewise.
3. **Commit ledger + any config edits** to the repo before each stage so every
   box gets identical code via rsync (ENV.md §10: rsync to EVERY fresh box).

---

## 4. Per-box lifecycle (repeat for every stage-group)

```bash
# on dev box; $ID from create, $IP/$PORT from `vastai show instance $ID --raw`
# 0) search per §2, create with a unique label, LOG THE CONTRACT ID (§0.1)
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk <per-§2> --ssh --direct --label oracle-lens-<stage>-$(date +%m%d) \
  --onstart-cmd 'touch ~/.no_auto_tmux; sleep infinity'
# 1) wait for running (poll actual_status; if stuck "stopped": vastai start instance $ID)
# 2) SPEED TEST (reject <150 MB/s — destroy & re-rent)
# 3) tokens + code
ssh -p $PORT root@$IP "umask 077; printf '%s' '$(cat ~/.hf_token)' > /root/.hf_token; \
  printf '%s' '$(cat ~/.wandb_key)' > /root/.wandb_key"
rsync -az -e "ssh -p $PORT" --exclude .git --exclude '.venv*' --exclude runs \
  --exclude __pycache__ /home/debian/oracle-lens/ root@$IP:/workspace/oracle-lens/
# 4) env — NOTE: `uv pip install`, NOT `uv sync`. The project's uv.lock pins
#    CPU-only torch for the dev box ([tool.uv.sources] index pin); `uv pip`
#    ignores that pin and resolves CUDA torch from PyPI. Install the two
#    vendored editables FIRST so `nla`/`jlens` resolve locally, and pin
#    transformers to the dev-tested minor (5.14.*; the vendored-repo 4.x pin
#    advice in ENV.md §4 does NOT apply to this pipeline — see third_party/README.md).
#    The install pulls 3-6 GB of CUDA wheels — WELL over the 120-s foreground
#    limit (ENV.md §5.2): run it DETACHED with a sentinel + background poller,
#    exactly like step 7. Write the script to a file and scp it (heredoc rule).
#    Script body:
#      set -e; curl -LsSf https://astral.sh/uv/install.sh | sh
#      cd /workspace/oracle-lens; ~/.local/bin/uv venv /workspace/venv --python 3.12
#      ~/.local/bin/uv pip install --python /workspace/venv/bin/python \
#        -e third_party/natural_language_autoencoders -e third_party/jacobian-lens \
#        -e . "transformers==5.14.*" pytest scipy
#      touch /workspace/DONE_env
# stage-0 box only, AFTER the above (also detached): uv pip install ... vllm,
#    then RE-VERIFY the pins — vllm may move transformers/torch, and this box
#    writes the template fingerprint that M1 hard-checks; a silent bump here
#    dead-ends the pipeline at collect with a full M0 redo:
#      /workspace/venv/bin/python -c "import transformers; assert transformers.__version__.startswith('5.14')"
#    (if the assert fails: uv pip install "transformers==5.14.*" again and re-run pytest)
# 5) sanity: cuda available, tests pass, HF reachable
ssh -p $PORT root@$IP '/workspace/venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())" \
  && cd /workspace/oracle-lens && /workspace/venv/bin/python -m pytest -q'
# 6) pull upstream artifacts from HF into /workspace/runs/qwen3-8b-v1/
#    (snapshot_download with ONE allow_patterns list, ls-verify after; 16-30 GB
#    -> DETACHED + poller, same as step 4), 7) run the stage DETACHED with a sentinel:
ssh -p $PORT root@$IP 'cd /workspace/oracle-lens && export HF_HOME=/workspace/hf \
  HF_TOKEN=$(cat /root/.hf_token) WANDB_API_KEY=$(cat /root/.wandb_key) && \
  nohup /workspace/venv/bin/python -m oracle_lens.cli <stage> --runs-root /workspace/runs \
    > /workspace/<stage>.log 2>&1 < /dev/null && touch /workspace/DONE_<stage> &'
# 8) poll with a BACKGROUND poller (run_in_background): every ~100 s check
#    `test -f DONE_<stage>` OR the process died (bracket trick: pgrep -f "oracle_len[s]"),
#    cap iterations (e.g. seq 1 200 ≈ 5.5 h — long stages may outlive one
#    poller: cap expiry is NOT failure, just re-arm a fresh poller),
#    tail the log through `tr "\r" "\n" | tail -5`.
# 9) push artifacts to HF (upload_folder — multi-GB, so DETACHED + poller like
#    step 4), VERIFY the upload listing (names + sizes) from the dev box,
#    record the env via `~/.local/bin/uv pip freeze --python /workspace/venv/bin/python > env.txt`
#    (uv venvs ship NO pip — `pip freeze` errors) + push it too, THEN destroy:
echo y | vastai destroy instance $ID   # exact contract id from the ledger
```

ENV.md gotchas that WILL bite here if ignored: 120-s foreground limit (never
foreground a training command or a >110 s sleep); `pkill -f` self-match
(bracket every pattern); `set -e` + pipe hides failures (check artifacts
exist, not exit codes); nested-heredoc breakage (scp scripts, don't inline);
one `hf download`-style call per file when pulling selectively (multi
`--include` silently drops patterns) — with `snapshot_download`, pass a single
`allow_patterns` list and `ls`-verify afterwards.

---

## 5. Stage notes (what README doesn't say)

Every stage: run from `/workspace/oracle-lens`, always `--runs-root /workspace/runs`,
config `configs/qwen3-8b.yaml` (edit on the DEV box, commit, rsync — never
hand-edit on a GPU box or provenance dies).

**M0 (`prompts → generate → verify → positions → split`).**
- Smoke first: `generate --limit 500` (~10 min) then `verify`; only then the
  full 100k run (detached; expect 2–4 h on one 48–141 GB card). Then re-run
  `verify` (exit 0 = gate pass; exit 1 = STOP, template drift).
- `prompts` streams WildChat (~several GB download) — run it on the GPU box
  (free bandwidth host), not the dev box.
- Record the exact `vllm` version in the pushed `env.txt`; if generation ever
  needs re-running, use the SAME version (sampler changes = different corpus).
- Sanity-eyeball 5 transcripts (decode a few `response_ids`): fluent English,
  no thinking tags, no CJK free-association.

**M1 (`collect → whiten → layer-check`).**
- `collect` re-checks the template fingerprint against the corpus sidecar and
  hard-fails on drift — do not "fix" by regenerating the sidecar; that defeats
  the on-policy guarantee. Investigate (wrong transformers version? modified
  config?) instead.
- `whiten` prints held-out stats; gate: `var_mean` in (0.8, 1.2), finite
  condition number. Push whitening.pt + the printed stats.
- `layer-check` fits a jlens lens (no pre-fitted Qwen3-8B asset exists on the
  Hub) and writes `slice_*.html`. Pull those to the dev box, show the user,
  and STOP for confirmation of `layer_index` (§0.5). If l\* moves: edit
  config on dev box, commit, and REDO collect + whiten (activations are
  layer-specific). The lens.pt is reused by the §8.1 J-lens baseline — push it.

**M2 (`recon-train → recon-eval`).**
- Single-GPU launch is the default path
  (`oracle-lens recon-train --device-batch-size 16`); on a multi-80GB box use
  `accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml`
  (edit `num_processes`). ALWAYS smoke `--max-steps 20` first on the rented
  box: catches OOM/env issues for cents (ENV.md §7: fixed overhead dominates).
- Watch wandb: `loss` falling, periodic `eval_cosine` rising. If
  `eval_cosine` is still clearly improving at the end of training, the M2
  gate says add data before proceeding. Note that more epochs on the SAME
  300k pairs is not "more data" (each position keeps its keyed-RNG phrase
  length across epochs); real added data means more positions — bump
  `positions.per_response`, re-run `positions → split → collect → whiten`
  on the M0/M1 box class, and retrain. That's a ~$5 detour, but consult the
  user first since it regenerates the store (row-alignment rule, §6), and if
  you do it: DELETE `reconstructor/`, `dictionary/`, `teacher/` locally AND
  on HF, and push the new positions/activations/whitening OVER the old ones
  (same paths, replaced — never side by side), so no later box can pull a
  stale row-index generation.
- Gate = `reconstructor/eval.json` (controls separation, continuation-FVE at
  N≤8). On fail after one remedy → STOP per §0.5.

**M3 (`dictionary → teacher`).**
- Both are throughput jobs; a 48 GB box is fine (dictionary encode is the
  truncated model; teacher NN-OMP is pure matmuls on ≤4 GB of directions).
- Gate: `teacher/gate.json` `fve_mean_overall ≥ 0.15`. On fail, PLAN.md §6.2
  order: more reconstructor data → bigger dictionary → ridge λ sweep (edit
  `whitening.ridge_frac` ±1 order, redo `whiten`+M2? NO — λ changes whitening,
  which invalidates the reconstructor: full redo from `whiten`; this is the
  expensive branch, get user sign-off first).

**M4 (`alpha-sweep → oracle-sft → oracle-eval → baselines → gallery`).**
- Sweep: 4 arms × 2000 steps. Default: `oracle-lens alpha-sweep` runs the
  arms sequentially on one box (~6 h) — simplest, recommended. If wall-clock
  matters, parallel arms across 4 boxes are possible ONLY via
  `python -c "from oracle_lens.oracle.sft import train_oracle_sft; ..."` with
  an explicit `alpha=` and `out_dir=` per box, collecting the 4 `holdout_ce`
  values by hand. Do NOT try to parallelize with `oracle-sft --max-steps
  2000`: the CLI exposes no alpha override, so all four boxes would train the
  SAME default alpha and clobber the real checkpoint path.
- After the sweep: set `oracle.alpha` in the config (absolute value =
  best-multiplier × √4096 = ×64), commit, rsync.
- `oracle-eval` + `baselines` + `gallery`: cheapest correct play is running
  all three on the 4b training box BEFORE destroying it (80+ GB, models
  already cached). `gallery` holds THREE models resident (subject 8B +
  oracle 8B + truncated reconstructor ≈ 44 GB of weights plus KV cache) —
  it does not fit 24–48 GB boxes. Only `baselines --skip-jlens` (a bare
  store_true flag — `--skip-jlens=false` is an argparse error; the default,
  jlens INCLUDED, is just omitting the flag) is cheap-box friendly.
- Deliverable review (§0.5): pull `eval/table.json` + `eval/gallery.md` to the
  dev box, present to the user with the M4 gate numbers, and stop. M5 (GRPO)
  is a separate decision and a separate stack (`oracle/grpo.py` runbook +
  ENV.md §8; budget ~$70–150 more).

---

## 6. Recovery & re-runs

- **Box dies mid-stage:** artifacts up to the previous stage are on HF. Rent a
  new box, pull, re-run the stage. All samplers are keyed-RNG — same inputs →
  same outputs (bit-identity across boxes is not guaranteed for model
  forwards, ENV.md §5.6; that's fine, only M0's corpus must never be silently
  regenerated — it's the ground truth for every downstream row index).
- **Row alignment is sacred:** `positions.parquet` row i ↔ activation-store
  row i ↔ `store_row` in teacher/eval files. Never regenerate positions or
  re-split after `collect` has run; if you must, delete every downstream
  artifact and restart from `collect`.
- **Config changes** always happen on the dev box + commit + rsync;
  `runs/<name>/config.yaml` (pinned per run) is the provenance record.
- **Every training relaunch** = new wandb run (name is fixed per stage; wandb
  dedups by run id, so relaunches appear as separate runs — leave the dead
  run there, note the ledger).

## 7. Final report

When M4 closes (either gate outcome), assemble `runs/qwen3-8b-v1/REPORT.md`
per PLAN.md §11: gate numbers (M1–M4), the §8.1 table, gallery verdicts with
per-category pass counts, every [choice]-tag deviation actually exercised,
the vast ledger totals (GPU $ + bandwidth $), wandb run links, and HF artifact
index. Negative results at full volume — a boring gallery is a result, not a
failure. Push the report to the artifacts repo and hand the user the link.
