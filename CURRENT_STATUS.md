# Oracle Lens execution handoff

Updated: 2026-07-17T02:10:00Z

Current operational state for an agent resuming the experiment in
`/home/debian/oracle-lens`. Read `PLAN.md`, `README.md`, `EXECUTION.md`, and
`~/ENV.md` before taking action.

## Stop point and user instructions

- M0-M3 are complete; every gate has passed. **M4 (alpha-sweep ->
  oracle-sft -> oracle-eval -> baselines -> gallery) is next; nothing blocks
  it** except the M4 plan's own human checkpoint at the end (gallery review).

## M3 result (gate PASS on the paper protocol)

Two teacher runs on H100 contract 45130567 (destroyed; ~$1.62 total):
- Run 1 (min_gain=5e-3 [choice]): fve_mean_overall 0.1294 -> gate FAIL.
  Diagnosis: the stop sat exactly at the ~250k-candidate selection-noise
  floor (~4.9e-3/atom), truncating teachers to mean 4.5/16 atoms (0% reached
  16). Sample sensitivity (N=8): 0.135 -> 0.153 (min_gain 2e-3) -> 0.162
  (16 atoms, no stop); random-Gaussian-dictionary floor at 16 atoms: 0.057
  (real dictionary carries ~2.9x the floor). Preserved at
  diagnostics/m3_min_gain_5e-3/ + diagnostics/m3/stop_rule_diag.py.
- Run 2 (paper protocol: fixed 16-atom budget, min_gain=0, commit 9e58ca5):
  **fve_mean_overall 0.1563 >= 0.15 PASS**; delimiter 0.1822, ordinary
  0.1455; by_n 0.140/0.158/0.162/0.162/0.160 for N=2/4/8/16/32.
- Dictionary: 795k phrases total (N2 96,852 / N4 166,125 / N8 181,457 /
  N16 179,271 / N32 171,187), encoded with the v2 reconstructor.
- Public: qwen3-8b-v1/dictionary/ (6.5 GB), qwen3-8b-v1/teacher/
  (teacher.parquet 272 MB + gate.json), diagnostics/m3*. All verified
  anonymously by name+size. Local copy of gate.json at
  runs/qwen3-8b-v1/teacher/.
- Layer 24 was treated as confirmed: the user reviewed the handoff that named
  it as the candidate and instructed "continue to the end of M2"; the four
  slice pages were also delivered to the user in-chat. If the user ever
  selects a different layer, redo `collect`/`whiten`/M2.
- User overrides that remain binding:
  - Every Hugging Face repository and uploaded artifact must be public.
  - Preserve useful intermediate results.
  - M0 generation uses `max_new_tokens: 4096`; do not reduce it.
  - Thinking disabled via Qwen3 template; leak rejection/resampling stands.
  - Long-running jobs and uploads must have a real background watcher;
    re-arm expired watchers.

## M2 result (v2 — gate PASS)

M2 was run twice. The v1 run (2026-07-16, contract 45116458) hit a real
training bug found by a four-agent review: the model was loaded in bf16 and
AdamW ran on bf16 master weights, so at lr 1e-5 most backbone updates fell
below the bf16 ulp and rounded to zero — only the value head trained.
Fixed in commit 9953d0e (fp32 master weights + bf16 autocast, model.train()
before the loop, injection dtype from embeddings, checkpoints cast back to
bf16 on save; EXECUTION.md GPU floors updated: [train-6B] needs H200-class,
[train-8B] on one H200 is tight). The v2 rerun (contract 45122991) used the
identical config/seed/pairs.

v2 results (`reconstructor/eval.json`, public; wandb qwen3-8b-v1-recon-train):
- eval_cosine 0.3059 (v1: 0.2311). Held-out cosine by N:
  0.137/0.206/0.266/0.307/0.328/0.340 for N=1/2/4/8/16/32.
- Continuation-FVE by N: 0.030/0.058/0.087/0.109/0.121/0.127
  (delimiter up to 0.153 at N=32); ~1.6-1.8x v1 in every bucket.
- Gate: control_separation 0.1142 >= 0.10 PASS;
  small_n_continuation_fve_min 0.0304 >= 0.02 PASS. Same thresholds as v1
  (deliberately unchanged).
- Loss 2.00 -> ~1.31; still mildly improving at end of schedule (schedule
  leaves a little on the table; acceptable, gate passed).

Review-panel notes to carry into M3/M4 and the final report:
- PLAN.md §5 "expect monotone degradation with N" is an erratum — improvement
  with N is the theoretically correct behavior (target is one fixed vector;
  longer phrases only add conditioning information).
- M3: report teacher FVE alongside a matched random-dictionary NN-OMP control
  (same K/N mix/mask/stop protocol) to separate real signal from the
  ~250k-candidate selection floor; verify the min-gain stop rule is measured
  in absolute FVE, not fraction-of-residual.
- The M2 gate's controls anchor to the N=1 bucket (hardest, and absent from
  the M3 dictionary); operationalization kept as-is since v2 passes anyway.
- v1 artifacts preserved: eval/metrics at diagnostics/m2_v1_bf16_bug/, v1 run
  logs at diagnostics/m2/, and the v1 checkpoint at model-repo revision
  bb2e60b808. diagnostics/m2_v2/ holds the v2 logs. Current model-repo HEAD
  (c8a0c0970b) is v2; LFS sha256 d02146f32b79fe41... verified against the box.

## Infrastructure state

- All oracle-lens Vast instances are destroyed; the ledger is current
  (M2 v1: 45116230 rejected ~$0.20 + 45116458 ~$5.59; M2 v2: 45122991
  ~1.6h @ $4.08 ~= $6.55). Cumulative oracle-lens spend ~= $60.
  NOTE: an unrelated instance 45125843 "suffix-eval-h200" exists on the
  account — NOT ours, NOT in the ledger, do not touch it.
- No uploader processes are running locally. Local disk: ~6.5 GB free —
  do not pull large artifacts here.
- Dev-box HF uploads: `hf_xet` ballooned RAM and got OOM-killed twice;
  export `HF_HUB_DISABLE_XET=1` for any dev-box upload (upload helpers work
  fine with it; ~60 MB/s sustained).

## Public Hugging Face state (all verified anonymously by name+size)

- `syvb/oracle-lens-qwen3-8b-artifacts` (dataset):
  - M0 corpus/positions/prompts + shards + diagnostics — complete.
  - M1 complete: `activations/` (10,482,196,480-byte store), `whitening.pt`,
    `layer_check/` (lens.pt, lens_ckpt.pt, checkpoints 050-200, 4 slice
    pages), `diagnostics/m1/` incl. `M1_SHA256SUMS`.
  - M2: `reconstructor/eval.json`, `reconstructor/metrics.jsonl`,
    `diagnostics/m2/` (setup/pull/smoke/full/eval logs, env.txt,
    nvidia-smi, code commit).
- `syvb/oracle-lens-qwen3-8b-recon` (model): the trained reconstructor —
  `model.safetensors` (10,892,013,280 bytes), `value_head.safetensors`,
  tokenizer + configs + `oracle_lens_recon.yaml`.
- `syvb/oracle-lens-qwen3-8b-oracle-sft` (model): still empty (M4).

The local 14 GB M1 bundle at `runs/qwen3-8b-v1/` is now redundant with HF
(all hashes/sizes verified both sides) but is retained for now. Local copies
of M2 `eval.json`/`metrics.jsonl` live at `runs/qwen3-8b-v1/reconstructor/`.

## Resumption sequence (M3 next)

1. Rent an [infer] 48-80 GB box (EXECUTION.md §2; dictionary encode +
   NN-OMP are inference/matmul only — no fp32-optimizer constraint), pull
   `corpus/`, `activations/`, `whitening.pt` from the artifact dataset and
   the reconstructor from `syvb/oracle-lens-qwen3-8b-recon` into
   `/workspace/runs/qwen3-8b-v1/reconstructor/checkpoint/`, then run
   `dictionary` and `teacher`. Gate: `teacher/gate.json`
   `fve_mean_overall >= 0.15`.
2. ops helpers: `ops/setup_m1.sh` (generic env build), `ops/pull_m2.py`
   (pattern for pulls), `ops/upload_*.py`. Dev-box uploads need
   HF_HUB_DISABLE_XET=1.
3. Boxes: speed-test egress first (>=150 MB/s; retest once on a cold link),
   ledger immediately, watcher on every detached job, verify uploads
   anonymously (name+size, LFS sha256 for same-size overwrites) before
   destroy.
