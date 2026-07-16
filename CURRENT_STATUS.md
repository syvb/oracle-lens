# Oracle Lens execution handoff

Updated: 2026-07-16T21:55:00Z

Current operational state for an agent resuming the experiment in
`/home/debian/oracle-lens`. Read `PLAN.md`, `README.md`, `EXECUTION.md`, and
`~/ENV.md` before taking action.

## Stop point and user instructions

- M0, M1, and M2 are complete. **M3 has not started and is blocked on a user
  decision about the M2 gate** (see below).
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

## M2 result and the pending gate decision

Training and eval completed 2026-07-16 on an H200 (contract 45116458,
destroyed after verification). wandb: project `oracle-lens`, entity
`octahedral-systems`, run `qwen3-8b-v1-recon-train` (x86r3wad = 20-step
smoke; the full run is the later one). 378,623 pairs, 5,915 optim steps,
effective batch 128, 2 epochs, loss 2.00 -> 1.507, eval_cosine plateaued at
0.2311 (flat for final ~800 steps -> the "add data" remedy does NOT apply).

`reconstructor/eval.json` (public, also local):

- Held-out cosine by N: 0.115 (N=1), 0.163 (2), 0.205 (4), 0.232 (8),
  0.247 (16), 0.254 (32). Monotone IMPROVEMENT with N (PLAN.md expected
  degradation; the failure mode it warns about — good only at N=1-2 — is
  absent).
- Continuation-FVE by N: 0.019 (N=1), 0.035 (2), 0.052 (4), 0.063 (8),
  0.069 (16), 0.072 (32); delimiter > non-delimiter in every bucket.
- Controls: true pairing 0.1145 vs shuffled 0.0006 (~200x) vs
  predict-mean 0.0228 (~5x).
- Numeric gate in config ([choice] thresholds): control_separation 0.0917
  vs required 0.10 -> FAIL; min continuation-FVE at N<=8 is 0.0194 (N=1)
  vs required 0.02 -> FAIL. Both misses are marginal and N=1-driven;
  N=2/4/8 pass comfortably.
- PLAN.md §5's qualitative gate (clear control separation; plateaued;
  continuation-FVE meaningfully above zero at N<=8) is arguably satisfied.

The user must decide: (a) accept as pass and proceed to M3, (b) accept but
first relax/re-derive the [choice] thresholds in config and commit, or
(c) regenerate more positions (bump positions.per_response, redo
positions/split/collect/whiten + retrain — invalidates the aligned store and
all downstream artifacts; needs the row-alignment teardown in EXECUTION.md §5).

## Infrastructure state

- Vast.ai has zero active instances. `ops/vast_ledger.tsv` is current
  (M2: 45116230 rejected for slow egress ~$0.20; 45116458 did the work,
  ~1.37h @ $4.08 ~= $5.59). Cumulative ledger spend ~= $56.
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

## Resumption sequence (after the user's M2 gate decision)

1. If proceeding to M3: rent an [infer] 48-80 GB box (EXECUTION.md §2),
   pull `corpus/`, `activations/`, `whitening.pt` from the artifact dataset
   and the reconstructor from `syvb/oracle-lens-qwen3-8b-recon` into
   `/workspace/runs/qwen3-8b-v1/reconstructor/checkpoint/`, then run
   `dictionary` and `teacher`. Gate: `teacher/gate.json`
   `fve_mean_overall >= 0.15`.
2. ops helpers: `ops/setup_m1.sh` (generic env build, works for any stage),
   `ops/pull_m2.py` (pattern for artifact pulls), `ops/upload_*.py`.
3. Boxes: speed-test egress first (>=150 MB/s), ledger immediately, watcher
   on every detached job, verify uploads anonymously before destroy.
