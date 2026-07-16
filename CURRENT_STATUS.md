# Oracle Lens execution handoff

Updated: 2026-07-16T20:05:16Z

This is the current operational state for an agent resuming the experiment in
`/home/debian/oracle-lens`. Read `PLAN.md`, `README.md`, `EXECUTION.md`, and
`~/ENV.md` before taking action. The repository commit immediately before this
handoff is `d0fd411` (`Record M1 H100 teardown`).

## Stop point and user instructions

- M0 and M1 are complete. M2 has **not** started.
- The immediate blocker is the mandatory human review of the four M1 slice
  pages. The current candidate is `layer_index: 24`, which already matches
  `configs/qwen3-8b.yaml`.
- Do not begin M2 until the user confirms layer 24 after reviewing the pages.
- The user asked to stop for now after artifacts were made safe locally because
  Hugging Face was having an outage.
- User overrides that remain binding:
  - Every Hugging Face repository and uploaded artifact must be public.
  - Preserve useful intermediate results.
  - M0 generation uses `max_new_tokens: 4096`; do not reduce it.
  - Thinking is disabled consistently with Qwen3's chat template. The template
    includes its normal empty think block when thinking is disabled, while
    model-generated reasoning-tag leakage is rejected/resampled.
  - Long-running jobs and uploads must have a real background watcher. Re-arm
    it if its iteration cap expires, and do not leave a remote job unwatched.

## Infrastructure state

- Vast.ai has zero active instances.
- The final M1 H100 was contract `45049353`, label
  `oracle-lens-m1-0716-0341`, and was destroyed at
  `2026-07-16T08:32:10Z` after the local backup passed its hashes.
- `ops/vast_ledger.tsv` is current. Only operate on new contracts created and
  recorded for this task.
- There are no active Hugging Face uploader processes.
- Local filesystem space is tight: the last check showed 6.4 GB free on `/`.
  Do not duplicate the 14 GB M1 bundle or download the M0 corpus here without
  first arranging more space. Hard-link staging is preferable if needed.

## Public Hugging Face repositories

- Dataset/artifacts: `syvb/oracle-lens-qwen3-8b-artifacts`
- Reconstructor model: `syvb/oracle-lens-qwen3-8b-recon`
- Oracle SFT model: `syvb/oracle-lens-qwen3-8b-oracle-sft`

All three were created public. M0 is public in the artifact dataset. During the
M1 upload, Hugging Face began returning repeated HTTP 504 responses from both
`/api/repos/create` and `/preupload/main`. Public M1 checkpoints 50, 100, and
150 were uploaded and anonymously size-verified before the outage. The full
M1 upload and checkpoint 200 were not completed on Hugging Face.

Uploader helpers are in:

- `ops/upload_artifact.py` — upload one file to the public artifact dataset.
- `ops/upload_large_folder.py` — resumable large-folder upload, patched to
  bypass the helper's redundant repository-create request.
- `ops/upload_run.py` — upload a completed run subtree at
  `qwen3-8b-v1/`.

Do not treat a successful API return as sufficient. Verify uploaded names and
exact sizes anonymously before relying on Hugging Face as the only copy.

## M0 complete

- Canonical corpus: 99,585 transcripts.
- Total response tokens: 89,452,076.
- Mean response length: 898.25 tokens.
- Fraction reaching the 4096 cap: 1.796%.
- Sampled positions: 1,279,565.
- `enable_thinking: false` is applied through the Qwen3 template.
- Thirteen rare model-generated closing-think-tag leaks were found. All were
  deterministically rejection-resampled (9 on retry 1, 4 on retry 2), and the
  originals plus replacements were preserved in public diagnostics.
- The final canonical corpus contains zero model-generated opening or closing
  think tags and passed its identity gate.
- Generation was parallelized over four H200 GPUs; all four shard outputs were
  retained publicly.

Never silently regenerate M0. Its row identity anchors every later artifact.

## M1 complete and locally verified

The complete nonduplicated M1 bundle is locally stored at:

`/home/debian/oracle-lens/runs/qwen3-8b-v1/`

It occupies about 14 GB. The remote `lens_ckpt_200.pt` was omitted only because
it was a byte-identical duplicate of the canonical local
`layer_check/lens_ckpt.pt`; the final checkpoint itself is present locally.

Local backup verification already run successfully:

```bash
cd /home/debian/oracle-lens/runs/qwen3-8b-v1
sha256sum -c diagnostics/m1/M1_SHA256SUMS
```

All 19 entries passed. The manifest is
`runs/qwen3-8b-v1/diagnostics/m1/M1_SHA256SUMS`.

Key files and exact sizes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `activations/vectors.f16.bin` | 10,482,196,480 | `1d3122d392178022758bfa8f4079d7f1e6da6332d28d4a9a063934b471d8bdf4` |
| `whitening.pt` | 134,236,339 | `b7446d2eb87694494b48a5b95af422ce8c76f18e8ee4e37dad176f321566e7be` |
| `layer_check/lens.pt` | 1,174,412,991 | `7e7b3752159f29383595ec96d5236b2693b90a8cf3a3e464f5e02ea4fe166c01` |
| `layer_check/lens_ckpt.pt` | 2,348,820,907 | `fae4825602c356444babcbac0042a562684fa14fce8d4a4e86e67709a32786e2` |
| `layer_check/slice_0.html` | 4,880,597 | `c60437f855f3b54a8c8b4e240689a409134ef9fd0445cf54227bb686c4600788` |
| `layer_check/slice_1.html` | 21,246,859 | `b201377b5cbe265d46383d238601c9f8a36130a57bc15eb87d15c29a452057b7` |
| `layer_check/slice_2.html` | 27,285,921 | `92f73df825949e7b23ebc258b0848ec797e7b95d295ad4d5efdb8a6d43e8153f` |
| `layer_check/slice_3.html` | 29,808,277 | `f4a0ede1713a47755c772fa94d6d78e71ffb03a1407b978291be6b2f80ac736d` |

Activation audit:

- Shape: 1,279,565 rows by 4,096 dimensions, bfloat16.
- A deterministic 2,000-row audit found all values finite.
- Zero zero-norm/unwritten rows.

Whitening gate passed:

- Fit rows: 702,815.
- `var_mean = 0.9442159533500671` (required range: 0.8 to 1.2).
- `var_min = 0.8533905148506165`.
- `var_max = 1.0620629787445068`.
- `mean_abs_mean = 0.0051262956112623215`.
- Condition number: `62865.721849916015`, finite.

Layer check fitted 200 prompts and produced the four review pages. Copies for
the user's browser are at:

- `/home/debian/oracle-lens/reviews/qwen3-8b-v1-layer-check/slice_0.html`
- `/home/debian/oracle-lens/reviews/qwen3-8b-v1-layer-check/slice_1.html`
- `/home/debian/oracle-lens/reviews/qwen3-8b-v1-layer-check/slice_2.html`
- `/home/debian/oracle-lens/reviews/qwen3-8b-v1-layer-check/slice_3.html`

Both `runs/` and `reviews/` are intentionally gitignored.

## Resumption sequence

1. Confirm the Hugging Face service has recovered.
2. Upload the missing M1 files from the verified local bundle to public paths
   under `qwen3-8b-v1/`. The canonical `layer_check/lens_ckpt.pt` is the final
   checkpoint; it may also be uploaded as
   `qwen3-8b-v1/layer_check/checkpoints/lens_ckpt_200.pt` if retaining the
   numbered intermediate convention is desired.
3. Verify every uploaded file anonymously by name and exact size. Preserve the
   local bundle until this has passed.
4. Give the user the four local review pages and wait for explicit confirmation
   of layer 24. If the user selects a different layer, update and commit the
   config, then redo `collect` and `whiten`; the existing activations are
   layer-specific and cannot be reused.
5. Only after layer confirmation, execute M2 (`recon-train` then
   `recon-eval`) per `EXECUTION.md`:
   - Rent a `[train-6B]` instance (prefer one H200; otherwise 2x80 GB with
     FSDP), speed-test it, and immediately log the contract.
   - Install the pinned environment and pull the public M0/M1 artifacts.
   - Export `WANDB_API_KEY` from `/root/.wandb_key`; a missing-key warning is a
     launch failure.
   - Run a 20-step smoke train before the full run.
   - Attach a durable background watcher to every detached setup, transfer,
     training, evaluation, and upload job. A watcher timing out means re-arm
     it, not that the underlying job failed.
   - Run full `recon-train`, then `recon-eval`, and apply the M2 gate from
     `reconstructor/eval.json`.
   - Upload the reconstructor to the public model repository and all M2
     diagnostics/evaluation artifacts to the public dataset repository.
   - Verify public names and sizes before destroying the instance, then record
     its destruction timestamp in the Vast ledger.
6. If the M2 gate fails after the plan's prescribed first remedy, stop and ask
   the user. Do not proceed to M3.

If `eval_cosine` is still clearly improving when M2 training ends, the remedy
requires genuinely more sampled positions, not more epochs over the same
pairs. This regenerates positions, activations, and whitening and therefore
requires user approval before replacing the aligned artifact generation.

## Relevant repository changes

- `1a402d0` — reject rare thinking-tag corpus leaks.
- `a4a04ee` — reproducible Hugging Face run uploader.
- `e1cf421` — record parallel H200 M0 completion.
- `7939be7` — log M1 H100 instance.
- `d000934`, `0213159` — reproducible CUDA 12.6 M1 environment/pull.
- `e7ed11d` — restore the pinned jlens visualization template.
- `41fde31`, `cfbbdd6`, `a20519f`, `ca1fed0` — resilient public artifact
  upload helpers and Hugging Face outage workarounds.
- `d0fd411` — record verified M1 H100 teardown.

At the time this handoff was written, the worktree was clean before adding
this file.
