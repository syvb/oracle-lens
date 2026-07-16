"""oracle-lens CLI — one subcommand per pipeline stage (PLAN.md §9).

M0: prompts -> generate -> verify -> positions -> split
M1: collect -> whiten -> layer-check
M2: recon-train -> recon-eval
M3: dictionary -> teacher
M4: alpha-sweep -> oracle-sft -> oracle-eval -> baselines -> gallery
M5: see oracle_lens/oracle/grpo.py (Miles/SGLang runbook)

Heavy imports are deferred into each handler so `--help` and CPU-only
commands work on boxes without GPUs or vLLM.
"""

from __future__ import annotations

import argparse

from oracle_lens.config import Config, load_config
from oracle_lens.runs import RunPaths


def _base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/qwen3-8b.yaml")
    parser.add_argument("--runs-root", default="runs")


def _setup(args) -> tuple[Config, RunPaths]:
    cfg = load_config(args.config)
    run = RunPaths.for_run(cfg, args.runs_root)
    run.pin_config(cfg)
    return cfg, run


def main() -> None:
    p = argparse.ArgumentParser(prog="oracle-lens", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, extra in {
        "prompts": [],
        "generate": [
            ("--limit", int, None),
            ("--shard-index", int, None),
            ("--num-shards", int, None),
        ],
        "merge-corpus": [("--num-shards", int, None)],
        "verify": [("--n-samples", int, 200)],
        "positions": [],
        "split": [],
        "collect": [],
        "whiten": [],
        "layer-check": [],
        "recon-train": [("--device-batch-size", int, 8), ("--max-steps", int, None)],
        "recon-eval": [("--n-per-bucket", int, 2000)],
        "dictionary": [],
        "teacher": [],
        "alpha-sweep": [("--steps", int, 2000), ("--device-batch-size", int, 4)],
        "oracle-sft": [("--device-batch-size", int, 4), ("--max-steps", int, None)],
        "oracle-eval": [("--n-samples", int, 2000)],
        "baselines": [("--n-samples", int, 1000), ("--skip-jlens", bool, False)],
        "gallery": [],
    }.items():
        sp = sub.add_parser(name)
        _base(sp)
        for flag, typ, default in extra:
            if typ is bool:
                sp.add_argument(flag, action="store_true")
            else:
                sp.add_argument(flag, type=typ, default=default)

    args = p.parse_args()
    cfg, run = _setup(args)

    if args.cmd == "prompts":
        from oracle_lens.corpus.wildchat import build_prompts

        build_prompts(cfg, run.prompts)
    elif args.cmd == "generate":
        from oracle_lens.corpus.generate import generate_corpus

        generate_corpus(
            cfg,
            run.prompts,
            run.corpus,
            limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.cmd == "merge-corpus":
        from oracle_lens.corpus.merge import merge_corpus_shards

        if args.num_shards is None:
            raise SystemExit("merge-corpus requires --num-shards")
        merge_corpus_shards(
            cfg, run.prompts, run.corpus, num_shards=args.num_shards
        )
    elif args.cmd == "verify":
        from oracle_lens.corpus.verify import verify_token_identity

        report = verify_token_identity(cfg, run.corpus, n_samples=args.n_samples)
        raise SystemExit(0 if report.passed else 1)
    elif args.cmd == "positions":
        from oracle_lens.corpus.positions import build_positions

        build_positions(cfg, run.corpus, run.positions)
    elif args.cmd == "split":
        from oracle_lens.corpus.splits import assign_splits

        assign_splits(cfg, run.positions)
    elif args.cmd == "collect":
        from oracle_lens.activations.collect import collect_activations

        collect_activations(cfg, run.corpus, run.positions, run.activations_dir)
    elif args.cmd == "whiten":
        from oracle_lens.activations.fit import fit_whitening_for_run

        fit_whitening_for_run(cfg, run.positions, run.activations_dir, run.whitening)
    elif args.cmd == "layer-check":
        from oracle_lens.activations.layer_check import run_layer_check

        run_layer_check(cfg, run.layer_check_dir)
    elif args.cmd == "recon-train":
        from oracle_lens.reconstructor.train import train_reconstructor

        train_reconstructor(
            cfg, run, device_batch_size=args.device_batch_size, max_steps=args.max_steps
        )
    elif args.cmd == "recon-eval":
        from oracle_lens.reconstructor.evaluate import evaluate_reconstructor

        evaluate_reconstructor(cfg, run, n_per_bucket=args.n_per_bucket)
    elif args.cmd == "dictionary":
        from oracle_lens.teacher.dictionary import build_dictionary

        build_dictionary(cfg, run)
    elif args.cmd == "teacher":
        from oracle_lens.teacher.decompose import run_teacher

        run_teacher(cfg, run)
    elif args.cmd == "alpha-sweep":
        from oracle_lens.oracle.sft import alpha_sweep

        alpha_sweep(cfg, run, steps=args.steps, device_batch_size=args.device_batch_size)
    elif args.cmd == "oracle-sft":
        from oracle_lens.oracle.sft import train_oracle_sft

        train_oracle_sft(
            cfg, run, device_batch_size=args.device_batch_size, max_steps=args.max_steps
        )
    elif args.cmd == "oracle-eval":
        from oracle_lens.oracle.evaluate import evaluate_oracle

        evaluate_oracle(cfg, run, n_samples=args.n_samples)
    elif args.cmd == "baselines":
        from oracle_lens.eval.baselines import run_baselines

        run_baselines(cfg, run, n_samples=args.n_samples, skip_jlens=args.skip_jlens)
    elif args.cmd == "gallery":
        from oracle_lens.eval.gallery import run_gallery

        run_gallery(cfg, run)


if __name__ == "__main__":
    main()
