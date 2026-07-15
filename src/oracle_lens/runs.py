"""Run-directory layout — stages find upstream artifacts here by convention.

runs/<run_name>/
    config.yaml               pinned copy of the config this run used
    corpus/prompts.parquet    stage 0: filtered/deduped WildChat first turns
    corpus/corpus.parquet     stage 0: on-policy transcripts (token IDs)
    corpus/positions.parquet  stage 0: sampled positions + split labels
    activations/              stage 1: fp16 memmap + index parquet
    whitening.pt              stage 1: WhiteningTransform
    layer_check/              stage 1: jlens slice pages for the l* sanity check
    reconstructor/            stage 2: checkpoint + eval report
    dictionary/               stage 3: phrase parquet + direction memmaps per N
    teacher/teacher.parquet   stage 3: NN-OMP decompositions + gate report
    oracle_sft/               stage 4: checkpoint + eval + gallery
    oracle_rl/                stage 5 (optional)
    eval/                     §8.1 table + §8.2 gallery output
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oracle_lens.config import Config, dump_config


@dataclass
class RunPaths:
    root: Path

    @classmethod
    def for_run(cls, cfg: Config, runs_root: str | Path = "runs") -> "RunPaths":
        return cls(root=Path(runs_root) / cfg.run_name)

    # stage 0
    @property
    def prompts(self) -> Path:
        return self.root / "corpus" / "prompts.parquet"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus" / "corpus.parquet"

    @property
    def positions(self) -> Path:
        return self.root / "corpus" / "positions.parquet"

    # stage 1
    @property
    def activations_dir(self) -> Path:
        return self.root / "activations"

    @property
    def whitening(self) -> Path:
        return self.root / "whitening.pt"

    @property
    def layer_check_dir(self) -> Path:
        return self.root / "layer_check"

    # stage 2
    @property
    def reconstructor_dir(self) -> Path:
        return self.root / "reconstructor"

    # stage 3
    @property
    def dictionary_dir(self) -> Path:
        return self.root / "dictionary"

    @property
    def teacher(self) -> Path:
        return self.root / "teacher" / "teacher.parquet"

    # stage 4/5
    @property
    def oracle_sft_dir(self) -> Path:
        return self.root / "oracle_sft"

    @property
    def oracle_rl_dir(self) -> Path:
        return self.root / "oracle_rl"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    def pin_config(self, cfg: Config) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.yaml").write_text(dump_config(cfg))

    def require(self, *paths: Path) -> None:
        """Gate helper: fail loudly when an upstream stage hasn't run."""
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "missing upstream artifacts (run the earlier stage first): "
                + ", ".join(missing)
            )
