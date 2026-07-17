"""Typed configuration for the whole pipeline (PLAN.md §2).

One YAML file (configs/qwen3-8b.yaml) holds every constant and [choice] knob;
every stage receives this same object. `RunPaths.pin_config` copies the YAML
into the run directory so each run records exactly what it ran with.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    name: str = "Qwen/Qwen3-8B"
    layer_index: int = 24  # l* = two-thirds depth [choice]; M1 layer check may revise
    d_model: int = 4096  # verified against config.json at collection time
    enable_thinking: bool = False  # v1: thinking disabled everywhere [choice]


@dataclass
class GenerationCfg:
    corpus: str = "allenai/WildChat-1M"
    n_conversations: int = 100_000
    english_only: bool = True  # [choice]
    temperature: float = 0.7  # Qwen3 non-thinking recommended; verify vs model card
    top_p: float = 0.8
    max_new_tokens: int = 600
    max_prompt_tokens: int = 2048
    seed: int = 20260715


@dataclass
class PositionsCfg:
    per_response: int = 10  # uniform over assistant span
    delimiter_extra: int = 3  # always also include a few delimiter positions [paper]
    seed: int = 1


@dataclass
class SplitsCfg:
    """Target position counts; realized by conversation-level hashing, so
    actual counts are approximate and splits are disjoint by conversation."""

    reconstructor: int = 300_000
    dictionary: int = 150_000
    teacher: int = 250_000
    rl: int = 250_000
    eval: int = 50_000
    seed: int = 2


@dataclass
class WhiteningCfg:
    ridge_frac: float = 1e-2  # [choice]; risk §10.4 says sweep this first


@dataclass
class ReconstructorCfg:
    # Template must contain {phrase} and end with a fixed >=2-token suffix:
    # extraction is suffix-anchored at tokens[-1] (nla convention).
    prompt_template: str = "<text>{phrase}</text> <summary>"
    n_min: int = 1  # phrase length range [paper]
    n_max: int = 32
    lr: float = 1e-5
    head_lr_mult: float = 10.0
    batch_size: int = 128
    epochs: int = 2
    warmup_steps: int = 100
    seed: int = 3
    # M2 gate thresholds [choice] — PLAN.md §5 declines absolute numbers
    gate_min_separation: float = 0.1
    gate_min_continuation_fve: float = 0.02


@dataclass
class DictionaryCfg:
    lengths: tuple[int, ...] = (2, 4, 8, 16, 32)  # [paper]
    encode_batch_size: int = 256


@dataclass
class TeacherCfg:
    max_atoms: int = 16  # [paper]
    # [paper] fixed 16-atom budget, no early stop. A 5e-3 marginal-FVE stop
    # [choice] was tried first but sits exactly at the ~250k-candidate
    # selection-noise floor (~4.9e-3/atom), truncating teachers to ~4.5 atoms
    # and the M3 gate to 0.129; see diagnostics/m3_min_gain_5e-3/.
    min_gain: float = 0.0
    batch_size: int = 4096
    seed: int = 4  # keys the per-example half-dictionary masks
    gate_min_fve: float = 0.15  # M3 gate [choice]


@dataclass
class OracleCfg:
    # {injection_char} slot is where the activation pseudo-token goes.
    prompt_template: str = (
        "Activation: <concept>{injection_char}</concept>\n"
        "List exactly {k} phrases of exactly {n} tokens each that describe "
        "what the model producing this activation is about to say or is "
        "considering. One phrase per line."
    )
    k_min: int = 4  # K sampled uniformly per example during training [choice]
    k_max: int = 16
    # Injection scale alpha: values to mini-sweep (multipliers on sqrt(d_model),
    # the ambient residual-stream scale nla defaults to). [choice]
    alpha_sweep: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0)
    alpha: float | None = None  # set after the sweep; None = sqrt(d_model)
    lr: float = 1e-5
    batch_size: int = 128
    epochs: int = 1
    warmup_steps: int = 100
    seed: int = 5
    gate_min_teacher_frac: float = 0.7  # M4 gate: oracle FVE >= 70% of teacher
    gate_min_format_valid: float = 0.95


@dataclass
class GrpoCfg:
    """M5 (optional, gated). Executed via Miles + SGLang per the nla repo;
    see oracle_lens/oracle/grpo.py for the reward and the runbook."""

    group_size: int = 8
    batch_size: int = 256
    temperature: float = 1.0
    kl_coef: float = 1e-3
    max_steps: int = 1000
    eval_every: int = 100
    # Stop rule [paper]: halt when held-out FVE gains fall below
    # stop_fve_gain over a stop_window of steps (plan: <0.5% per 200 steps).
    stop_fve_gain: float = 5e-3
    stop_window: int = 200
    n_penalty: float = 0.01  # per-token deviation from requested N
    k_penalty: float = 0.05  # per-phrase deviation from requested K


@dataclass
class Config:
    run_name: str = "qwen3-8b-v1"
    model: ModelCfg = field(default_factory=ModelCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    positions: PositionsCfg = field(default_factory=PositionsCfg)
    splits: SplitsCfg = field(default_factory=SplitsCfg)
    whitening: WhiteningCfg = field(default_factory=WhiteningCfg)
    reconstructor: ReconstructorCfg = field(default_factory=ReconstructorCfg)
    dictionary: DictionaryCfg = field(default_factory=DictionaryCfg)
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    oracle: OracleCfg = field(default_factory=OracleCfg)
    grpo: GrpoCfg = field(default_factory=GrpoCfg)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(valid)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    for name, value in data.items():
        f = valid[name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return _from_dict(Config, data)


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(asdict(cfg), sort_keys=False)
