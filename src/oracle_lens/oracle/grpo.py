"""Stage 5 (M5, OPTIONAL — gated on M4 qualitative results): GRPO reward.

PLAN.md §7.3. GRPO itself is NOT implemented here: per the reuse decision, RL
runs on the vendored nla repo's Miles + SGLang stack, where GRPO, FSDP2
weight-sync, and input_embeds rollouts already work. The ONLY piece that
project owns is the reward — this module — which replaces nla.reward.nla_rm's
"-MSE via live critic" with "whitened FVE of the NNLS-refit reconstruction,
minus K/N format penalties". The reconstructor stays FROZEN — that is the
property separating an oracle lens from an NLA; do not unfreeze it.

Runbook (when M5 is green-lit):
 1. Install Miles at the SHA pinned in
    third_party/natural_language_autoencoders/nla/miles_patches/UPSTREAM_PIN
    and apply the two patches in that directory; install SGLang >= 0.5.6 and
    apply patches/ per docs/setup.md (all inside the vendored tree).
 2. Convert the SFT checkpoint with the vendored tools/ converters; write an
    nla_meta.yaml sidecar whose actor_prompt_template is
    oracle_lens.oracle.prompts.canonical_actor_template(cfg) and whose
    injection_scale is the swept alpha.
 3. Adapt configs/rl.sh from the vendored repo: --custom-rm-path pointing at
    a thin wrapper that loads the frozen reconstructor once and calls
    grpo_reward below; group size / batch / KL / steps from cfg.grpo.
 4. Monitor: held-out FVE every cfg.grpo.eval_every steps, and phrase entropy
    over a fixed probe batch (collapse onto generic high-coverage phrases is
    THE failure mode — if entropy drops while FVE stalls, raise KL or stop
    and ship the SFT checkpoint).
 5. Stop rule [paper]: halt when held-out FVE improves by less than
    cfg.grpo.stop_fve_gain over the last cfg.grpo.stop_window steps
    (default: <0.5% per 200 steps), or at cfg.grpo.max_steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from oracle_lens.config import Config
from oracle_lens.nnls import nnls_refit_fve
from oracle_lens.oracle.prompts import parse_phrases


@dataclass
class RewardBreakdown:
    reward: float
    fve: float
    k_penalty: float
    n_penalty: float
    n_phrases: int


def grpo_reward(
    generated_text: str,
    activation_w: torch.Tensor,
    *,
    requested_k: int,
    requested_n: int,
    encode,  # callable: list[str] -> [K, d] unit whitened dirs (FROZEN reconstructor)
    count_tokens,  # callable: str -> int (subject tokenizer)
    cfg: Config,
) -> RewardBreakdown:
    """Reward [paper]: whitened FVE of the NNLS-refit reconstruction of the
    generated phrase list, minus small penalties for deviating from the
    requested K and N. Malformed output (no parseable phrases) gets FVE 0
    plus the full-K penalty, mirroring nla's failed-extraction convention."""
    phrases = parse_phrases(generated_text)
    gcfg = cfg.grpo
    k_pen = gcfg.k_penalty * abs(len(phrases) - requested_k)
    if not phrases:
        return RewardBreakdown(
            reward=-k_pen, fve=0.0, k_penalty=k_pen, n_penalty=0.0, n_phrases=0
        )
    n_pen = gcfg.n_penalty * sum(
        abs(count_tokens(p) - requested_n) for p in phrases
    ) / len(phrases)
    dirs = encode(phrases)  # [K, d]
    _, fve = nnls_refit_fve(dirs.unsqueeze(0), activation_w.reshape(1, -1))
    fve_val = float(fve[0])
    return RewardBreakdown(
        reward=fve_val - k_pen - n_pen,
        fve=fve_val,
        k_penalty=k_pen,
        n_penalty=n_pen,
        n_phrases=len(phrases),
    )


def phrase_entropy(all_phrase_lists: list[list[str]]) -> float:
    """Collapse monitor: empirical entropy (nats) of the phrase distribution
    over a fixed probe batch. Track each eval; falling entropy + stalled FVE
    means raise KL or stop (PLAN.md §7.3)."""
    import math
    from collections import Counter

    counts = Counter(p for ps in all_phrase_lists for p in ps)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())
