"""Stage 4 activation injection (PLAN.md §7.1).

The pure splice is the vendored nla.injection.inject_at_marked_positions
(neighbor-verified, crash-loud). This module adds the two thin pieces around
it: alpha scaling of the WHITENED [choice] activation, and the forward hook
on the embedding layer for training/local decode (nla's train_actor pattern,
minus the Miles plumbing).
"""

from __future__ import annotations

import math

import torch
from nla.injection import inject_at_marked_positions
from nla.schema import NLATokenMeta, normalize_activation

from oracle_lens.config import Config


def resolve_alpha(cfg: Config) -> float:
    """Injection scale: cfg.oracle.alpha, defaulting to sqrt(d_model) — the
    ambient residual-stream scale (nla's default). The mini-sweep multiplies
    this by cfg.oracle.alpha_sweep values."""
    return cfg.oracle.alpha or math.sqrt(cfg.model.d_model)


def scale_for_injection(vectors_whitened: torch.Tensor, alpha: float) -> torch.Tensor:
    """L2-rescale whitened activations to norm alpha (mandatory: the model is
    out-of-distribution otherwise — nla docs)."""
    return normalize_activation(vectors_whitened, alpha)


class InjectionState:
    """Holder the training loop fills right before each forward pass; the hook
    reads it. Scanning happens INSIDE the hook against live input_ids — never
    precompute positions (nla invariant: batches get reordered)."""

    def __init__(self) -> None:
        self.vectors: torch.Tensor | None = None  # [n_sites, d], already scaled


def register_injection_hook(
    model: torch.nn.Module, meta: NLATokenMeta, state: InjectionState
):
    def hook(_module, inputs, output):
        if state.vectors is None:
            return output
        return inject_at_marked_positions(
            inputs[0],
            output,
            state.vectors,
            meta.injection_token_id,
            meta.injection_left_neighbor_id,
            meta.injection_right_neighbor_id,
        )

    return model.get_input_embeddings().register_forward_hook(hook)
