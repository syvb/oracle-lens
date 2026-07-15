"""The deliverable inference API (PLAN.md §1.3):

    decode(activation, N, K) -> [(phrase, coefficient), ...] + FVE

Local HF-generate path with inputs_embeds (adapted from the vendored
nla_inference client's injection mechanics, minus its SGLang transport —
gallery/eval-scale decoding doesn't need a serving stack). Phrases map
through the FROZEN reconstructor and coefficients come from NNLS against the
true activation — the standard decode path [paper].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

from oracle_lens.activations.whitening import WhiteningTransform
from oracle_lens.config import Config
from oracle_lens.nnls import nnls_refit_fve
from oracle_lens.oracle.injection import scale_for_injection
from oracle_lens.oracle.prompts import (
    oracle_token_meta,
    parse_phrases,
    render_oracle_prompt,
)
from oracle_lens.oracle.sft import META_FILE
from oracle_lens.reconstructor.model import encode_phrases, load_reconstructor
from oracle_lens.rendering import load_subject_tokenizer


@dataclass
class DecodeResult:
    phrases: list[str]
    coeffs: list[float]
    fve: float
    raw_text: str

    @property
    def ranked(self) -> list[tuple[str, float]]:
        pairs = sorted(zip(self.phrases, self.coeffs), key=lambda t: -t[1])
        return [(p, c) for p, c in pairs]


class OracleDecoder:
    def __init__(
        self,
        cfg: Config,
        oracle_dir: Path,
        *,
        reconstructor_dir: Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.device = device
        meta = yaml.safe_load((Path(oracle_dir) / META_FILE).read_text())
        self.alpha: float = meta["alpha"]
        self.whitening = WhiteningTransform.load(meta["whitening_path"])
        self.tokenizer = load_subject_tokenizer(str(oracle_dir))
        self.token_meta = oracle_token_meta(self.tokenizer, cfg)
        self.oracle = AutoModelForCausalLM.from_pretrained(
            str(oracle_dir), torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.reconstructor = load_reconstructor(
            reconstructor_dir or meta["reconstructor"]
        ).to(device).eval()

    @torch.no_grad()
    def decode_batch(
        self,
        activations_raw: torch.Tensor,
        *,
        n: int,
        k: int,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> list[DecodeResult]:
        """activations_raw: [B, d] RAW (unwhitened) layer-l* activations, all
        decoded with the same requested (N, K)."""
        bsz = activations_raw.shape[0]
        prompt_ids = render_oracle_prompt(self.tokenizer, self.cfg, self.token_meta, k=k, n=n)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).repeat(bsz, 1).to(self.device)

        targets_w = self.whitening.whiten(activations_raw).to(self.device)
        vectors = scale_for_injection(targets_w, self.alpha)

        from nla.injection import inject_at_marked_positions

        embed = self.oracle.get_input_embeddings()
        embeds = inject_at_marked_positions(
            input_ids,
            embed(input_ids),
            vectors.to(embed.weight.dtype),
            self.token_meta.injection_token_id,
            self.token_meta.injection_left_neighbor_id,
            self.token_meta.injection_right_neighbor_id,
        )
        generated = self.oracle.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones_like(input_ids),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # with inputs_embeds, generate returns only the new tokens

        results: list[DecodeResult] = []
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
        all_phrases = [parse_phrases(t) for t in texts]
        flat = [p for ps in all_phrases for p in ps]
        if flat:
            dirs_flat = encode_phrases(
                self.reconstructor, self.tokenizer, flat,
                self.cfg.reconstructor.prompt_template, device=self.device,
            )
        offset = 0
        k_max = max((len(ps) for ps in all_phrases), default=0)
        atoms = torch.zeros(bsz, max(k_max, 1), targets_w.shape[1])
        for i, ps in enumerate(all_phrases):
            if ps:
                atoms[i, : len(ps)] = dirs_flat[offset : offset + len(ps)]
                offset += len(ps)
        coeffs, fve = nnls_refit_fve(atoms, targets_w.cpu())
        for i, ps in enumerate(all_phrases):
            results.append(
                DecodeResult(
                    phrases=ps,
                    coeffs=coeffs[i, : len(ps)].tolist(),
                    fve=float(fve[i]) if ps else 0.0,
                    raw_text=texts[i],
                )
            )
        return results

    def decode(self, activation_raw: torch.Tensor, *, n: int, k: int, **kw) -> DecodeResult:
        return self.decode_batch(activation_raw.reshape(1, -1), n=n, k=k, **kw)[0]
