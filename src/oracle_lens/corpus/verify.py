"""Stage 0c: token-identity check — the M0 gate (PLAN.md §3).

Two properties, checked on a deterministic sample of transcripts:

1. PROMPT RENDER DETERMINISM (hard gate): re-rendering the stored prompt text
   through rendering.render_first_turn reproduces the stored prompt_ids
   exactly. If this fails, the template/tokenizer drifted since generation and
   every downstream position index is suspect.

2. RESPONSE DECODE ROUND-TRIP (reported, soft): decode(response_ids) then
   re-encode equals response_ids. Token IDs are saved directly at generation,
   so collection never depends on this — but the reconstructor consumes
   phrases as DECODED TEXT, so a low round-trip failure rate tells us decoded
   phrases faithfully represent the underlying tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from oracle_lens.config import Config
from oracle_lens.rendering import load_subject_tokenizer, render_first_turn


@dataclass
class VerifyReport:
    n_checked: int
    prompt_mismatches: int
    response_roundtrip_failures: int

    @property
    def passed(self) -> bool:
        return self.n_checked > 0 and self.prompt_mismatches == 0


def verify_token_identity(cfg: Config, corpus_path: Path, *, n_samples: int = 200) -> VerifyReport:
    table = pq.read_table(
        corpus_path, columns=["conversation_id", "prompt", "prompt_ids", "response_ids"]
    )
    n_rows = table.num_rows
    # Deterministic sample keyed on conversation_id (nla per-doc RNG pattern).
    ranked = sorted(
        range(n_rows),
        key=lambda i: hashlib.sha256(
            f"verify|{table['conversation_id'][i].as_py()}".encode()
        ).hexdigest(),
    )[: min(n_samples, n_rows)]

    tokenizer = load_subject_tokenizer(cfg.model.name)
    prompt_bad = 0
    roundtrip_bad = 0
    for i in ranked:
        stored = table["prompt_ids"][i].as_py()
        rerendered = render_first_turn(tokenizer, table["prompt"][i].as_py(), cfg)
        if rerendered != stored:
            prompt_bad += 1
        resp = table["response_ids"][i].as_py()
        text = tokenizer.decode(resp)
        if tokenizer(text, add_special_tokens=False)["input_ids"] != resp:
            roundtrip_bad += 1

    report = VerifyReport(
        n_checked=len(ranked),
        prompt_mismatches=prompt_bad,
        response_roundtrip_failures=roundtrip_bad,
    )
    status = "PASS" if report.passed else "FAIL"
    print(
        f"token-identity [{status}]: {report.n_checked} checked, "
        f"{prompt_bad} prompt mismatches (hard gate), "
        f"{roundtrip_bad} response decode round-trip failures (informational)"
    )
    return report
