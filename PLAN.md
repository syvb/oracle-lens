# Experiment Plan: Oracle Lens for Qwen3-8B

**Status:** Draft v1 · **Target model:** Qwen/Qwen3-8B · **Budget class:** ~$150–350 all-in (prototype)
**Audience:** an engineer/agent with no prior context on this line of interpretability work. Everything needed is defined here or linked.

---

## 1. What we are building and why

An **oracle lens** is an interpretability tool that takes a single internal activation vector from a language model and produces a short list of free-form natural-language phrases that, together, approximately "explain" that activation — in the specific sense that vectors derived from those phrases linearly reconstruct a meaningful fraction of the activation's variance. Reading these phrases at interesting token positions reveals latent content (plans, assessments, intermediate reasoning steps) that never appears in the model's output text.

The method was introduced by Anthropic in the appendix "Extending the Jacobian lens to multi-token concepts" of the *global workspace* paper (July 2026), demonstrated on their Haiku 4.5 model. **No code for it was released.** This plan reconstructs it for Qwen3-8B from the paper's description, using two open-source codebases as scaffolding.

### 1.1 Required reading (in this order)

1. **NLA paper:** https://transformer-circuits.pub/2026/nla/ — introduces the verbalizer/reconstructor architecture, activation injection, and the RL training pattern we reuse. Read fully.
2. **Workspace paper:** https://transformer-circuits.pub/2026/workspace/ — main text for the Jacobian lens (J-lens) and the "workspace" concept; then read the appendix *"Extending the Jacobian lens to multi-token concepts"* very carefully — it is the primary spec for this build (template lens → oracle lens, four training stages).
3. **Scaffolding repos:**
   - https://github.com/kitft/natural_language_autoencoders — full NLA training repo (data gen, SFT, GRPO RL, activation injection via `input_embeds`, checkpoint conversion). Our main scaffold.
   - https://github.com/kitft/nla-inference — lightweight inference client, useful reference for injection mechanics.
   - https://github.com/anthropics/jacobian-lens — Apache-2.0 J-lens reference implementation (fit/apply/visualize on HF decoders). Used here for a baseline and layer-selection sanity checks, not on the critical path.
4. **Interactive demos** (build intuition before coding): https://www.neuronpedia.org/jlens and the NLA demo linked from https://www.anthropic.com/research/natural-language-autoencoders

### 1.2 Minimal background (if you read nothing else)

- A transformer's **residual stream** at layer ℓ, position t is a vector h ∈ R^d (d = hidden size). It is the model's working state; it is not human-readable.
- A **Natural Language Autoencoder (NLA)** is a pair of fine-tuned copies of the subject model: an *activation verbalizer* (AV: activation → text) and an *activation reconstructor* (AR: text → activation), jointly trained with RL to minimize round-trip reconstruction error. Powerful, but the jointly-trained AR can co-adapt to the AV's text, which risks confabulated explanations.
- The **Jacobian lens (J-lens)** is a cheap linear readout: it transports an activation to the final-layer basis via an averaged input–output Jacobian and decodes it through the unembedding, giving a ranked list of *single vocabulary tokens* the activation is disposed to make the model say. Grounded and cheap, but single-token only ("blackmail" shows up only as "black").
- The **template lens** extends the J-lens to a fixed vocabulary of multi-token words: for word w, average the model's activations over many passages where w is the natural next word, then center and whiten → a direction t_w readable and steerable like a J-lens vector.
- The **oracle lens** (this project) removes the fixed-vocabulary limit. A *reconstructor* model maps *any* phrase to its template-style direction; an *oracle* model is trained to look at an activation and propose K phrases of length N whose reconstructor-directions, combined by non-negative least squares, reconstruct the activation. Crucially, unlike an NLA, **the reconstructor is trained first on a well-specified objective and frozen during oracle RL**, and reconstruction is constrained to linear combinations of its outputs — the paper's authors argue this makes confabulation less likely, at the cost of lower reconstruction (their Haiku 4.5 oracle explains ~31% of *whitened* activation variance; NLAs reach 0.6–0.8 FVE on a raw metric — different metrics, not directly comparable).

### 1.3 Deliverables

1. A fitted whitening transform and (optional) template-lens vocabulary for Qwen3-8B at the chosen layer.
2. A trained **reconstructor** checkpoint (phrase → whitened activation direction).
3. A trained **oracle** checkpoint (activation → K phrases of length N), SFT version required, RL-refined version optional (gated).
4. An evaluation report: whitened-FVE numbers vs. baselines, plus a qualitative probe gallery (§8).
5. Reusable inference code: `decode(activation, N, K) → [(phrase, coefficient), ...] + FVE`.

---

## 2. Definitions, notation, and global design decisions

Provenance tags used throughout: **[paper]** = specified in the workspace-paper appendix; **[NLA]** = taken from the NLA paper/repo; **[choice]** = our adaptation decision, revisit freely.

- **Subject model:** Qwen/Qwen3-8B. Verify from `config.json`: expected ~36 layers, d_model = 4096. If numbers differ, adjust everything downstream; nothing in this plan depends on exact values. **[choice]**
- **Thinking mode:** disabled (`enable_thinking=False`) everywhere — generation, activation collection, evals — for v1. Thinking traces are scientifically interesting but change the data distribution; treat as a v2 extension. **[choice]**
- **Read layer ℓ\*:** two-thirds depth → layer 24 of 36. **[NLA convention; choice]** Sanity-check before committing (§4.3): fit a J-lens with the `jacobian-lens` repo (~100 prompts suffices per its README) and confirm layer 24 sits inside the band where readouts are interpretable; if the band is centered elsewhere, move ℓ\* to its center. All stages use activations from this one layer, at the position *preceding* a phrase.
- **Whitened space [paper]:** all reconstruction losses, teacher decompositions, rewards, and FVE numbers live in whitened coordinates: x̃ = Σ^(-1/2)(x − μ), where μ, Σ are the mean/covariance of layer-ℓ\* activations over the assistant-turn corpus, with ridge: Σ ← Σ + λI, λ = 10^-2 · tr(Σ)/d **[choice]**. Compute Σ^(-1/2) by eigendecomposition; store μ, Σ^(-1/2), Σ^(1/2).
- **Whitened FVE** of a reconstruction x̂ for target x: 1 − ‖x̃ − x̂̃‖² / ‖x̃ − mean‖² averaged over the eval set (with the mean of whitened data ≈ 0, denominator ≈ ‖x̃‖²). This is the paper's headline metric (Haiku 4.5 oracle: ~31%). Do not mix it up with the NLA papers' raw/normalized FVE.
- **Phrase lengths:** N ∈ {2, 4, 8, 16, 32} tokens; per-example N is fixed and stated in the oracle prompt. **[paper]**
- **On-policy data:** all activations come from text *generated by Qwen3-8B itself* (assistant turns), not from other models' text. See §3 for why and how.
- **Precision/infra:** bf16 training, FSDP or ZeRO-2/3 across one 8×H100 (or 8×A100-80GB) node. 8B full fine-tuning with Adam ≈ 130 GB of states → ~16 GB/GPU sharded, comfortable. **[choice]** LoRA would be cheaper but is unvalidated for this method (both papers used full fine-tuning); if you use LoRA, treat it as an experiment variable and say so in the report.

---

## 3. Stage 0 — On-policy data generation (WildChat regeneration)

**Why regeneration:** WildChat-1M (HF: `allenai/WildChat-1M`) contains real user prompts but the assistant responses were written by GPT-3.5/4. Activations must be collected on Qwen3-8B's *own* outputs, or the oracle learns to explain off-policy states. So: take WildChat's **first user turn only**, generate a fresh Qwen3-8B response, and use those responses as the corpus. First-turn-only avoids the on-policy contamination problem in multi-turn context and keeps input cost linear.

**Spec:**
- ~100k conversations, English-filtered **[choice]** (WildChat is ~68% English; the multilingual tail is a v2 question), deduplicated prompts.
- Sampling: temperature 0.7, top-p 0.8 (Qwen3 non-thinking recommended settings — verify against the model card), max ~600 new tokens, chat template applied exactly as the tokenizer defines it.
- Output: ~45M assistant tokens. Cost: ~$25 via an API at $0.117/$0.455 per Mtok in/out, or ~1–2 H100-hours locally with vLLM (preferred — see warning).
- **Critical consistency warning:** the chat template, system prompt (use none), sampling settings, and thinking-mode flag used at *generation* time must exactly match those used at *activation collection* time (§4). If you generate via API and collect locally, verify token-level identity on a sample: re-tokenize the API transcript locally and confirm the rendered prompt+response token sequence matches what local generation infrastructure would produce. Any silent mismatch quietly breaks the on-policy assumption. Generating locally with vLLM and saving token IDs directly is the safest path.
- **Position sampling:** from assistant spans only, sample ~10 positions per response (uniform over the span, but always also include a few *delimiter* positions — sentence-final periods, newlines, end-of-turn — the paper found these carry qualitatively different, commentary-like content). Total ~1M positions.
- **Splits (disjoint by conversation):** reconstructor-train 300k · dictionary 150k · teacher 250k · RL pool 250k · held-out eval 50k.

**Exit criteria:** corpus generated; token-identity check passes on ≥100 random samples; splits materialized as (conversation_id, position) lists.

---

## 4. Stage 1 — Activation collection, whitening, layer check

**Compute: ~1–2 H100-h.**

1. Run batched prefill over all corpus transcripts, hook layer ℓ\*'s residual stream, save activations at the sampled positions (fp16). 1M × 4096 × 2B ≈ 8 GB — trivially storable. Save *position metadata* (token, is-delimiter flag, offset in turn) alongside; you will need it for evals.
2. Estimate μ, Σ on the reconstructor-train + teacher splits (≥550k vectors ≫ d = 4096, so Σ is well-estimated); apply ridge; eigendecompose; persist the whitening transform. Sanity-check conditioning: after ridge, condition number should be finite and the whitened data should have ~unit variance per direction on held-out.
3. **Layer sanity check [choice]:** fit a J-lens for Qwen3-8B (`jlens.fit`, ~100–1000 prompts) and eyeball slice visualizations on a few prompts (the repo's walkthrough notebook renders these). Confirm layer 24 gives interpretable readouts; relocate ℓ\* if not. This costs ~1 GPU-hour and de-risks the single most irreversible choice in the plan.

**Exit criteria:** whitening transform saved and validated; ℓ\* confirmed or revised; activation store complete.

---

## 5. Stage 2 — Reconstructor (phrase → whitened activation direction)

**The most important stage. Every downstream stage inherits its quality. Compute: ~3–5 H100-h.**

**Task [paper]:** given a phrase (a span of N tokens of on-policy text, N uniform in 1–32), predict the (whitened, unit-normalized) residual-stream activation at the position *immediately preceding* the phrase. Loss: MSE between unit-normalized whitened prediction and unit-normalized whitened target — equivalently cosine error in the whitened metric. The intuition: this generalizes the template lens — the reconstructor learns "what does the model's state look like right before it says X," for arbitrary X.

**Data:** 300k pairs from the reconstructor split, generated mechanically (sample position → phrase = next N tokens, target = activation at position−1... precisely: target is the activation at the sampled position, phrase is the N tokens that follow it). N sampled uniformly from 1–32 **[paper]**. Zero labeling cost — this is one of the method's advantages over NLA warm-starts, so if held-out error hasn't plateaued at 300k, generating 1M more pairs is nearly free.

**Architecture [choice, mirroring NLA's AR]:** initialize from Qwen3-8B; feed the phrase with a minimal fixed wrapper (e.g., `Phrase: "<phrase>"` — keep it boring and constant); take the last-token hidden state at the final layer; apply a learned affine head → R^4096; whiten+normalize; cosine loss against the whitened+normalized target. The NLA repo's AR training path (truncated model + affine head, text→vector regression) is the closest existing code; adapt it rather than writing from scratch.

**Hyperparameters [choice, starting points]:** LR 1e-5 (head 10×), cosine decay, batch 128, 1–2 epochs, bf16, FSDP.

**Evaluation & gate (M2):**
- Held-out whitened cosine, reported by phrase length N (expect monotone degradation with N; a reconstructor that is only good at N=1–2 is a failure).
- **Continuation-FVE probe:** on held-out positions, take the *true* next-N-token phrase, map it through the reconstructor, NNLS-refit the single coefficient, measure whitened FVE. This number is the "PastLens-style floor" — the FVE obtainable from trivially correct continuations — and later becomes the baseline the oracle must beat at delimiter positions.
- Controls: shuffled-phrase pairing and predict-the-mean must both score dramatically worse. If they don't, the metric or whitening is broken.
- **Gate:** no absolute threshold is defensible a priori (nothing published to anchor on for 8B); gate on (a) clear separation from controls, (b) held-out cosine still improving or plateaued — if still improving, add data before proceeding, (c) continuation-FVE probe meaningfully above zero at N ≤ 8.

---

## 6. Stage 3 — Dictionary + teacher labels (NN-OMP)

**Compute: ~1 H100-h (dictionary) + ~3–8 GPU-h (OMP, pure matmuls).**

### 6.1 Dictionary [paper, scaled down]
From the dictionary split's 150k start positions, take the 2-, 4-, 8-, 16-, and 32-token continuations (750k phrases), deduplicate within each length (→ ~500k), and run every phrase through the trained reconstructor. Store unit-normalized whitened direction vectors per length bucket (~4 GB fp16). The dictionary does **not** limit what the final oracle can say — it exists only to manufacture supervised targets.

### 6.2 Teacher decompositions [paper]
For each of 250k activations in the teacher split:
1. Sample one phrase length N (uniform over the five lengths) — each decomposition is restricted to a single length. **[paper]**
2. Restrict to a **random half** of that length's dictionary entries. **[paper — this prevents the distilled oracle from memorizing a fixed ranking over dictionary entries; do not skip it.]**
3. Run **non-negative orthogonal matching pursuit** in whitened space: greedily select the dictionary direction with the largest positive correlation with the current residual, refit all selected coefficients by non-negative least squares, subtract, repeat. Stop at 16 atoms **[paper]** or when marginal FVE gain < 0.5% **[choice]**.
4. Record: ordered phrase list, fitted coefficients, cumulative whitened FVE. **[paper]**

Implementation notes: this is embarrassingly parallel batched GPU matmul (500k × 4096 dictionary against batches of activations); write it directly in PyTorch rather than hunting for a library. Keep the per-example dictionary-half selection seeded/reproducible.

**Evaluation & gate (M3):** the teacher's mean whitened FVE is the effective supervised ceiling for the oracle's SFT phase and the single best health check of the whole pipeline so far. Report it overall, by N, and by delimiter/non-delimiter. **Gate [choice]:** teacher FVE ≥ ~15% on average (the Haiku 4.5 *oracle* reached 31% after RL; the teacher presumably sat in that vicinity; an 8B model plausibly lands lower — but single-digit teacher FVE means the reconstructor or whitening is too weak to bother training an oracle on). If the gate fails: more reconstructor data, bigger dictionary, revisit λ and ℓ\* — in that order.

---

## 7. Stage 4 — Oracle SFT, then optional GRPO RL

### 7.1 Activation injection [NLA]
The oracle is a second fine-tuned copy of Qwen3-8B that must *perceive* an activation. Follow the NLA repo's mechanism: construct the input embedding sequence directly and splice in the (whitened **[choice]**) activation as a pseudo-token at a fixed slot in a fixed prompt, scaled by α so its norm matches the typical input-embedding norm at that layer of processing. α is documented as a touchy knob in the NLA work — before the main run, do a mini-sweep (3–4 values, 2k-step SFT probes, pick by held-out teacher-imitation loss).

Prompt template **[choice]** (fixed, boring, with N and K explicit per the paper):
```
Activation: <INJECT>
List exactly {K} phrases of exactly {N} tokens each that describe
what the model producing this activation is about to say or is
considering. One phrase per line.
```
Target: the teacher's first K phrases (K uniform in 4–16 during training **[choice]**), in teacher order, newline-separated.

### 7.2 SFT
250k teacher-labeled examples, LR 1e-5, batch 128, 1 epoch, bf16, FSDP. **Compute: ~5–10 H100-h.**

**Evaluation & gate (M4):** on held-out activations, sample K phrases, map each through the *frozen* reconstructor, recover coefficients by non-negative least squares against the true activation (this NNLS refit is the standard decode path **[paper]**), compute whitened FVE. **Gate [choice]:** SFT oracle ≥ 70% of teacher FVE, format validity ≥ 95% (right K, right N ± 1 token). Also run the qualitative probe set (§8.2) now — **the SFT checkpoint is the primary de-risking milestone; expect it to already produce usable readouts, since the teacher decompositions are the real signal and RL is refinement.**

### 7.3 GRPO RL — optional, gated on M4 qualitative results
Reward **[paper]**: whitened FVE of the NNLS-refit reconstruction of the generated phrase list, minus small penalties for deviating from the requested K and N. Reconstructor stays **frozen** — this is the property that distinguishes the oracle lens from an NLA; do not be tempted to unfreeze it.
Config **[choice]**: GRPO per the NLA repo, group size 8, batch 256 activations from the RL split, temperature 1.0 rollouts, small KL to the SFT checkpoint, 500–1000 steps with FVE evaluated every 100. Stop when held-out FVE gains < 0.5% per 200 steps. **Compute: ~20–50 H100-h.** Rationale for the small budget: published NLA training shows FVE gains roughly linear in log(steps) — most value arrives early — and this method's known ceiling is modest (~31% whitened FVE at Haiku 4.5 scale); RL's specific job here is only to let free-form phrases edge out dictionary-constrained ones.
**RL failure mode to monitor:** collapse onto generic high-coverage phrases repeated across activations. Track phrase entropy across a fixed probe batch each eval; if it drops while FVE stalls, raise KL or stop.

---

## 8. Evaluation suite

### 8.1 Quantitative (all in whitened FVE on the 50k held-out split, reported overall / by N / by K / by position class)
| Comparison | What it establishes |
|---|---|
| Random dictionary phrases (matched K, N) + NNLS | floor |
| True-continuation phrase (K=1..4) + NNLS | "trivial prediction" baseline; the oracle must beat it clearly at delimiter positions, may only tie at ordinary positions |
| Teacher NN-OMP decompositions | supervised ceiling for SFT |
| Oracle SFT / Oracle RL | the deliverables |
| J-lens top-k tokens as length-1 phrases | single-token method comparison |

### 8.2 Qualitative probe gallery (~40 hand-checked prompts, adapted from the papers' demonstrations)
Multi-hop latent facts, 10 (e.g., "The currency of the country shaped like a boot is…" — does *Italy* appear in the readout at pre-answer positions?); buggy code, 5 (dict-mutation-during-iteration style — does the readout name the bug/exception before any output?); hold-in-mind / suppression instructions, 5; planning tasks, 5 (rhyme-constrained couplets — read at the line-break token); harm-assessment prompts, 5 (e.g., an overdose-range dosage question — does danger-recognition appear before the reply?); and a **delimiter scan** over 10 full transcripts checking for the paper's signature phenomenon: at ordinary tokens, readouts look like continuations; at periods/newlines/turn-ends, they shift toward first-person-ish situation commentary that sampled continuations at the same positions do *not* contain. Score each probe pass/fail with notes; this gallery, not the FVE number, is what decides whether the tool is useful.

### 8.3 Expectations management
The workspace paper's showcase readouts come from Haiku 4.5, and it reports workspace phenomena strengthening with model scale. An 8B oracle lens that is mechanically sound may still read out fainter, noisier content. A pipeline that passes M2–M4 with a boring gallery is a *successful negative-ish result* at 8B — the follow-up is the same pipeline on a larger Qwen (a pre-fitted J-lens for Qwen3.6-27B already exists in the jacobian-lens Hub assets), not more 8B tuning.

---

## 9. Budget, schedule, milestones

| Milestone | Contents | Compute | Gate |
|---|---|---|---|
| M0 | WildChat regeneration + splits | ~$25 API or 1–2 H100-h | token-identity check |
| M1 | Activations + whitening + layer check | 1–2 H100-h | conditioning, J-lens band |
| M2 | Reconstructor | 3–5 H100-h | §5 gate |
| M3 | Dictionary + teacher | 4–9 GPU-h | teacher FVE ≥ ~15% |
| M4 | Oracle SFT + full eval + gallery | 5–10 H100-h | ≥70% of teacher FVE; gallery review |
| M5 (opt.) | GRPO RL + final eval | 20–50 H100-h | FVE plateau; entropy healthy |

Totals: **~15–25 H100-h to M4** (the de-risk point), **~40–75 H100-h with RL**; multiply by 1.5–2× wall-clock for evals/restarts. At $2–3/H100-h: **~$50–100 to M4, ~$150–350 all-in.** Sequential wall-clock on one 8×H100 node: M0–M4 in ~2–3 working days, M5 adds ~1–2.

## 10. Risks
1. **Reconstructor is the load-bearing wall.** Weakness here silently caps everything; hence the M2 controls and the cheap-data escape hatch.
2. **Injection scale α** — known-touchy; mini-sweep before committing.
3. **On-policy mismatch** (template/sampling/thinking-mode drift between generation and collection) — the token-identity check exists for this; run it.
4. **Whitening pathologies** — ridge λ too small amplifies noise directions (garbage teacher atoms), too large reverts toward raw-space geometry; if teacher FVE is bad, sweep λ one order of magnitude each way before touching anything else.
5. **RL phrase collapse** — entropy monitor, KL, or simply ship the SFT checkpoint.
6. **Small-model faintness** (§8.3) — an interpretive risk, not an execution one; the gallery is designed to detect it honestly.

## 11. Provenance and honesty notes for the report
This plan reconstructs an unreleased method from its paper description; there is no public replication to compare against, and the ~31% FVE reference point is a different (larger) model on a different data distribution. All compute figures are FLOP-arithmetic estimates, not measurements. Tag every deviation from the paper spec ([choice] items above) in the final report, and report negative results at full volume — the delimiter-commentary phenomenon and the SFT-vs-RL gap are open empirical questions this build can genuinely answer.

## 12. Reference index
- Workspace paper (primary spec, see appendix): https://transformer-circuits.pub/2026/workspace/
- NLA paper (architecture + RL pattern): https://transformer-circuits.pub/2026/nla/
- NLA training repo (scaffold): https://github.com/kitft/natural_language_autoencoders · inference: https://github.com/kitft/nla-inference
- J-lens repo (baseline/layer check): https://github.com/anthropics/jacobian-lens · demo: https://www.neuronpedia.org/jlens
- Data: https://huggingface.co/datasets/allenai/WildChat-1M · Model: https://huggingface.co/Qwen/Qwen3-8B
