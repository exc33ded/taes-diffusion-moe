# Research Plan v4 — Training-Free Timestep-Aware Expert Steering for Diffusion MoE

**Author:** Sarim · **Date:** 31 July 2026
**Target:** Journal — Pattern Recognition, Neurocomputing, or IEEE TIP
**Budget:** ₹0. Kaggle free tier only.
**Supersedes:** v1, v2, v3 (all required paid compute)

---

## 1. Working title

*TAES: Training-Free Timestep-Aware Expert Steering for Memory-Efficient Diffusion Mixture-of-Experts*

## 2. Why this version exists

v3 required 300–500 A100-hours. Your actual budget is Kaggle's free tier — 30 GPU-hours/week on P100 or dual T4, roughly 120–200 A100-equivalent hours over five months. v3 did not fit. This version does, and it is still a **method paper**, not an analysis, because it belongs to a category that is cheap by construction: **training-free methods**.

Training-free methods are real contributions with real baselines — reviewers do not discount them — and they cost inference compute only. That is the one category where a zero-budget researcher is not at a disadvantage.

## 3. The gap, stated precisely

There is a mature and growing family of **training-free expert steering** methods for MoE. All of them are language:

| Method | Year | Mechanism |
|---|---|---|
| [RICE](https://consensus.app/papers/details/3132c0b6b6a85fcb95356dfb23793c8a/) | 2025 | nPMI identifies "cognitive experts", reinforced at inference |
| [MoTE](https://consensus.app/papers/details/b8f295008666522cb54665e5aabe1dde/) | 2025 | Localizes behaviour to ~10 experts, switches them off |
| [Ban&Pick](https://consensus.app/papers/details/d0e3846be58752f4b9b739e828da7dc4/) | 2025 | Post-training routing: reinforce key experts, prune redundant |
| [DERN](https://consensus.app/papers/details/d2c5c8e4b7e15bbca23316661376ef75/) | 2025 | Retraining-free expert pruning + neuron recombination |
| [DSMoE](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) | 2026 | Training-free domain steering, zero added inference cost |
| [MASCing](https://consensus.app/papers/details/8836fce80a9453c8bfdcfa23ebc8b0e1/) | 2026 | Steering masks applied to routing gates |
| [SSMoE](https://consensus.app/papers/details/ab80d6eef03a5246b37875a3cfdb559d/) | 2026 | Training-free router from expert-weight eigenvectors |

**None has been applied to image diffusion.** And there is a principled reason it is not a trivial port, which is what makes this a paper rather than an exercise.

## 4. The technical insight that makes it novel

Every method above scores experts using **token-level semantic statistics** — nPMI over tokens, router frequency per domain. Diffusion has no token semantics. What it has instead is an axis that does not exist in language at all: **the denoising timestep**.

DiT-MoE already reported that expert selection in diffusion is driven by timestep and spatial position, and that specialization is "concentrated at the early time step and then gradually uniform after half." So a naive port — compute one global expert-importance ranking per domain, prune — throws away the dominant structure and should fail.

**TAES makes expert importance a function of (domain, timestep band) rather than domain alone.** That single change is diffusion-specific, cheap, and directly testable against the naive port as a baseline. The ablation *is* the contribution.

## 5. Method — TAES

Entirely training-free. No gradients anywhere.

**Step 1 — Calibration (cheap).** Take 200–500 in-domain images. Run forward diffusion to sample noisy latents across the full t range. Single forward pass each, hooks on every router. Record gate scores, top-k selections, and expert output L2 norms, indexed by (domain, timestep, layer, expert).

**Step 2 — Banded importance scoring.** Partition t into B bands (start B=4). For each (domain d, band b, layer l, expert e):

```
I(e | d, b, l) = freq(e) · mean_gate_score(e) · mean_output_L2(e)
```

The multiplicative form follows EASY-EP and Ban&Pick; the banding is ours.

**Step 3 — Two deployment modes.**
- **Memory mode.** Keep the union of top-k experts across all bands; physically delete the rest and re-index routers. Yields a genuinely smaller checkpoint on disk. This is the deliverable you originally wanted.
- **Compute mode.** Use the per-band set S_b as an inference-time mask on the routing gates. Fewer experts active per step; no weights deleted; composable with offloading.

**Step 4 — Optional gate steering.** Add a fixed bias to the gate logits of high-importance experts (RICE/Ban&Pick style). Costs nothing at inference and may *improve* FID rather than merely preserving it — a free upside if it works.

## 6. Experiments

**Backbone.** `feizc/DiT-MoE-S/2`, ImageNet-256 latents, 8 experts/layer, released checkpoints. Single backbone — B/2 only if time and quota permit.

**Domains.** Four ImageNet superclasses (animals / vehicles / food / structures). Labels are free.

**Baselines.** All training-free, so all cheap:
1. No pruning (upper bound)
2. Random expert removal
3. Frequency-only ranking
4. **Global (non-banded) importance** — the naive LLM port. *This is the critical comparison.*
5. DERN-style retraining-free pruning, adapted
6. INT8 weight quantization
7. TAES (ours)
8. TAES + INT8

**Metrics.** FID-10k for sweeps, FID-50k for the final table, precision/recall, peak GPU memory, latency, on-disk parameter count.

**Ablations.** Number of bands B ∈ {1, 2, 4, 8} — B=1 *is* baseline 4, so this ablation doubles as the core result · k sweep · each scoring term dropped · calibration size 50/200/500 · band boundary placement · cross-domain transfer (prune on animals, test on vehicles).

## 7. Compute — fits the free tier

| Item | P100-hours |
|---|---|
| Calibration across 4 domains | ~10 |
| Expert scoring | negligible (CPU) |
| FID-10k sweeps, ~25 configs @ 25 DDIM steps | ~75 |
| FID-50k final table, ~6 configs | ~60 |
| Reruns, seeds, mistakes | ~40 |
| **Total** | **~185** |

At Kaggle's 30 h/week that is **6–8 weeks of quota**, comfortably inside a five-month calendar. **Cost: ₹0.**

Cost controls if you run tight: DDIM 20–25 steps rather than 50 for all sweeps; FID-10k everywhere except the final table; precompute and cache VAE latents once; drop to two domains.

## 8. Why reviewers should accept it

- It is a method with a mechanism, not an observation.
- The core claim is falsifiable and has a clean ablation: banded beats global.
- It targets a real deployment problem — MoE memory footprint — with a measurable Pareto curve.
- The training-free framing means results are reproducible by anyone, which reviewers like.
- Honest scope: single backbone, one dataset. Journals accept this when the claim is correspondingly scoped. Do not overclaim generality.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Banding gives no gain over global | That is a real negative result about diffusion routing and still publishable, but check it in week 3 before committing |
| DiT-MoE-S/2 too small for meaningful pruning at 8 experts | Test early; if 8 experts is too coarse, switch to a Diff-MoE config with more experts |
| Kaggle quota exhausted mid-experiment | Cache everything to Kaggle Datasets; design every run to be resumable |
| FID unstable at 10k | Report ±std over 3 seeds; use 50k for anything load-bearing |
| Someone ports training-free steering to diffusion first | Highest risk. Step 1–2 is ~3 weeks; arXiv early |

## 10. Timeline (~4 months, part-time, free tier)

- **Wk 1–2.** Load DiT-MoE-S/2 on Kaggle; reproduce sampling and a reference FID. Hard gate.
- **Wk 3–4.** Calibration + banded scoring. Produce the importance-vs-timestep heatmaps.
- **Wk 5.** **Go/no-go:** does banded importance differ from global? If yes, proceed. If no, pivot to reporting it as a negative finding.
- **Wk 6–9.** Memory and compute modes; baselines 1–5.
- **Wk 10–12.** Quantization baselines, combination, Pareto curve.
- **Wk 13–15.** Ablations, cross-domain transfer.
- **Wk 16–18.** Figures, writing, arXiv, submit.

## 11. Next actions

1. Set up Kaggle notebook; clone `feizc/DiT-MoE`; confirm checkpoints load and sample. **Nothing else matters until this works.**
2. Read DSMoE and Ban&Pick closely — their scoring functions are what you are adapting, and you need the exact formulations.
3. Read the DiT-MoE expert-specialization section; its timestep findings are the empirical basis for banding.
4. Write the router telemetry hook.
5. Install TooManyPapers and load this reference list into it.

---

## References

**Training-free MoE steering — the family being ported**
- [RICE: Two Experts Are All You Need for Steering Thinking](https://consensus.app/papers/details/3132c0b6b6a85fcb95356dfb23793c8a/) — Wang et al., 2025
- [Mixture of Tunable Experts (MoTE)](https://consensus.app/papers/details/b8f295008666522cb54665e5aabe1dde/) — Dahlke et al., 2025
- [Ban&Pick](https://consensus.app/papers/details/d0e3846be58752f4b9b739e828da7dc4/) — Chen et al., 2025
- [DERN](https://consensus.app/papers/details/d2c5c8e4b7e15bbca23316661376ef75/) — Zhou et al., 2025
- [DSMoE / Do Domain-specific Experts exist in MoE-based LLMs?](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) — Do et al., 2026
- [MASCing](https://consensus.app/papers/details/8836fce80a9453c8bfdcfa23ebc8b0e1/) — te Lintelo et al., 2026
- [SSMoE](https://consensus.app/papers/details/ab80d6eef03a5246b37875a3cfdb559d/) — Do et al., 2026
- [EASY-EP](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) — Dong et al., 2025

**Diffusion MoE — the target**
- [DiT-MoE](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) — Fei et al., 2024
- [SharpMoE](https://consensus.app/papers/details/dbd920a3439c5825b6c78aa73f87c7c2/) — Deng et al., 2026
- [ProMoE](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) — Wei et al., 2025
- [ALTER](https://consensus.app/papers/details/70b9be1944e252b38596709c856d87af/) — Yang et al., 2025

**Context / caveats**
- [EMO](https://consensus.app/papers/details/0f6963a6f9105932b765ed00dfd6ab1c/) — 2026 (modularity needs pretraining — we test the cheap alternative)
- [The Illusion of Specialization](https://consensus.app/papers/details/4c2a11747360510db1349851fa791f16/) — 2026
- [Expert-Data Alignment Governs Generation Quality in DDMs](https://consensus.app/papers/details/c82f0b8290055b69b3299cfdfd31a7f7/) — 2026

**Code**
- [feizc/DiT-MoE](https://github.com/feizc/DiT-MoE) · [giangdip2410/Domain-specific-Experts](https://github.com/giangdip2410/Domain-specific-Experts) · [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE)
