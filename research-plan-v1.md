# Research Plan v1 — Domain-Conditioned Expert Localization in Diffusion MoE

**Author:** Sarim · **Date:** 31 July 2026 · **Target:** mid-tier conference / journal (WACV, BMVC, ICASSP, Pattern Recognition, Neurocomputing)

---

## 1. The one-sentence pitch

Vanilla MoE diffusion transformers route by *timestep and spatial position*, not by *semantic content* — which means the obvious recipe of "profile a domain, keep its experts, drop the rest" does not work out of the box. We show why, and propose a routing intervention that makes domain-conditioned expert localization work, yielding permanently smaller domain-specialized generators.

## 2. Why this framing, and not the one I started with

My original idea was: profile a target domain, find the expert subset it activates, prune everything else. That is exactly [EASY-EP](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) for LLMs, and it works there because language tokens are semantically dense.

The critical finding that reframes the paper is in **DiT-MoE** (Fei et al., 2024), which analyzed expert specialization in diffusion transformers and reported:

> "Expert selection shows preference with spatial position and denoising time step, while **insensitive with different class-conditional information**."

So the naive transplant fails. **This is good news.** A negative result you then fix is a much better paper than a positive result that was obvious. [ProMoE](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) (2025) independently confirms that vision MoE needs *explicit routing guidance* to specialize at all, because visual tokens have spatial redundancy and functional heterogeneity. That's the mechanism our fix will exploit.

## 3. Research questions

- **RQ1 (diagnostic).** To what extent is expert selection in a pretrained diffusion MoE separable by semantic domain, once timestep and spatial position are controlled for? Prior work reports "insensitive" but does not disentangle the confounds.
- **RQ2 (method).** Can a lightweight router intervention — fine-tuning *only* the routers on a small in-domain set, with a specialization objective — induce a stable, sparse, domain-specific expert subset without touching expert weights?
- **RQ3 (deployment).** At matched memory budgets, how does domain-localized expert pruning compare against (a) uniform expert pruning, (b) INT4/INT8 quantization, (c) both combined?

RQ3 is the safety net: even if RQ2 fails, RQ1 + RQ3 is a publishable empirical paper.

## 4. Positioning against the closest work

| Work | Specialization axis | What we do differently |
|---|---|---|
| DiT-MoE (2024) | Analysis only; timestep + spatial | We disentangle domain from timestep/position and act on it |
| DiffPruning (2024) | Timestep intervals | Domain, not time |
| ALTER (2025) | Timestep + layer pruning | Orthogonal; can be composed with ours |
| ProMoE (2025) | Conditional vs. unconditional tokens | We guide by *domain*, not by conditioning role |
| Dense2MoE (ICCV'25) | Dense→MoE conversion | We compress an existing MoE, not build one |
| EASY-EP (2025) | Domain, but LLM-only | We port it to diffusion and show it needs repair |
| Diff-MoE (ICML'25) | Time-aware + space-adaptive | Same confound we control for |

**Novelty claim:** first study of *semantic-domain* expert localization in diffusion MoE, and first to show the LLM recipe does not transfer without a routing intervention.

## 5. Method sketch

**Stage 0 — Routing telemetry.** Instrument a pretrained DiT-MoE. For each layer, log the expert-selection distribution as a function of (domain d, timestep t, spatial position p). Compute mutual information I(expert; domain | t, p). This is the diagnostic that answers RQ1 and is publishable on its own.

**Stage 1 — Router-only domain fine-tuning.** Freeze all expert weights and the backbone. Fine-tune *only* the router linear layers on ~5–20k in-domain images, with a combined loss:
- standard diffusion loss
- a **routing concentration loss** (entropy penalty over the marginal expert distribution for the domain) — pushes mass onto few experts
- a **timestep-consistency term** — discourages the retained set from churning across t, so a single static subset suffices

This is the novel bit. It is cheap: routers are a tiny fraction of parameters, so this trains on one GPU.

**Stage 2 — Static expert extraction.** Rank experts by domain-conditioned activation mass × output L2 norm (EASY-EP's criterion, adapted). Keep top-k, physically delete the rest, re-index the router. Output is a genuinely smaller checkpoint — not a masked one. This is the "lightweight model tailored to your need" you wanted.

**Stage 3 — Short recovery fine-tune.** Brief LoRA or full fine-tune of the surviving experts to recover FID.

## 6. Experimental design

**Backbone.** `feizc/DiT-MoE` — DiT-MoE-S/2 and DiT-MoE-B/2, released checkpoints, ImageNet-256 latents, 8 experts/layer. This is the only realistic choice: it's open, small, and the paper we're arguing with. Add `kunncheng/Diff-MoE` (ICML'25) as a second backbone for generality.

**Domains.** ImageNet-256 superclasses (animals / vehicles / food / structures) give clean class labels for free. Add one out-of-distribution domain (FFHQ faces or a style dataset) to test cross-dataset transfer.

**Baselines.** Uniform random expert pruning · frequency-only pruning (no router FT) · Diff-Pruning (Fang et al.) · INT8/INT4 weight quantization · our method · ours + quantization.

**Metrics.** FID-10k and FID-50k, CLIP score or IS, precision/recall, **peak GPU memory**, wall-clock latency, parameter count. Report a memory-vs-FID Pareto curve — that's the figure reviewers will look at.

**Ablations.** k sweep (fraction of experts kept) · each loss term removed · router-FT vs. no router-FT (this is the money ablation) · per-layer vs. global k · domain sample size (20 / 200 / 5k images).

## 7. Compute requirement — the honest number

**Verdict: Colab Pro+ (~$50/month) is sufficient. Free tier is not.**

| Stage | Requirement |
|---|---|
| Routing telemetry (Stage 0) | Inference only. Free T4 handles it. ~10 GPU-hours. |
| Router-only fine-tuning (Stage 1) | Routers are <1% of params but need full forward/backward. A100-40GB, ~20–40 GPU-hours per domain. |
| Recovery fine-tune (Stage 2–3) | ~30–60 GPU-hours per configuration. Main cost. |
| FID evaluation | 50k samples per config. ~2–4 GPU-hours each, and you'll have ~20 configs. Budget 60 hours. |

**Total: roughly 300–500 A100-hours** across all domains, baselines and ablations. Colab Pro+ gets you there over ~2–3 months. If you want to move faster, ~$300 of RunPod A100 time compresses it to ~4–6 weeks.

Do **not** attempt FLUX, HunyuanImage-3.0, or Nucleus-Image (17B). They are not reachable on this budget and you do not need them — DiT-MoE-S/B on ImageNet-256 is an accepted scale for this class of claim.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RQ1 confirms domains are genuinely inseparable, even controlled | Pivot to RQ3 — the pruning-vs-quantization Pareto study stands alone as a solid empirical paper |
| Router-only FT is too weak to induce specialization | Escalate to router + LoRA-on-experts; report the router-only result as a negative ablation |
| FID is noisy at small scale | Use FID-50k with fixed seeds, report ±std over 3 runs, add precision/recall |
| DiT-MoE checkpoints don't load cleanly | Fall back to Diff-MoE; budget one week for infrastructure |
| Scooped mid-project | Post to arXiv as soon as RQ1 results are solid — the diagnostic alone is a claim worth staking |

## 9. Timeline (assuming Colab Pro+, part-time)

- **Weeks 1–2.** Environment, load DiT-MoE checkpoints, reproduce reported FID. Non-negotiable sanity gate.
- **Weeks 3–5.** Stage 0 telemetry. Produce the I(expert; domain | t, p) heatmaps. **Decision point:** if signal exists, continue; if not, pivot to RQ3.
- **Weeks 6–9.** Implement and tune Stage 1 router fine-tuning.
- **Weeks 10–13.** Extraction, recovery fine-tuning, full baseline sweep.
- **Weeks 14–16.** Ablations, figures, writing.
- **Week 17.** arXiv preprint, then submit.

## 10. Immediate next actions

1. Clone `feizc/DiT-MoE`, confirm checkpoints download and sample correctly on a free T4.
2. Read the DiT-MoE paper's Section on expert specialization in full — the exact experiment they ran determines how strong our "they didn't control for confounds" claim can be.
3. Write the Stage 0 telemetry hook (a forward hook on each router logging argmax + gate scores).
4. Pick target venue and check its next deadline to fix the timeline.

---

## References

- [Scaling Diffusion Transformers to 16 Billion Parameters (DiT-MoE)](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) — Fei et al., 2024
- [Routing Matters in MoE (ProMoE)](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) — Wei et al., 2025
- [Domain-Specific Pruning of Large MoE Models with Few-shot Demonstrations (EASY-EP)](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) — Dong et al., 2025
- [ALTER: All-in-One Layer Pruning and Temporal Expert Routing](https://consensus.app/papers/details/70b9be1944e252b38596709c856d87af/) — Yang et al., 2025
- [Mixture of Efficient Diffusion Experts (DiffPruning)](https://consensus.app/papers/details/e34f2879033a544abb3ba6065654f23a/) — Ganjdanesh et al., 2024
- [Dense2MoE: Restructuring Diffusion Transformer to MoE](https://consensus.app/papers/details/d4719f1be64f5755be0a9d6296f09539/) — Zheng et al., ICCV 2025
- [MELINOE: Fine-Tuning Enables Memory-Efficient Inference for MoE Models](https://consensus.app/papers/details/60fd2749bc165ae1aac1f2ef4799416f/) — Raje et al., 2026
- [Structural Pruning for Diffusion Models (Diff-Pruning)](https://consensus.app/papers/details/a3fe817bd0195f96950dfbe7f5cbe8b2/) — Fang et al., 2023
- [EC-DiT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing](https://consensus.app/papers/details/a712c6391d3d5d908beda79f59ca143a/) — Sun et al., 2024
- Code: [feizc/DiT-MoE](https://github.com/feizc/DiT-MoE) · [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE)
