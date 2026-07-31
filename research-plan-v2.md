# Research Plan v2 — Do Domain-Specific Experts Exist in Diffusion MoE?

**Author:** Sarim · **Date:** 31 July 2026 · **Target:** mid-tier conference / journal (WACV, BMVC, ICASSP, Pattern Recognition, Neurocomputing)
**Supersedes:** research-plan-v1.md (framing revised after 2026 literature check)

---

## 1. What changed from v1

I re-ran the search restricted to 2025–2026 and found two things that matter.

**First: the LLM side is now dead ground.** In 2026 alone: *Less is MoE: Trimming Experts in Domain-Specialist Language Models*, *Domain-Specific Expert Pruning for MoE LLMs* (PMLR v317), *REAP the Experts*, *AIMER*, *ConMoE*, *MoE Pathfinder*, *FlexMoE*, *Attribution-Guided and Coverage-Maximized Pruning*, plus a dedicated study on pruned MoE reliability in biomedicine. Writing anything about LLM expert pruning now means competing with a dozen 2026 papers. **Confirmed: do not go there.**

**Second, and this is the opening:** there is a live, unresolved disagreement in 2026 about whether domain-specific experts exist at all — and it is being fought entirely in language.

- [Do Domain-specific Experts exist in MoE-based LLMs?](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) (Do et al., 2026) evaluates ten MoE LLMs from 3.8B–120B and answers **yes**, providing empirical evidence and building DSMoE on top of it.
- [The Illusion of Specialization](https://consensus.app/papers/details/4c2a11747360510db1349851fa791f16/) (Wang et al., 2026) answers **essentially no** — it finds a domain-invariant "Standing Committee," a compact expert coalition capturing most routing mass *across* domains, and argues specialization is far less pervasive than assumed.
- [EMO](https://consensus.app/papers/details/0f6963a6f9105932b765ed00dfd6ab1c/) (Wang et al., 2026) splits the difference with the most actionable finding: in *standard* MoEs, restricting inference to a domain's expert subset causes **severe degradation** — you have to bake modularity in during pretraining. EMO retains 25% of experts for a 1% drop; standard MoEs "break under the same setting."

**Nobody has asked this question of diffusion models.** That is the paper.

## 2. The pitch

Three 2026 papers disagree about whether MoE experts specialize by semantic domain, and all three study language. Diffusion transformers are the harder and more interesting case, because their routers are known to key on *timestep and spatial position* rather than semantics. We run the domain-specificity audit on diffusion MoE, resolve which side of the controversy holds there, and turn the answer into a router intervention that makes domain-localized expert pruning actually work — producing permanently smaller, domain-specialized image generators.

This is your original idea, but positioned as an answer to an open question that the field is currently arguing about, rather than as one more pruning method. That framing is what gets a mid-tier paper accepted.

## 3. Research questions

- **RQ1 (the audit).** Do domain-specific experts exist in diffusion MoE? Apply a Standing-Committee-style group-level audit, controlling for the timestep and spatial-position confounds that diffusion routers are known to be dominated by. Measure I(expert; domain | t, p).
- **RQ2 (the fix).** If specialization is weak — which EMO predicts for any non-purpose-built MoE — can a **router-only fine-tune** with a specialization objective induce it *post hoc*, without EMO's requirement of pretraining from scratch? EMO's fix costs 1T tokens of pretraining. Ours should cost a few GPU-days. That contrast is the contribution.
- **RQ3 (the payoff).** At matched memory budgets, how does domain-localized expert pruning compare to uniform expert pruning, to INT4/INT8 quantization, and to both combined? Report a memory-vs-FID Pareto frontier.

Each RQ is independently publishable. RQ1 alone is a solid short paper. That's your insurance.

## 4. Why diffusion is the right testbed, not just an unclaimed one

The confound structure is genuinely different, and that's a real scientific reason rather than a land-grab. [DiT-MoE](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) reported that expert selection in diffusion transformers is sensitive to spatial position and denoising timestep but **insensitive to class-conditional information** — a striking claim that nobody has since revisited with the analytical machinery the 2026 language papers developed. [ProMoE](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) (2025) independently argues visual tokens resist expert specialization because of spatial redundancy and functional heterogeneity, and that explicit routing guidance is *required*. So diffusion is the setting where the "illusion of specialization" hypothesis should be **strongest** — which makes it the sharpest test.

## 5. Method

**Stage 0 — The audit (answers RQ1).**
Instrument a pretrained DiT-MoE with forward hooks on every router. For each (domain d, timestep t, spatial position p), log gate scores and top-k selections. Then:
- Compute conditional mutual information I(expert; domain | t, p) per layer. This is the number the whole paper hangs on.
- Run a group-level Standing Committee analysis (per Wang et al. 2026) to test whether a domain-invariant coalition exists in diffusion too.
- Report specialization as a function of depth, since ViMoE and DiT-MoE both find it varies by layer.

**Stage 1 — Router-only domain fine-tuning (answers RQ2).**
Freeze every expert weight and the backbone. Train *only* the router projections on 5–20k in-domain images, with:
- the standard diffusion loss,
- a **routing concentration loss** — entropy penalty on the domain-marginal expert distribution, pushing mass onto few experts,
- a **timestep-consistency loss** — penalizes churn in the retained set across t, so that one *static* subset suffices for the whole denoising trajectory.

The second term is diffusion-specific and has no analogue in the language work. It exists precisely because timestep is the dominant routing axis. That is the technically novel piece.

**Stage 2 — Static extraction.** Rank experts by domain-conditioned activation mass × output L2 norm (EASY-EP's criterion, adapted to visual tokens). Keep top-k, **physically delete** the rest, re-index routers. Output is a genuinely smaller checkpoint on disk — the thing you originally wanted, and strictly better than a masked model.

**Stage 3 — Recovery.** Short LoRA or full fine-tune of surviving experts to claw back FID.

## 6. Experiments

**Backbones.** `feizc/DiT-MoE` (DiT-MoE-S/2 and B/2, ImageNet-256, 8 experts/layer, released checkpoints) as primary. `kunncheng/Diff-MoE` (ICML 2025) as the generality check. These are the only open MoE-diffusion checkpoints at a scale you can actually train.

**Domains.** ImageNet-256 superclasses (animals / vehicles / food / structures) — free labels, clean partition. Plus one out-of-distribution domain (FFHQ or a style set) for transfer.

**Baselines.** Random expert pruning · frequency-only pruning without router FT · Diff-Pruning (Fang et al.) · INT8 and INT4 weight quantization · ours · ours + quantization.

**Metrics.** FID-50k, IS or CLIP score, precision/recall, **peak GPU memory**, latency, on-disk parameter count. The memory-vs-FID Pareto plot is the figure reviewers will judge you on.

**Ablations.** k sweep · each loss term ablated · **router-FT vs. no router-FT** (this is the money ablation — it's the direct test of EMO's claim in the diffusion setting) · per-layer vs. global k · domain sample size at 20 / 200 / 5k images.

## 7. Compute — the honest number

**Verdict: Colab Pro+ (~$50/month) is the minimum that works. Free tier will not get you to a mid-tier submission.**

| Stage | Requirement |
|---|---|
| Stage 0 audit | Inference only. Free T4 is fine. ~10 GPU-hours. |
| Stage 1 router FT | Routers are <1% of params but need full forward/backward. A100-40GB, ~20–40 h per domain. |
| Stage 2–3 recovery | ~30–60 h per configuration. The main cost. |
| FID-50k evaluation | ~2–4 h per config × ~20 configs ≈ 60 h. People routinely underestimate this. |

**Total ≈ 300–500 A100-hours.** Colab Pro+ covers it over 2–3 months. If you want to compress to 4–6 weeks, budget ~$300 of RunPod/Lambda A100 time on top.

Do **not** attempt FLUX, HunyuanImage-3.0, or Nucleus-Image (17B). Out of reach and unnecessary — DiT-MoE-S/B on ImageNet-256 is an accepted scale for this claim.

## 8. Risks

| Risk | Mitigation |
|---|---|
| RQ1 finds domains genuinely inseparable even after controlling confounds | That *is* the result — "the Standing Committee generalizes to vision" is a publishable negative finding, and RQ3 still stands alone |
| Router-only FT too weak to induce specialization | Escalate to router + LoRA on experts; report router-only as a negative ablation, which is itself evidence for EMO's pretraining claim |
| Someone publishes the diffusion audit first | Highest-probability risk. Stage 0 is only ~4 weeks — get the audit onto arXiv as a standalone note before starting Stage 1 |
| FID noisy at small scale | FID-50k, fixed seeds, ±std over 3 runs, plus precision/recall |
| DiT-MoE checkpoints don't load cleanly | Fall back to Diff-MoE; budget one week for infrastructure pain |

## 9. Timeline (Colab Pro+, part-time)

- **Weeks 1–2.** Load DiT-MoE checkpoints, reproduce reported FID. Hard sanity gate — do not proceed until this passes.
- **Weeks 3–5.** Stage 0 audit. Produce the I(expert; domain | t, p) heatmaps and the Standing Committee test. **Decision point.**
- **Week 6.** Post the audit to arXiv as a standalone note. Stakes the claim, costs a week, and de-risks being scooped.
- **Weeks 7–10.** Stage 1 router fine-tuning.
- **Weeks 11–14.** Extraction, recovery, full baseline sweep.
- **Weeks 15–17.** Ablations, figures, writing.
- **Week 18.** Submit.

## 10. Immediate next actions

1. Clone `feizc/DiT-MoE`; confirm checkpoints download and sample correctly on a free T4 before spending any money.
2. Read Do et al. 2026 and Wang et al. 2026 in full — their exact audit protocols are what you will port to diffusion, and matching their methodology is what makes the comparison legitimate.
3. Read the DiT-MoE expert-specialization section closely; the precise experiment behind "insensitive to class-conditional information" determines how strongly you can claim they didn't control for confounds.
4. Write the router telemetry hook.
5. Pick a venue and check its next deadline to anchor the timeline.

---

## References (2025–2026 unless foundational)

**The controversy this paper enters**
- [Do Domain-specific Experts exist in MoE-based LLMs?](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) — Do et al., 2026
- [The Illusion of Specialization: Unveiling the Domain-Invariant "Standing Committee" in MoE Models](https://consensus.app/papers/details/4c2a11747360510db1349851fa791f16/) — Wang et al., 2026
- [EMO: Pretraining Mixture of Experts for Emergent Modularity](https://consensus.app/papers/details/0f6963a6f9105932b765ed00dfd6ab1c/) — Wang et al., 2026
- [MoE Lens — An Expert Is All You Need](https://consensus.app/papers/details/03d7d6b635d158dfaf16578db5b8a55a/) — Chaudhari et al., 2026

**Diffusion MoE**
- [Routing Matters in MoE (ProMoE)](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) — Wei et al., 2025
- [Focusing on What Matters: Saliency-Harnessing Accurate Routing for Diffusion MoE (SharpMoE)](https://consensus.app/papers/details/dbd920a3439c5825b6c78aa73f87c7c2/) — Deng et al., 2026
- [Dense2MoE](https://consensus.app/papers/details/d4719f1be64f5755be0a9d6296f09539/) — Zheng et al., ICCV 2025
- [Efficient Training of Diffusion MoE Models: A Practical Recipe](https://consensus.app/papers/details/97271df25aca5a789f0fbc5706b5ec0b/) — Liu et al., 2025
- [Scaling Diffusion Transformers to 16B Parameters (DiT-MoE)](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) — Fei et al., 2024 *(foundational; the claim we test)*

**Diffusion compression baselines**
- [ALTER](https://consensus.app/papers/details/70b9be1944e252b38596709c856d87af/) — Yang et al., 2025
- [OBS-Diff](https://consensus.app/papers/details/2fda0488ae46584bae34a6ce0c24276e/) — Zhu et al., 2025
- [HierarchicalPrune](https://consensus.app/papers/details/10c78e7400d457229c0475fe0b573143/) — Kwon et al., 2025
- [Diff-ES](https://consensus.app/papers/details/dd2ed4f22a7f55a38c09fd371e4b0ffe/) — Liu et al., 2026
- [TMP: Tree-structured Mixed-policy Pruning](https://consensus.app/papers/details/aac4443e27cd5c6897fdb2cb66b3425f/) — Zhang et al., 2026

**LLM-side saturation — the field to avoid, cited for positioning**
- [EASY-EP](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) — Dong et al., 2025 *(the method we port)*
- [Less is MoE: Trimming Experts in Domain-Specialist LMs](https://arxiv.org/abs/2606.05538) — 2026
- [Domain-Specific Expert Pruning for MoE LLMs](https://proceedings.mlr.press/v317/yao26a.html) — Yao et al., 2026
- [REAP the Experts](https://arxiv.org/html/2510.13999v3) · [AIMER](https://arxiv.org/pdf/2603.18492) · [ConMoE](https://arxiv.org/pdf/2605.29350) · [MoE Pathfinder](https://arxiv.org/pdf/2512.18425) · [FlexMoE](https://arxiv.org/pdf/2606.27866)
- [On the Utility and Factual Reliability of Pruned MoE Models in the Biomedical Domain](https://consensus.app/papers/details/7e11cf0856fb5d0193b97397ead7d7d5/) — Yamaguchi et al., 2026

**Code**
- [feizc/DiT-MoE](https://github.com/feizc/DiT-MoE) · [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE) · [ali-vilab/ProMoE](https://github.com/ali-vilab/ProMoE)
