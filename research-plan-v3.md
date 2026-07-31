# Research Plan v3 — Post-Hoc Expert Modularity in Diffusion Mixture-of-Experts

**Author:** Sarim · **Date:** 31 July 2026
**Target:** Journal — Pattern Recognition, Neurocomputing, or IEEE TIP
**Supersedes:** v1 (framing), v2 (venue + contribution ordering + routing-variance RQ)

---

## 1. Working title

*Modularity Without Pretraining: Post-Hoc Router Adaptation for Domain-Specialized Diffusion Mixture-of-Experts*

## 2. The paper in one paragraph

MoE diffusion transformers carry every expert in memory even though any given generation task uses few of them. Porting the LLM recipe — profile a domain, keep its experts, drop the rest — fails, because diffusion routers key on denoising timestep and spatial position rather than semantic content. EMO (2026) showed the fix in language costs a full pretraining run. We show that in diffusion it does not: a **router-only fine-tune**, with expert weights frozen, induces domain-localized routing post hoc in a few GPU-days, after which unused experts can be physically deleted. We support this with the first audit of domain specialization in diffusion MoE, and evaluate against expert-pruning and quantization baselines on a memory-versus-FID Pareto frontier.

## 3. Contributions, in the order the paper should claim them

- **C1 — Method (headline).** Post-hoc router-only adaptation with two losses: a *routing concentration* loss and a *timestep-consistency* loss. The latter is diffusion-specific and has no analogue in the language literature, because timestep is not a routing axis there. Delivers EMO-style modularity without EMO's from-scratch pretraining.
- **C2 — Artifact.** An extraction procedure yielding a genuinely smaller on-disk checkpoint, not a masked model.
- **C3 — Analysis.** First audit of whether domain-specific experts exist in diffusion MoE, porting the 2026 language controversy to vision and controlling for the timestep/position confounds.
- **C4 — Empirical.** Memory-vs-FID Pareto against uniform pruning and INT4/INT8 quantization, including the combination.

C3 is placed third deliberately. It motivates and justifies C1 rather than standing alone — but in a journal it can occupy real space, which is why journal beats conference here.

## 4. Research questions

- **RQ1.** Do domain-specific experts exist in diffusion MoE, once timestep and spatial position are controlled for? Measure I(expert; domain | t, p); run a Standing-Committee group-level audit.
- **RQ2.** Does sparse routing *add* output variance? For a fixed prompt across seeds, does a dense model's fixed computation path get replaced by a seed-dependent one? **Direction of effect is an open empirical question — see §6.**
- **RQ3.** Can router-only fine-tuning induce domain-localized routing post hoc, at a small fraction of EMO's cost?
- **RQ4.** At matched memory budgets, how does domain-localized expert pruning compare to uniform pruning, quantization, and both?

## 5. Positioning — why each neighbor doesn't cover us

| Work | What it does | Why we're different |
|---|---|---|
| EMO (2026) | Domain-modular MoE via pretraining, 1T tokens | We do it post-hoc on an existing checkpoint. This is the核 contrast. |
| EASY-EP (2025) | Domain expert pruning, training-free, LLM | Training-free fails in diffusion; we show why and train routers |
| ProMoE (2025) | Routing guidance, trained from scratch, for quality | We compress an existing model; different goal |
| ALTER (2025), DiffPruning (2024) | Timestep as the expert axis | Domain axis; composable with theirs |
| DiT-MoE (2024) | Reports experts insensitive to class conditioning | The claim we test with proper confound control |
| SharpMoE (2026) | Routers corrupted by noisy latents; post-training fix | Supplies our mechanism for RQ2; they target quality, we target modularity |

## 6. The RQ2 caveat — do not assume the sign

The intuition that "stabler routing → less random, better output" is **already falsified in a neighboring setting**. [Expert-Data Alignment Governs Generation Quality in Decentralized Diffusion Models](https://consensus.app/papers/details/c82f0b8290055b69b3299cfdfd31a7f7/) (2026) reports a *stability–quality dissociation*: full ensemble routing gives the most stable sampling dynamics and the **worst** FID (47.9 vs 22.6 for sparse Top-2). Their governing principle is expert–data alignment, not stability.

Evidence points both ways, which is what makes measuring it worthwhile:

- **Routing is genuinely unstable.** [R3](https://consensus.app/papers/details/bda4a1db02db504caabb6ef4bf96d9fd/) (2025): "even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes." [VSRAQ](https://consensus.app/papers/details/b0ab519ff08e558db235fbdf8b958de5/) (2026): tiny perturbations flip top-k and alter the computation path. [VMoER](https://consensus.app/papers/details/ba46ec0ce97d5a14ad23842cfda5ff5f/) (2026) treats "routing stability under noise" as a headline metric.
- **But MoE may reduce sensitivity.** [MoE as Soft Clustering](https://consensus.app/papers/details/b66a1032cdd950e19f54d0e3cce9358d/) (2026): MoE routing yields smaller Jacobian singular values and flatter local curvature than dense.

**Framing rule for the writeup:** present RQ2 as a measurement, never as a predicted direction. A result that contradicts the hypothesis is publishable; a hypothesis asserted and then caught by a reviewer who knows the 2026 paper is not.

## 7. Method

**Stage 0 — Audit.** Forward hooks on every router. Log gate scores and top-k selections indexed by (domain, timestep, spatial position, seed). Compute I(expert; domain | t, p) per layer; run the Standing Committee group test; measure cross-seed routing agreement for fixed prompts as a function of timestep. Inference only.

**Stage 1 — Router-only fine-tuning.** Freeze all expert weights and the backbone. Train only router projections on 5–20k in-domain images:
- standard diffusion loss
- **routing concentration loss** — entropy penalty on the domain-marginal expert distribution
- **timestep-consistency loss** — penalizes churn in the retained set across t, so one static subset serves the whole trajectory

**Stage 2 — Static extraction.** Rank by domain-conditioned activation mass × output L2 norm. Keep top-k, physically delete the rest, re-index routers.

**Stage 3 — Recovery.** Short LoRA or full fine-tune of surviving experts.

## 8. Experiments

**Backbones.** `feizc/DiT-MoE` (S/2 and B/2, ImageNet-256, 8 experts/layer, released checkpoints) primary; `kunncheng/Diff-MoE` (ICML'25) for generality.

**Domains.** ImageNet-256 superclasses (animals / vehicles / food / structures) plus one OOD domain (FFHQ or a style set).

**Baselines.** Random pruning · frequency-only pruning without router FT · Diff-Pruning · INT8 · INT4 · ours · ours + quantization.

**Metrics.** FID-50k, IS/CLIP, precision–recall, peak GPU memory, latency, on-disk parameters.

**Ablations.** k sweep · each loss term removed · **router-FT vs. none** (the money ablation; direct test of EMO's claim in diffusion) · per-layer vs. global k · domain sample size 20 / 200 / 5k.

## 9. Compute

**Colab Pro+ (~$50/mo) minimum. Total ≈ 300–500 A100-hours.**

| Stage | Cost |
|---|---|
| Stage 0 audit | ~10 h, free T4 sufficient |
| Stage 1 router FT | 20–40 h per domain, A100-40GB |
| Stage 2–3 recovery | 30–60 h per config |
| FID-50k eval | ~2–4 h × ~20 configs ≈ 60 h (routinely underestimated) |

Add ~$300 RunPod/Lambda to compress the calendar. Do not attempt FLUX, HunyuanImage-3.0, or Nucleus-Image (17B) — out of reach and unnecessary.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Router-only FT too weak | Escalate to router + expert LoRA; report router-only as a negative ablation, itself evidence for EMO |
| RQ1 finds domains inseparable | "The Standing Committee generalizes to vision" is publishable; C1 method still stands on the induced-modularity claim |
| RQ2 contradicts the intuition | Fine — §6 framing rule handles it |
| Scooped on the audit | Stage 0 is ~4 weeks; post it to arXiv standalone at week 6 |
| FID noise at small scale | FID-50k, fixed seeds, ±std over 3 runs, plus precision/recall |
| Checkpoints don't load | Fall back to Diff-MoE; budget one week |

## 11. Timeline (journal — no hard deadline, ~5 months part-time)

- **Wk 1–2.** Load DiT-MoE, reproduce reported FID. Hard gate.
- **Wk 3–5.** Stage 0 audit (RQ1 + RQ2). Decision point.
- **Wk 6.** arXiv the audit standalone — stakes the claim, de-risks scooping.
- **Wk 7–11.** Stage 1 router fine-tuning. The core method work.
- **Wk 12–16.** Extraction, recovery, baseline sweep.
- **Wk 17–20.** Ablations, figures, writing.
- **Wk 21.** Submit.

## 12. Next actions

1. Clone `feizc/DiT-MoE`; verify checkpoints sample correctly on a free T4 **before spending money**.
2. Read Do et al. 2026 and Wang et al. 2026 in full — port their audit protocols so the comparison is legitimate.
3. Read EMO (2026) closely; the cost contrast is your headline and you need their numbers exact.
4. Read the DiT-MoE expert-specialization section; its precise protocol determines how strongly you can claim uncontrolled confounds.
5. Write the router telemetry hook.
6. Pick the target journal and read two recent MoE/efficiency papers from it to calibrate scope.

---

## Key references

**The controversy**
- [Do Domain-specific Experts exist in MoE-based LLMs?](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) — Do et al., 2026
- [The Illusion of Specialization](https://consensus.app/papers/details/4c2a11747360510db1349851fa791f16/) — Wang et al., 2026
- [EMO: Pretraining MoE for Emergent Modularity](https://consensus.app/papers/details/0f6963a6f9105932b765ed00dfd6ab1c/) — Wang et al., 2026
- [MoE Lens](https://consensus.app/papers/details/03d7d6b635d158dfaf16578db5b8a55a/) — Chaudhari et al., 2026

**Routing stability / variance (RQ2)**
- [Stabilizing MoE RL by Aligning Training and Inference Routers (R3)](https://consensus.app/papers/details/bda4a1db02db504caabb6ef4bf96d9fd/) — Ma et al., 2025
- [VSRAQ](https://consensus.app/papers/details/b0ab519ff08e558db235fbdf8b958de5/) — Park et al., 2026
- [Variational Routing (VMoER)](https://consensus.app/papers/details/ba46ec0ce97d5a14ad23842cfda5ff5f/) — Li et al., 2026
- [Expert-Data Alignment Governs Generation Quality in DDMs](https://consensus.app/papers/details/c82f0b8290055b69b3299cfdfd31a7f7/) — Villagra et al., 2026 **(the caveat)**
- [MoE as Soft Clustering](https://consensus.app/papers/details/b66a1032cdd950e19f54d0e3cce9358d/) — Liu, 2026

**Diffusion MoE**
- [DiT-MoE](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) — Fei et al., 2024
- [ProMoE](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) — Wei et al., 2025
- [SharpMoE](https://consensus.app/papers/details/dbd920a3439c5825b6c78aa73f87c7c2/) — Deng et al., 2026
- [Dense2MoE](https://consensus.app/papers/details/d4719f1be64f5755be0a9d6296f09539/) — Zheng et al., ICCV 2025
- [Sparse MoE Routing in Visual DiTs: Routing Collapse to Selective Deadlock](https://consensus.app/papers/details/0450aefbf2105cf496d413c9cb40606b/) — Sha, 2026

**Compression baselines**
- [EASY-EP](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) · [ALTER](https://consensus.app/papers/details/70b9be1944e252b38596709c856d87af/) · [OBS-Diff](https://consensus.app/papers/details/2fda0488ae46584bae34a6ce0c24276e/) · [HierarchicalPrune](https://consensus.app/papers/details/10c78e7400d457229c0475fe0b573143/) · [DiffPruning](https://consensus.app/papers/details/e34f2879033a544abb3ba6065654f23a/)

**Code**
- [feizc/DiT-MoE](https://github.com/feizc/DiT-MoE) · [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE) · [ali-vilab/ProMoE](https://github.com/ali-vilab/ProMoE)
