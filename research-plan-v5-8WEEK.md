# Research Plan v5 — TAES, 8-Week Sprint Edition

**Author:** Sarim · **Date:** 31 July 2026
**Target:** arXiv at week 7, journal submission week 8 (Pattern Recognition / Neurocomputing / IEEE TIP)
**Budget:** ₹0 — Kaggle free tier only
**Supersedes:** v1–v4. Those can be deleted; this is the plan.

> **Amended 31 July 2026 after Phase 0.** Backbone changed from DiT-MoE to **DSMoE-S-E48 (N=48, top-k=5)**. DiT-MoE's advertised N=16 small checkpoint does not exist; all usable releases are N=8. Week 0's precondition gate fired and was resolved the same day. Consequences: **memory mode is the primary claim**, the sampler is rectified flow not DDIM, and the k-sweep must satisfy k ≥ 5. Evidence and reasoning: `taes/notes/backbone-survey.md`; decision entry: `taes/notes/results-log.md`.

---

## 1. The paper, unchanged

*TAES: Training-Free Timestep-Aware Expert Steering for Memory-Efficient Diffusion Mixture-of-Experts*

Every training-free MoE steering method (RICE, MoTE, Ban&Pick, DERN, DSMoE, MASCing, SSMoE) scores experts by token-level semantics. Diffusion has none. What it has is the denoising timestep — an axis absent from language, and the one DiT-MoE reports actually dominates routing. **TAES scores expert importance over (domain, timestep band) instead of domain alone**, then either deletes the unused experts (smaller checkpoint) or masks them per band (less compute).

The headline experiment is banded vs. global scoring, where global is the naive port of the LLM recipe. That single comparison is the contribution, and it is also the cheapest experiment in the plan.

## 2. What I cut to reach 8 weeks, and what it costs you

| Cut | Was | Now | Cost to the paper |
|---|---|---|---|
| Domains | 4 superclasses | **2** (animals, vehicles) | Weaker generality claim. Acceptable if scoped honestly. |
| Quantization baselines | INT8 + INT4 + combined | **dropped** | Loses the pruning-vs-quantization Pareto. This was a "nice to have", not the claim. |
| Cross-domain transfer | full ablation | **dropped** | Reviewers may ask. Answer: future work. |
| ~~Compute mode~~ | both modes | **RESTORED — see Decision 8** | Cutting it was an error. Memory mode suffers union saturation at low expert counts; compute mode is immune. Which one leads is decided by the day-1 expert count. |
| Second backbone | DiT-MoE + Diff-MoE | **DSMoE-S-E48 only** (DSMoE-B-E48 if quota allows) | Single-backbone results. State the limitation explicitly. |
| FID protocol | 50k everywhere | **10k for sweeps, 50k for 4 final configs** | Standard practice; not a real weakness. |
| Writing | after experiments | **in parallel from week 3** | Non-negotiable for this timeline. |

**What survives is the full core claim.** Nothing cut touches whether banding beats global.

## 3. Experiments — the minimum viable set

**Backbone.** `YHLLEO/DSMoE-S-E48`, ImageNet-256 latents, **48 routed experts/layer, top-k=5**, MoE in 6 of 12 blocks (`interleave: true`), rectified flow, 1.11 GB. Shared expert present and **not prunable**.
**Domains.** 2 ImageNet superclasses.

**Configs (~18 total):**
1. No pruning (upper bound)
2. Random expert removal
3. Frequency-only ranking
4. **Global importance — the naive LLM port**
5. **TAES (banded)**
- × 2 domains, × 3 values of k for configs 4–5
- Band ablation B ∈ {1, 2, 4, 8}, where **B=1 is exactly config 4** — so this ablation doubles as the core result

**k sweep: k ∈ {5, 8, 12, 16, 24, 32, 48}** (of N=48). **k ≥ 5 is a hard floor** — the router's own top-k is 5, so below that it cannot fill its selection. k=24 is the N/2 headline configuration.

**Metrics.** FID-10k (sweeps), FID-50k (4 final configs), precision/recall, peak GPU memory, on-disk parameters, latency.

## 4. Compute budget

| Item | P100-hours |
|---|---|
| Calibration, 2 domains | 6 |
| FID-10k × 18 configs @ 25 RF steps | 45 |
| Band ablation | 15 |
| FID-50k × 4 final configs | 40 |
| Reruns, seeds, mistakes | 30 |
| **Total** | **~136** |

Kaggle gives 30 h/week → **240 hours over 8 weeks**. That is ~100 hours of genuine slack. Compute is not your problem.

## 5. The 8-week schedule

| Week | Work | Gate |
|---|---|---|
| **0** (day 1) | ~~Count experts per layer in available DiT-MoE checkpoints.~~ ✅ **DONE 31 Jul.** DiT-MoE failed at N=8; switched to **DSMoE-S-E48, N=48, top-k=5**. Repo skeleton built. | ✅ **GATE PASSED.** N=48 ≥ 32 → memory mode primary. See `taes/notes/backbone-survey.md`. |
| **1** | Kaggle env, clone **EfficientMoE** (+ DiffMoE deps), load `checkpoints/0700000.pt`, sample images, reproduce a reference FID | **HARD GATE — if this slips past week 2, the 8-week plan is dead. See §6.** |
| **2** | Router telemetry hooks. Cache VAE latents for both domains. Start FID-10k on no-prune baseline. | Telemetry producing clean logs |
| **3** | Calibration runs. Banded importance scoring. Produce importance-vs-timestep heatmaps. **Start writing Methods.** | |
| **4** | **GO/NO-GO: does banded importance differ measurably from global?** | If no → pivot to negative-result paper, same data, §7 |
| **5** | Extraction + memory mode. Baselines 1–4. FID-10k sweeps running continuously. **Write Related Work.** | |
| **6** | Band ablation B ∈ {1,2,4,8}. k sweep. **Write Experiments.** | |
| **7** | FID-50k on 4 final configs. Figures. **Write Intro + Discussion. Post to arXiv.** | Preprint live |
| **8** | Polish, limitations section, format for target journal, submit. | Submitted |

**Rule: never leave the GPU idle.** Queue the next FID run before you start writing. Kaggle quota is use-it-or-lose-it weekly.

## 6. The one risk that actually matters

**Week 1 setup.** Everything else has slack; this has none.

**This risk went up after the backbone switch.** EfficientMoE is a December-2025 codebase; it depends on [DiffMoE](https://github.com/KlingTeam/DiffMoE)'s repo structure, pins `torch==2.6.0` / `tensorflow==2.15.0`, and needs a git-installed `torch-fidelity` fork. Less battle-tested than DiT-MoE. Get the environment pinned in a Kaggle Dataset the moment it works, so a dead session does not cost you the install.

Mitigation, in order:
1. Try `DSMoE-S-E48` first. Budget **3 days maximum.**
2. If it will not load, switch to `DSMoE-B-E48` — same N=48, same recipe, larger. Do not debug for a week out of stubbornness.
3. If both fail by end of week 1, **train your own tiny MoE-DiT on CIFAR-10 or CelebA-64 with N=32**. Roughly 15 GPU-hours, gives full control of N, and at that scale the timestep-banding claim is still testable. The paper gets weaker but stays alive.

~~`kunncheng/Diff-MoE` as fallback~~ — **removed.** It has no released checkpoints. It stays in the plan as the nearest prior work for Related Work only.

Decide by **end of week 1**, not week 3. The failure mode that kills sprints is refusing to switch.

## 7. If the go/no-go fails at week 4

If banded importance turns out indistinguishable from global, you have a real finding: *expert importance in diffusion MoE is timestep-invariant, contradicting the specialization structure DiT-MoE reports.* That contradicts a published claim, it is cheap to verify, and it is publishable as a short empirical paper. Same data, same schedule, different framing. **Do not treat this as failure.**

## 8. Honest expectations

- **Week 7:** arXiv preprint live. Citable immediately.
- **Week 8:** journal submitted.
- **+6–12 months:** review outcome. Nothing compresses this — plan around the preprint, not the acceptance.

If you need a line on a CV before then, the arXiv preprint at week 7 is the deliverable that matters.

## 9. Start here, today

1. Open a Kaggle notebook, GPU on. Clone `yhlleo/EfficientMoE` and its DiffMoE dependency. Download `YHLLEO/DSMoE-S-E48` → `checkpoints/0700000.pt`. Get one image out of it. **Nothing else matters until this works.**
2. Confirm which 6 of 12 blocks carry MoE (`interleave: true`) from `models/models_DSMoE.py`, before writing any hook.
3. Install TooManyPapers, load the reference list.
4. Read DSMoE **(Do et al., the LLM paper)** and Ban&Pick scoring functions — you are adapting them and need the exact formulations. ⚠️ Do not confuse this with DSMoE **(Liu et al.)**, the diffusion backbone.
5. Read the DiT-MoE expert-specialization section — it is the empirical basis for banding, even though DiT-MoE is no longer the backbone.
6. Read Diff-MoE (ICML 2025) figures — check whether they already show band-wise expert divergence.

---

## References

**Training-free MoE steering — the family being ported**
- [RICE](https://consensus.app/papers/details/3132c0b6b6a85fcb95356dfb23793c8a/) · [MoTE](https://consensus.app/papers/details/b8f295008666522cb54665e5aabe1dde/) · [Ban&Pick](https://consensus.app/papers/details/d0e3846be58752f4b9b739e828da7dc4/) · [DERN](https://consensus.app/papers/details/d2c5c8e4b7e15bbca23316661376ef75/) · [DSMoE](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) · [MASCing](https://consensus.app/papers/details/8836fce80a9453c8bfdcfa23ebc8b0e1/) · [SSMoE](https://consensus.app/papers/details/ab80d6eef03a5246b37875a3cfdb559d/) · [EASY-EP](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/)

**Diffusion MoE — the target**
- [DiT-MoE](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) · [SharpMoE](https://consensus.app/papers/details/dbd920a3439c5825b6c78aa73f87c7c2/) · [ProMoE](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/) · [ALTER](https://consensus.app/papers/details/70b9be1944e252b38596709c856d87af/)

**Context**
- [EMO](https://consensus.app/papers/details/0f6963a6f9105932b765ed00dfd6ab1c/) · [The Illusion of Specialization](https://consensus.app/papers/details/4c2a11747360510db1349851fa791f16/) · [Expert-Data Alignment Governs Generation Quality in DDMs](https://consensus.app/papers/details/c82f0b8290055b69b3299cfdfd31a7f7/)

**Backbone (new)**
- [yhlleo/EfficientMoE](https://github.com/yhlleo/EfficientMoE) · [arXiv 2512.01252](https://arxiv.org/abs/2512.01252) · [YHLLEO/DSMoE-S-E48](https://huggingface.co/YHLLEO/DSMoE-S-E48) · [KlingTeam/DiffMoE](https://github.com/KlingTeam/DiffMoE)

**Code**
- [feizc/DiT-MoE](https://github.com/feizc/DiT-MoE) · [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE) · [giangdip2410/Domain-specific-Experts](https://github.com/giangdip2410/Domain-specific-Experts)
