# TAES — Master Reference Guide

**Project:** Training-Free Timestep-Aware Expert Steering for Diffusion Mixture-of-Experts
**Author:** Sarim · **Started:** 31 July 2026 · **Target submission:** week 8 (~late September 2026)
**Companion document:** `research-plan-v5-8WEEK.md` (the schedule). This document is the *why*.

---

## 0. How to use this document

Read §1 and §2 before every work session. They are the anti-drift mechanism.

When you feel the pull to add "just one more experiment" or chase a new idea from a paper you just read, go to **§8, the Decision Log**. Most temptations have already been considered and rejected for a recorded reason. If your new idea is not in the log, add it there with a decision — do not silently act on it.

---

## 1. The north star

> **We take the family of training-free expert-steering methods that exists for MoE language models, and port it to MoE diffusion models. The port is not mechanical: language methods score experts using token semantics, which diffusion does not have. Diffusion instead has the denoising timestep, an axis absent from language and the one that empirically dominates diffusion routing. TAES therefore scores expert importance per *timestep band* rather than once globally, and uses that score to drop experts that a given band never needs — with no training, on free compute.**

One sentence for a conversation: *"Diffusion MoE models carry every expert even though the denoising process uses different ones at different stages. We find which experts matter at which stage, and drop the rest without any retraining."*

**Axis priority — do not invert this.** The **timestep band** is the primary axis and the novel contribution. **Domain** is a secondary conditioning variable, not the headline. Domain-conditioned expert pruning is EASY-EP's contribution, already published, in language. If domain leads, the paper reads as a port and invites the "isn't this just EASY-EP?" objection. If timestep leads, the paper has an axis that does not exist in the prior work at all. See Decision 6 and Decision 9.

## 2. Scope guardrails — what this paper is NOT

Violating any of these breaks the 8-week timeline. They are not preferences.

| We are NOT | Why not |
|---|---|
| Inventing a new MoE architecture | Dense2MoE, Race-DiT, EC-DiT, ProMoE already did. Not our contribution. |
| Training or fine-tuning anything | Training-free is what makes this affordable and fast. The moment gradients enter, the budget dies. |
| Working on LLMs | Saturated. See §8, Decision 1. |
| Chasing text-to-image at scale (FLUX, HunyuanImage) | Needs compute we do not have. DSMoE-S-E48 on ImageNet-256 is the right scale. |
| Claiming reduced output randomness | Mechanically unsupported, and already contradicted in a neighbouring setting. See §8, Decision 5. |
| Proving generality across many models/datasets | One backbone, two domains, scoped honestly. Overreach costs weeks and invites rejection. |
| Beating state-of-the-art image quality | We are a compression method. Our win condition is the Pareto curve, not absolute FID. |

**The single question this paper answers:** *Does making expert-importance scoring timestep-aware beat scoring it globally, when pruning experts in a diffusion MoE?*

Everything that does not serve that question is out of scope.

---

## 3. Background — what you need to understand

### 3.1 Mixture-of-Experts, briefly

A standard transformer block has one feed-forward network (FFN). An MoE block replaces it with *N* parallel FFNs called **experts**, plus a small **router** (usually a single linear layer) that scores each token against each expert and sends the token to its **top-k** experts.

The payoff is that compute scales with *k*, not *N* — you get a big model's capacity at a small model's compute cost.

The catch, and our entire motivation: **all N experts must sit in memory**, because you cannot predict which the router will pick. So a model that computes like a 2B model still occupies the RAM of a 17B one. Sparse in compute, dense in memory.

### 3.2 Diffusion models, briefly

A diffusion model generates by starting from pure noise and denoising over T steps. Each step calls the network once with the current noisy latent and the **timestep** t, which tells it how much noise remains.

Two facts matter for us:

1. **The network is called repeatedly with different t.** The task at t=999 (near-pure noise, deciding coarse layout) is genuinely different from the task at t=10 (near-clean, refining texture). Same weights, different job.
2. **The timestep has no analogue in language models.** An LLM's forward pass has no equivalent axis. This is precisely the gap our method exploits.

### 3.3 The intersection: MoE diffusion transformers

A Diffusion Transformer (DiT) with MoE layers. `DiT-MoE` (Fei et al., 2024) is the canonical open one and the source of the empirical claim we test. **Our backbone is `DSMoE-S-E48`** (Liu et al., 2025, *EfficientMoE*) — same architecture family, DeepSeek-style routed experts, but N=48 rather than DiT-MoE's usable N=8. See `taes/notes/backbone-survey.md` for why DiT-MoE could not be used.

**The critical prior finding**, from the DiT-MoE paper itself:

> "Expert selection shows preference with spatial position and denoising time step, while insensitive with different class-conditional information... Expert specialization tends to be more concentrated at the early time step and then gradually uniform after half."

Read that twice. It says routing in diffusion is driven by *where in the image* and *when in the denoising*, and **not** by *what the image is of*. That is the opposite of language MoE, where routing is driven by semantic content.

This single quote is the empirical foundation of our method **and** our biggest risk. See §7.

---

## 4. Why this is new

### 4.1 The family we are porting

Training-free expert steering is an established, active line of work. Every entry below manipulates which experts fire, at inference, with no gradient updates:

| Method | Year | What it does | Why it matters to us |
|---|---|---|---|
| [RICE](https://consensus.app/papers/details/3132c0b6b6a85fcb95356dfb23793c8a/) | 2025 | Uses nPMI to find "cognitive experts", reinforces them at inference | The identify-then-steer template |
| [MoTE](https://consensus.app/papers/details/b8f295008666522cb54665e5aabe1dde/) | 2025 | Localizes a behaviour to ~10 of 14,848 experts, switches them off | Proof that behaviour localizes to very few experts |
| [Ban&Pick](https://consensus.app/papers/details/d0e3846be58752f4b9b739e828da7dc4/) | 2025 | Reinforces key experts, prunes redundant ones, post-training | Closest scoring function to ours |
| [DERN](https://consensus.app/papers/details/d2c5c8e4b7e15bbca23316661376ef75/) | 2025 | Retraining-free expert pruning + neuron recombination | A baseline we adapt |
| [EASY-EP](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) | 2025 | Few-shot domain expert localization: gate score × output L2 norm | **Our scoring function comes from here** |
| [DSMoE](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) | 2026 | Training-free domain steering, zero added inference cost, beats SFT | Direct precedent for domain-conditioned steering |
| [MASCing](https://consensus.app/papers/details/8836fce80a9453c8bfdcfa23ebc8b0e1/) | 2026 | Steering masks applied to routing gates at inference | Mechanism for our compute mode |
| [SSMoE](https://consensus.app/papers/details/ab80d6eef03a5246b37875a3cfdb559d/) | 2026 | Training-free router from expert-weight eigenvectors | Shows the space is still opening up |

**Every single one is a language model. Not one has been applied to image diffusion.**

### 4.2 Why the port is non-trivial (this is the contribution)

A reviewer's first objection will be: *"This is just EASY-EP applied to a different model. Where is the novelty?"*

The answer:

**All of these methods score an expert once per domain.** They compute something like "how important is expert *e* for the math domain?" and get a single number. That works because in an LLM, a forward pass is a forward pass — there is no additional axis.

**In diffusion there is.** The same expert may be essential at t=900 and useless at t=100, because the denoising task itself changes. A single per-domain number averages that structure away. DiT-MoE's own analysis says specialization is "concentrated at the early time step and then gradually uniform after half" — so the averaging is not a minor loss, it destroys the dominant signal.

**TAES makes importance a function of (domain, timestep band).** The naive port is exactly our B=1 configuration, so the ablation over band count *is* the experiment. That is a clean, cheap, falsifiable claim.

### 4.3 Why this hasn't been done

Not because it is hard, but because the two literatures barely talk to each other. Training-free steering grew inside the LLM-efficiency community; diffusion MoE grew inside the vision-generation community. The 2026 papers in each cite almost none from the other.

**This means our window is short.** It is an obvious move once someone stands in both fields. This is why the plan puts an arXiv preprint at week 7 rather than polishing to week 12.

---

## 5. The experiment, in detail

### 5.1 Setup

- **Backbone:** **`YHLLEO/DSMoE-S-E48`** — **N = 48 routed experts, top-k = 5**, depth 12 (MoE interleaved into 6 blocks), hidden 384, patch 2, ImageNet-256 latents, 1.11 GB checkpoint (`checkpoints/0700000.pt`, 700K steps). Reported FID-50K 14.81 @ CFG 1.5 with 30M active params. *Resolved 2026-07-31: the N≥16 precondition of §7 Risk 1 is satisfied with margin. DiT-MoE was rejected — its advertised `S/2-16E2A` checkpoint does not exist and every usable release is N=8.*
- **Shared expert:** DSMoE has `use_shared_expert: true`. The shared expert is always applied and **is not routed**. Exclude it from N, from importance scoring, and from every memory-reduction figure.
- **Data:** ImageNet-256. Two superclass domains — **animals** and **vehicles** — built from ImageNet class labels. Labels are free; no annotation needed.
- **Sampler:** **rectified flow** (the checkpoint is trained with `rf: true`), 25 steps for sweeps, fixed seeds throughout. *Not DDIM — the v1–v4 plans assumed DiT-MoE's DDIM sampler.* Timesteps are continuous over `[0, 1000]`; band boundaries are defined on the RF schedule.
- **Latents:** precompute VAE latents once and cache to a Kaggle Dataset. Do not re-run the VAE per experiment.

### 5.2 Step 1 — Calibration (no gradients)

Take 200–500 in-domain images. For each, sample noisy latents across the full timestep range, run **one forward pass** each with hooks installed on every router.

Record, per (domain, timestep, layer, expert):
- gate score assigned by the router
- whether the expert was in the top-k
- L2 norm of the expert's output

Output: a tensor of routing telemetry. Cost: roughly 3 hours per domain on a P100.

### 5.3 Step 2 — Banded importance scoring

Partition the timestep range into **B bands** (default B=4, e.g. t∈[750,1000), [500,750), [250,500), [0,250)).

For each (domain *d*, band *b*, layer *l*, expert *e*):

```
I(e | d, b, l)  =  freq(e | d,b,l)  ×  mean_gate_score(e | d,b,l)  ×  mean_output_L2(e | d,b,l)
```

- `freq` — fraction of tokens in that band routed to *e*
- `mean_gate_score` — the router's confidence when it does pick *e*
- `mean_output_L2` — how much *e* actually contributes when active

The multiplicative form follows EASY-EP and Ban&Pick. **The banding by *b* is ours.**

*Note the failure mode this guards against:* frequency alone is misleading — an expert can fire often and contribute little. That is why the L2 term is there, and it is also why "frequency-only" is one of our baselines.

### 5.4 Step 3 — Deployment, and which mode is primary

Two modes exist. **Which one is the paper's headline depends on the week-1 expert count.** Read §7 Risk 1 before choosing.

**Compute mode (default primary).** At band *b*, mask the routing gates so only S_b is active. Weights stay on disk; per-step active compute drops. Composable with expert offloading. **This mode has no union-saturation problem** — divergent bands are pure upside, so it works at any N.

**Memory mode (primary only if N is large).** Take the union over bands of the top-k experts per layer, physically delete the remainder from the state dict, re-index the router's output projection. Output is a genuinely smaller checkpoint on disk, not a masked model.

**Decision rule:**
- **N ≥ 32** → memory mode primary, compute mode secondary. Both reportable.
- **16 ≤ N < 32** → run both, let the measured union sizes decide which leads.
- **N < 16** → compute mode primary. Report memory mode as a limitation with the saturation analysis as the explanation. Do not force it.

### 5.5 The configuration matrix

| # | Config | Purpose |
|---|---|---|
| 1 | No pruning | Upper bound on quality |
| 2 | Random expert removal | Sanity floor — anything must beat this |
| 3 | Frequency-only ranking | Tests whether the L2 and gate terms earn their place |
| 4 | **Global importance (B=1)** | **The naive LLM port. THE baseline that matters.** |
| 5 | **TAES, B=4** | **Ours** |

Run across 2 domains × 3 values of k (fraction of experts kept). Plus the band ablation B ∈ {1,2,4,8}, in which **B=1 is identical to config 4** — so that ablation and the headline result are the same experiment. Roughly 18 configurations total.

### 5.6 Metrics

- **FID** — Fréchet Inception Distance; lower is better. FID-10k for sweeps, FID-50k for the four final configurations.
- **Precision / Recall** for generative models — separates fidelity from diversity. Include this: if pruning collapses diversity, FID alone can hide it.
- **Peak GPU memory**, **on-disk parameter count**, **latency**.

### 5.7 What a win looks like

> At matched memory budget, TAES (B=4) achieves lower FID than global importance scoring (B=1), consistently across k values and both domains.

Plot: memory on x, FID on y, one curve per method. **If the TAES curve sits below the global curve, the paper is done.** That is the whole result.

### 5.8 Reporting discipline

- Fixed seeds, reported.
- 3 seeds and ±std for any load-bearing number.
- Report failures. If TAES loses at some k, say so and discuss.
- State the scope limits in the paper before a reviewer does: one backbone, one dataset, two domains.

---

## 6. How it stands out

| Existing work | What it does | Why it does not cover us |
|---|---|---|
| EASY-EP, DSMoE, Ban&Pick, RICE, MoTE, DERN | Training-free expert steering | Language only; no timestep axis exists there |
| DiT-MoE | Builds MoE DiT; analyses specialization | Analysis only; proposes no pruning method; is the claim we test |
| ALTER, DiffPruning | Timestep-aware efficiency for diffusion | Use timestep as the *specialization* axis for training experts; we use it as a *scoring* axis for pruning, training-free |
| **Diff-MoE (ICML 2025)** | **Trains time-aware, space-adaptive experts into the architecture** | **Nearest prior work — must be positioned against explicitly.** They *build* time-awareness during pretraining; we *extract* time-banded expert subsets post-hoc from a time-agnostic pretrained model, with no gradient updates. **Check their figures before the week-4 GO/NO-GO:** if they already report band-wise expert divergence, our diagnostic figure 2 becomes confirmatory rather than novel, and the contribution must lean harder on the pruning result. |
| EfficientMoE / DSMoE | Training recipe for diffusion MoE; releases N=48 checkpoints | Trains better models; we compress the ones they released. **Our backbone** — so we are compressing, not competing with, this work |
| Dense2MoE, Race-DiT, EC-DiT, ProMoE | Better MoE diffusion architectures | Build models; we compress existing ones |
| OBS-Diff, Diff-Pruning, HierarchicalPrune | Diffusion compression | Prune weights/layers/blocks, not experts; MoE-unaware |
| EMO | Domain-modular MoE | Requires pretraining from scratch (1T tokens); we are post-hoc and free |

**The empty cell:** training-free, expert-level, timestep-aware, diffusion. That is us.

---

## 7. Known risks, honestly stated

### Risk 1 — Union saturation. **The most serious risk. Read this before starting.**

Memory mode keeps the **union** of the per-band top-k expert sets. With *N* experts per layer, *k* kept per band, and *B* bands:

```
|union|  ∈  [ k ,  min(N, k·B) ]
```

- **Bands agree** → union ≈ k → large memory saving → **but banding didn't matter, so the scientific claim is dead.**
- **Bands diverge** → union → N → **no memory saving, so the engineering claim is dead.**

At N=8, k=4, B=4 the union lies in [4,8] and realistically saturates at 8. **At small N, the stronger your result, the weaker your result.** This is structural, not a tuning problem.

It dissolves as N grows. At N=64, k=8, B=4 the union is in [8,32] — 50–87% saving while bands still differ meaningfully.

**Status as of 2026-07-31 — largely mitigated by the backbone choice.** On DSMoE-S-E48, N=48 and the router's own top-k is 5, i.e. ~10.4% of experts active per token. At k=12, B=4 the union lies in [12, 48]; at k=8, B=4 it lies in [8, 32]. There is real room for bands to diverge *and* still leave a reportable memory saving — the regime the risk analysis said we needed. Under DiT-MoE's 2-of-8 (25% active) that room did not exist. **The risk is reduced, not removed:** where the union actually falls inside those intervals is an empirical question, and it is exactly what diagnostic figure 3 (Step 3.3) measures before any GPU time is spent on pruning.

**Consequences:**
1. **Expert count is a viability precondition.** ✅ *Resolved — N=48. See `taes/notes/backbone-survey.md`.* Needed N ≥ 16, ideally 32+; verified from config files and the HuggingFace API, no compute.
2. **Compute mode is immune.** At band *b* only S_b activates, so divergent bands are pure benefit. If N is small, compute mode becomes the primary contribution. See §5.4 and Decision 8.
3. **Report union size as a result.** Mean |union| / N per layer is a genuine finding about how band-localized diffusion routing is, whichever way it comes out.

### Risk 2 — DiT-MoE says routing is insensitive to class conditioning
If expert selection does not vary by domain, domain-conditioned pruning has nothing to grip. *Mitigation:* timestep is the primary axis and DiT-MoE says routing **is** sensitive to it. Domain is secondary by design — see §1 and Decision 9. The week-4 go/no-go catches this after ~20 GPU-hours.

### Risk 3 — Being scooped
Highest-probability risk. The move is obvious to anyone standing in both fields, and both are active in 2026. *Mitigation:* arXiv at week 7, not week 12.

### Risk 4 — Setup failure in week 1
*Mitigation:* hard 3-day limit on **DSMoE-S-E48**, then **DSMoE-B-E48** (same N=48, same recipe, larger), then a tiny MoE-DiT trained on CIFAR-10 (~15 GPU-hours, and you control N — set it to 32). Decide by end of week 1.

*Note:* this risk went **up**, not down, with the backbone switch. EfficientMoE is a Dec-2025 codebase that depends on DiffMoE's repo structure and pins `torch==2.6.0` plus a git-installed `torch-fidelity` fork. It is less battle-tested than DiT-MoE. The 3-day rule is not decorative.

### Risk 5 — Negative results are not a free safety net
A negative result publishes when it comes with a **mechanism** — "expert importance is timestep-invariant in diffusion MoE, and here is why" — not as "we tried and it didn't work." The telemetry gives you the diagnostic depth to do this properly, but the negative version requires *more* analysis, not less. Budget for it rather than assuming it is the easy exit.

---

## 8. Decision log

*Consult this before changing direction. If a temptation is here, it was already considered.*

**Decision 1 — Not LLMs. (31 Jul)**
2026 alone produced Less is MoE, Domain-Specific Expert Pruning (PMLR), REAP, AIMER, ConMoE, MoE Pathfinder, FlexMoE, and more. The field is saturated. *Do not reopen this.*

**Decision 2 — Not paid compute. (31 Jul)**
₹26,000 is not a reasonable ask for someone starting out. Kaggle's free 30 h/week is the budget. Any plan that needs an A100 rental is wrong by construction.

**Decision 3 — Training-free, not fine-tuning. (31 Jul)**
Resolves the tension between "must be a method, not an analysis" and "must be free." Training-free methods are accepted as methods and cost inference only. The earlier router-fine-tuning plan (v3) needed 300–500 A100-hours and is dead.

**Decision 4 — Journal, not CV conference. (31 Jul)**
Conferences want a method beating baselines with no room for analysis; journals accept scoped empirical contributions and have no deadline pressure. Targets: Pattern Recognition, Neurocomputing, IEEE TIP.

**Decision 5 — Drop the "MoE reduces output randomness" idea. (31 Jul)**
Diffusion randomness comes from the initial latent and sampler noise; routing touches neither. Worse, [Expert-Data Alignment](https://consensus.app/papers/details/c82f0b8290055b69b3299cfdfd31a7f7/) (2026) found *more* stable routing gave *worse* FID (47.9 vs 22.6). If routing variance appears in the paper at all, it is a measurement with no predicted direction — never a claim.

**Decision 6 — Timestep banding is the contribution, not domain conditioning. (31 Jul)**
Domain conditioning alone is EASY-EP, and DiT-MoE says diffusion routing may be domain-insensitive anyway. Banding over timestep is both the novel part and the part DiT-MoE's own findings support.

**Decision 7 — 8 weeks, with named cuts. (31 Jul)**
Cut: 4 domains→2, quantization baselines, cross-domain transfer, second backbone, FID-50k everywhere. Kept: the entire core claim. *Superseded in part by Decision 8 — compute mode is restored.*

**Decision 8 — Compute mode restored; mode choice is now data-dependent. (31 Jul)**
Cutting compute mode in v5 was an error. Memory mode suffers union saturation (§7 Risk 1): at small N, bands that genuinely differ produce a union covering nearly all experts, so no memory is saved. Compute mode is structurally immune, because at band *b* only S_b activates. **Which mode leads is decided by the week-1 expert count**, per the rule in §5.4. Do not commit to memory mode before that number is known.

**Decision 9 — Timestep is primary, domain is secondary. (31 Jul)**
Promoted from contingency to plan. Domain-conditioned expert pruning is EASY-EP, already published in language; leading with it invites "isn't this just EASY-EP on a different model?" Timestep banding is an axis that does not exist in any prior work in this family. B=1 vs B=4 is the entire story; domain is a conditioning variable that adds a second dimension to the table, nothing more.

**Decision 10 — Expert count is a go/no-go precondition, checked before any GPU work. (31 Jul)**
Requires N ≥ 16, ideally 32+. Costs an afternoon of reading configs, no compute. If only N=8 checkpoints exist, switch backbone immediately rather than proceeding hopefully. This check now precedes even the checkpoint-loading gate.

**Decision 11 — Backbone is DSMoE-S-E48, not DiT-MoE. (31 Jul) — EXECUTED, Decision 10 in action**
Decision 10 fired on day one. DiT-MoE's advertised `S/2-16E2A` checkpoint does not exist in the HuggingFace file tree; every usable release is N=8, and the sole N=16 release (G/2) is 33 GB. Fallback search found `YHLLEO/DSMoE-S-E48` — N=48, top-k=5, 1.11 GB, ImageNet-256, rectified flow. **Memory mode is now the primary claim** (§5.4 rule, N≥32). Full reasoning: `taes/notes/backbone-survey.md`.

Downstream changes, all already made in this guide: sampler is rectified flow not DDIM (§5.1); the shared expert is excluded from N and from all pruning ratios (§5.1); MoE sits in 6 of 12 blocks via `interleave: true`, so `layer_idx` in telemetry means MoE-layer index; k-sweep must satisfy k ≥ 5 = router top-k.

**Decision 12 — Do not switch back to DiT-MoE for "comparability with the prior finding." (31 Jul)**
Tempting, because DiT-MoE is the paper whose specialization claim we test. Rejected: we test that claim as a *hypothesis about diffusion MoE routing generally*, not as a reproduction of one repo. DSMoE is the same architectural family with a router of the same form. If a reviewer wants DiT-MoE specifically, the answer is that its N=8 makes memory mode structurally untestable (§7 Risk 1) — which is itself worth one sentence in Limitations.

---

## 9. Complete bibliography

### 9.1 Primary — training-free MoE steering (the family we port)

1. **RICE** — Wang et al., 2025. *Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training.* [link](https://consensus.app/papers/details/3132c0b6b6a85fcb95356dfb23793c8a/)
2. **MoTE** — Dahlke et al., 2025. *Mixture of Tunable Experts — Behavior Modification of DeepSeek-R1 at Inference Time.* [link](https://consensus.app/papers/details/b8f295008666522cb54665e5aabe1dde/)
3. **Ban&Pick** — Chen et al., 2025. *Enhancing Performance and Efficiency of MoE-LLMs via Smarter Routing.* [link](https://consensus.app/papers/details/d0e3846be58752f4b9b739e828da7dc4/)
4. **DERN** — Zhou et al., 2025. *Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse MoE LLMs.* [link](https://consensus.app/papers/details/d2c5c8e4b7e15bbca23316661376ef75/)
5. **EASY-EP** — Dong et al., 2025. *Domain-Specific Pruning of Large MoE Models with Few-shot Demonstrations.* [link](https://consensus.app/papers/details/c7799a80bf395846b722f938973968a3/) — **our scoring function**
6. **DSMoE** — Do et al., 2026. *Do Domain-specific Experts exist in MoE-based LLMs?* [link](https://consensus.app/papers/details/cc2d16cc2961545bb6f55a8298e5aa17/) · [code](https://github.com/giangdip2410/Domain-specific-Experts)
7. **MASCing** — te Lintelo et al., 2026. *Configurable MoE Behavior via Activation Steering Masks.* [link](https://consensus.app/papers/details/8836fce80a9453c8bfdcfa23ebc8b0e1/)
8. **SSMoE** — Do et al., 2026. *Eigenvectors of Experts are Training-free Non-collapsing Routers.* [link](https://consensus.app/papers/details/ab80d6eef03a5246b37875a3cfdb559d/)

### 9.2 Primary — diffusion MoE (our target)

9. **DiT-MoE** — Fei et al., 2024. *Scaling Diffusion Transformers to 16 Billion Parameters.* [link](https://consensus.app/papers/details/6cd0f0dc646e53b5bb121ceeba473e9d/) · [code](https://github.com/feizc/DiT-MoE) — **the specialization claim we test.** *No longer the backbone (Decision 11): usable releases are N=8.*
9a. **EfficientMoE / DSMoE** — Liu et al., 2025. *Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe.* [arXiv 2512.01252](https://arxiv.org/abs/2512.01252) · [code](https://github.com/yhlleo/EfficientMoE) · [DSMoE-S-E48](https://huggingface.co/YHLLEO/DSMoE-S-E48) — **our backbone.** Read §4 (expert-count study) before Phase 2.
10. **Diff-MoE** — Cheng et al., ICML 2025. *Diffusion Transformer with Time-Aware and Space-Adaptive Experts.* [code](https://github.com/kunncheng/Diff-MoE) · [proceedings](https://proceedings.mlr.press/v267/cheng25d.html) — **nearest prior work; must be positioned against in Related Work (§6).** No released checkpoints, so not a usable fallback.
11. **ProMoE** — Wei et al., 2025. *Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance.* [link](https://consensus.app/papers/details/b0fa723d4364540b8810c0795b22591b/)
12. **SharpMoE** — Deng et al., 2026. *Saliency-Harnessing Accurate Routing for Diffusion MoE.* [link](https://consensus.app/papers/details/dbd920a3439c5825b6c78aa73f87c7c2/)
13. **Race-DiT** — Yuan et al., 2025. *Expert Race: A Flexible Routing Strategy.* [link](https://consensus.app/papers/details/1dec6e1d4e53562e84f950bd231be7d2/)
14. **EC-DiT** — Sun et al., 2024. *Scaling Diffusion Transformers with Adaptive Expert-Choice Routing.* [link](https://consensus.app/papers/details/a712c6391d3d5d908beda79f59ca143a/)
15. **Dense2MoE** — Zheng et al., ICCV 2025. *Restructuring Diffusion Transformer to MoE.* [link](https://consensus.app/papers/details/d4719f1be64f5755be0a9d6296f09539/)

### 9.3 Diffusion compression (related, non-MoE — for Related Work)

16. **Diff-Pruning** — Fang et al., 2023. *Structural Pruning for Diffusion Models.* [link](https://consensus.app/papers/details/a3fe817bd0195f96950dfbe7f5cbe8b2/)
17. **ALTER** — Yang et al., 2025. *Layer Pruning and Temporal Expert Routing.* [link](https://consensus.app/papers/details/70b9be1944e252b38596709c856d87af/)
18. **DiffPruning** — Ganjdanesh et al., 2024. *Mixture of Efficient Diffusion Experts.* [link](https://consensus.app/papers/details/e34f2879033a544abb3ba6065654f23a/)
19. **OBS-Diff** — Zhu et al., 2025. *Accurate Pruning for Diffusion Models in One-Shot.* [link](https://consensus.app/papers/details/2fda0488ae46584bae34a6ce0c24276e/)
20. **HierarchicalPrune** — Kwon et al., 2025. [link](https://consensus.app/papers/details/10c78e7400d457229c0475fe0b573143/)
21. **LD-Pruner** — Castells et al., 2024. [link](https://consensus.app/papers/details/9799ab66d8835288b133887bfa8905a6/)

### 9.4 MoE expert pruning, LLM side (context — the saturated field)

22. **EEP** — Liu et al., 2024. [link](https://consensus.app/papers/details/b0cd8c57aca05b42b53982bd5e2e13ae/)
23. **MoE-Pruner** — Xie et al., 2024. [link](https://consensus.app/papers/details/042371d94ec95dd58ef4eea91ac83e71/)
24. **Not All Experts are Equal** — Lu et al., 2024. [link](https://consensus.app/papers/details/6c2526f21955597880a896832396cd2d/)
25. **SEER-MoE** — Muzio et al., 2024. [link](https://consensus.app/papers/details/dd921410763550a18ac98f88a4df88f0/)
26. **DiEP** — Bai et al., 2025. [link](https://consensus.app/papers/details/a4b706ac80035b528936a9a246d62f31/)
27. **C-Prune** — Guo et al., 2025. [link](https://consensus.app/papers/details/f15c2e7522675e7f8747f180bbf744f5/)
28. **Sub-MoE** — Li et al., 2025. [link](https://consensus.app/papers/details/6a229b72e0ec5635b564cf626252a386/)
29. **Mosaic Pruning** — Hu et al., 2025. [link](https://consensus.app/papers/details/0574472f8f605472933e9dc9d5cf0377/)
30. **MoNE** — Zhang et al., 2025. [link](https://consensus.app/papers/details/bb55603713a155bc87015bd76e42988f/)
31. **EAC-MoE** — Chen et al., 2025. [link](https://consensus.app/papers/details/0658f98329a95fae9a575de9bdb2f4c6/)
32. **Task-Specific Expert Pruning** — Chen et al., 2022. [link](https://consensus.app/papers/details/56c42f3cf6d55d03a125aefafba8c326/)

### 9.5 Expert specialization — the 2026 controversy (framing)

33. **The Illusion of Specialization** — Wang et al., 2026. [link](https://consensus.app/papers/details/4c2a11747360510db1349851fa791f16/)
34. **EMO** — Wang et al., 2026. *Pretraining MoE for Emergent Modularity.* [link](https://consensus.app/papers/details/0f6963a6f9105932b765ed00dfd6ab1c/)
35. **MoE Lens** — Chaudhari et al., 2026. [link](https://consensus.app/papers/details/03d7d6b635d158dfaf16578db5b8a55a/)
36. **ViMoE** — Han et al., IEEE TIP 2025. [link](https://consensus.app/papers/details/6ffb490e191a5c20952280e04b3e1980/)

### 9.6 Routing stability (for the discussion section)

37. **R3** — Ma et al., 2025. *Stabilizing MoE RL by Aligning Training and Inference Routers.* [link](https://consensus.app/papers/details/bda4a1db02db504caabb6ef4bf96d9fd/)
38. **VSRAQ** — Park et al., 2026. [link](https://consensus.app/papers/details/b0ab519ff08e558db235fbdf8b958de5/)
39. **VMoER** — Li et al., 2026. *Variational Routing.* [link](https://consensus.app/papers/details/ba46ec0ce97d5a14ad23842cfda5ff5f/)
40. **Expert-Data Alignment Governs Generation Quality in DDMs** — Villagra et al., 2026. [link](https://consensus.app/papers/details/c82f0b8290055b69b3299cfdfd31a7f7/) — **the caveat; read before writing Discussion**
41. **MoE as Soft Clustering** — Liu, 2026. [link](https://consensus.app/papers/details/b66a1032cdd950e19f54d0e3cce9358d/)

### 9.7 Foundational

42. **Mixture-of-Experts with Expert Choice Routing** — Zhou et al., 2022. [link](https://consensus.app/papers/details/ae8eb98b3c005a1da7c1b6bafd8bd3b8/)
43. **StableMoE** — Dai et al., 2022. [link](https://consensus.app/papers/details/97e7944d0db957038e3fa91f60d2e8ba/)
44. **OLMoE** — Muennighoff et al., 2024. [link](https://consensus.app/papers/details/f3b3439fb53e53beb3bc1d63296b9608/)
45. **ControlNet** — Zhang et al., ICCV 2023. [link](https://consensus.app/papers/details/4c4c4002b91853f3879b756ade9d56f8/)

### 9.8 Code repositories

- [yhlleo/EfficientMoE](https://github.com/yhlleo/EfficientMoE) — **primary backbone** · ckpt: [YHLLEO/DSMoE-S-E48](https://huggingface.co/YHLLEO/DSMoE-S-E48) · fallback ckpt: [YHLLEO/DSMoE-B-E48](https://huggingface.co/YHLLEO/DSMoE-B-E48)
- [KlingTeam/DiffMoE](https://github.com/KlingTeam/DiffMoE) — EfficientMoE depends on its repo structure and eval protocol; read before Phase 1
- [feizc/DiT-MoE](https://github.com/feizc/DiT-MoE) — *rejected as backbone (N=8); still the source of the specialization claim*
- [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE) — no checkpoints released; Related Work only
- [giangdip2410/Domain-specific-Experts](https://github.com/giangdip2410/Domain-specific-Experts) — DSMoE **(the LLM paper, Do et al.)** reference implementation. ⚠️ **Name clash:** "DSMoE" refers to two unrelated things in this project — Do et al.'s LLM domain-steering paper (§9.1 #6) and Liu et al.'s diffusion backbone (§9.2 #9a). In the paper, always disambiguate.
- [ali-vilab/ProMoE](https://github.com/ali-vilab/ProMoE)
- [VainF/Diff-Pruning](https://github.com/VainF/Diff-Pruning)
- [MoE-Inf/awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference) — survey repo, useful for Related Work

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Expert** | One FFN inside an MoE layer |
| **Router / gate** | Small linear layer choosing which experts a token goes to |
| **Top-k** | Number of experts activated per token |
| **Gate score** | Router's confidence in an expert for a token |
| **Timestep (t)** | Diffusion denoising step; high = noisy, low = nearly clean |
| **Band** | A contiguous range of timesteps; our unit of scoring |
| **DiT** | Diffusion Transformer |
| **Latent** | Compressed VAE representation the diffusion model operates in |
| **FID** | Fréchet Inception Distance; image quality metric, lower better |
| **Precision / Recall** | Generative metrics separating fidelity from diversity |
| **Training-free** | No gradient updates; inference and statistics only |
| **Calibration set** | Small sample used to collect routing statistics |
| **Memory mode** | Experts physically deleted → smaller checkpoint |
| **Compute mode** | Experts masked at inference → less compute, same memory |
