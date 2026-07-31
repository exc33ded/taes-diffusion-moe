# Backbone Survey — Step 0.1 / 0.2

**Date:** 2026-07-31
**Question:** What is N (experts per MoE layer) in *actually released* diffusion-MoE checkpoints?
**Why it matters:** N sets the ceiling on how much banded expert selection can differ across bands. Small N (=8) with top-k=2 leaves almost no room for band-specific subsets → the whole TAES premise collapses (§7 Risk 1).

---

## 1. DiT-MoE (feizc/DiT-MoE, arXiv 2407.11633)

### Architecture facts (from `models.py`)
- **Every** DiT block contains a `SparseMoeBlock` → **MoE layers = depth**. No interleaving.
- Router: `MoEGate`, softmax over experts, `torch.topk`, `norm_topk_prob=False` (gate scores are raw softmax probs, NOT renormalised — good, they are directly comparable as a confidence term).
- `n_shared_experts = 2` — but note this is implemented as **one** `MoeMLP` with `intermediate_size = 2 * embed_dim`, always applied, added to the routed output. **It is not routed and cannot be pruned.** Must be excluded from N and from all importance scoring.
- Routed expert width: `intermediate_size = mlp_ratio * embed_dim = 4 * embed_dim`.
- `N` and top-k are **constructor arguments**, not baked into variant names — `DiT_models` registry only fixes depth/hidden/patch/heads.

### Variant geometry (`DiT_models`)

| Variant | depth (= # MoE layers) | hidden | heads | patch |
|---|---|---|---|---|
| DiT-S/2 | 12 | 384 | 6 | 2 |
| DiT-B/2 | 12 | 768 | 12 | 2 |
| DiT-L/2 | 24 | 1024 | 16 | 2 |
| DiT-XL/2 | 28 | 1152 | 16 | 2 |
| DiT-G/2 | 40 | 1408 | 16 | 2 |

### Released checkpoints — **verified against the HF file tree**, not the README

HF API: `https://huggingface.co/api/models/feizhengcong/DiT-MoE/tree/main`

| Checkpoint | N | top-k | MoE layers | Res | Ckpt size | Released? |
|---|---|---|---|---|---|---|
| dit_moe_s_8E2A.pt | 8 | 2 | 12 | 256 | 0.80 GB | ✅ |
| dit_moe_b_8E2A.pt | 8 | 2 | 12 | 256 | 3.18 GB | ✅ |
| dit_moe_xl_8E2A.pt | 8 | 2 | 28 | 256 | 8.34 GB | ✅ |
| dit_moe_g_16E2A.pt | 16 | 2 | 40 | 256 | 33.0 GB | ✅ |
| **dit_moe_s_16E2A.pt** | 16 | 2 | 12 | 256 | — | ❌ **README links it; file is NOT in the repo** |

> ⚠️ **Finding 1.** The README advertises a `DiT-MoE-S/2-16E2A` checkpoint. It does not exist on HuggingFace. The tree listing contains exactly four `.pt` model files and the VAE; `dit_moe_s_16E2A.pt` is absent. Anyone planning around it (as the research plan did) is planning around a dead link.

> ⚠️ **Finding 2.** Therefore the **only** released DiT-MoE with N=16 is **G/2 at 33 GB** — a 16.5B-parameter model. Not loadable on a Kaggle T4/P100 (16 GB), and not sampleable at 10k images within quota. Effectively unusable for this project.

**Net for DiT-MoE: every usable released checkpoint has N=8, top-k=2.**

---

## 2. Decision rule (WORKOUT-PLAN Step 0.1)

- N ≥ 32 released → proceed, memory mode primary
- 16 ≤ N < 32 → proceed, both modes
- **Only N = 8 → stop, do Step 0.2** ← **this is where DiT-MoE lands**

**→ Step 0.2 triggered.**

---

## 3. Step 0.2 — Fallback search

### 3.1 Diff-MoE (kunncheng/Diff-MoE, ICML 2025)
"Diffusion Transformer with Time-Aware and Space-Adaptive Experts."
- GitHub exists; **no HuggingFace model repo** under `kunncheng` (API returns empty).
- Repo README covers install + ImageNet-256 training only. **No released checkpoints found.**
- Also worth noting for Related Work: this paper is *adjacent to our thesis* — it builds time-awareness into training. Our claim is different (post-hoc, training-free extraction from an off-the-shelf MoE), but §Related Work must distinguish us from it explicitly. See results-log.
- **Verdict: not usable as a backbone.**

### 3.2 EfficientMoE (yhlleo/EfficientMoE, arXiv 2512.01252) — ✅ **WINNER**
*"Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe"* (Liu et al., 2025).
Explicitly studies **varying expert counts** — exactly the axis we need. Checkpoints ARE on HF.

Released model repos under `YHLLEO`:

| Repo | N | Family | Notes |
|---|---|---|---|
| DSMoE-S-E16 | 16 | latent DiT | |
| DSMoE-B-E16 | 16 | latent DiT | |
| DSMoE-L-E16 | 16 | latent DiT | |
| DSMoE-3B-E16 | 16 | latent DiT | |
| **DSMoE-S-E48** | **48** | latent DiT | **primary candidate** |
| **DSMoE-B-E48** | **48** | latent DiT | **scale-up candidate** |
| DSMoE-L-E48 | 48 | latent DiT | too large for quota |
| JiTMoE-B16-E16 | 16 | pixel-space | |
| JiTMoE-L16-E16 | 16 | pixel-space | |

#### DSMoE-S-E48 — verified config (`config_2025-11-08T22-26-38.yaml`)

```yaml
target: models.models_DSMoE.DiT
depth: 12          hidden_size: 384      num_heads: 6      patch_size: 2
input_size: 32     num_classes: 1000     image_size: 256   rf: true   # rectified flow
MoE_config:
  num_experts: 48                 # N = 48
  num_experts_per_tok: 5          # top-k = 5
  moe_intermediate_size: 128      # narrow experts (vs 4*384=1536 in DiT-MoE)
  interleave: true                # <-- MoE in alternate blocks only
  skip_first2: false  skip_last2: false
  use_shared_expert: true         # shared expert present, NOT prunable
  n_group: 1  topk_group: 1  routed_scaling_factor: 2.5  capacity: 1
vae_path: stabilityai/sd-vae-ft-mse
```

- **Checkpoint:** `checkpoints/0700000.pt`, **1.11 GB** — comfortably Kaggle-sized.
- **Reported quality:** FID-50K 14.81 @ CFG 1.5, IS 96.51, **30M active params** (700K steps).
- `interleave: true` ⇒ MoE in **6 of 12** blocks, not all 12. Confirm the exact indices from `models_DSMoE.py` before writing `hooks.py`.

---

## 4. DECISION

**Backbone: `YHLLEO/DSMoE-S-E48` (N = 48, top-k = 5, depth 12, 256×256, rectified flow, 1.11 GB).**

**Reasoning.**
1. **N = 48 clears the N ≥ 32 bar by a wide margin.** DiT-MoE cannot supply this: its only ≥16 release is a 33 GB model. This is the single fact that decides the backbone.
2. **top-k = 5 of 48 ⇒ ~10.4% of experts active per token.** Sparse enough that per-band top-k sets have real room to differ. Contrast DiT-MoE S/2: 2 of 8 = 25%, where any two bands are near-forced to overlap. §7 Risk 1 is materially reduced but **not eliminated** — union size across bands is still an empirical question, which is exactly what diagnostic figure 3 measures.
3. **1.11 GB fits Kaggle** with headroom for batch-64 sampling — Step 1.2's seconds-per-image budget stays realistic.
4. **Memory mode becomes the primary claim.** With N=48 and k=24 (N/2), physically slicing 24 experts per layer is a large, reportable on-disk reduction. Under DiT-MoE's N=8 this claim was never going to be interesting.
5. **The narrow experts (`moe_intermediate_size: 128`) are a caveat, not a blocker.** Each expert is smaller, so pruning a fixed *fraction* removes less absolute memory than in DiT-MoE. Report reductions as fraction-of-routed-params as well as absolute MB.

**Backbone risk accepted:** this is a Dec-2025 codebase depending on DiffMoE's repo structure. Step 1.2's 3-day rule applies. **Fallback if it does not load: `DSMoE-B-E48`** (same N=48, same recipe, larger); **second fallback: train our own N=32 MoE-DiT on CIFAR-10** per Step 0.2.3.

**Secondary backbone (for the generalisation claim, if quota allows):** `DSMoE-B-E48` — same N, different scale. A second backbone at the same N is worth more to reviewers than a second N at the same scale.

---

## 5. Consequences for the plan — carry these forward

1. **`REFERENCE-GUIDE.md` and `research-plan-v5` assume DiT-MoE.** They need amending: architecture, router API, and the rectified-flow sampler (not DDIM) all change. Step 1.3's "compare to the number in the backbone's paper" now means **FID-50K 14.81 @ CFG 1.5** for S-E48 — but our Step 1.3 gate is FID-**10k** at our own settings, so expect a gap; we need it *stable*, not equal.
2. **Rectified flow, not DDIM.** Timesteps run t ∈ [0,1] continuous, `timestep_start: 0, timestep_end: 1000`. The banding axis is still time, but bin boundaries must be defined on the RF schedule. `hooks.py` should record the raw t it was called with and bin later — keeps B free, as Step 3.2 requires.
3. **Shared expert must be excluded** from N, from importance scoring, and from all pruning ratios in both DiT-MoE and DSMoE. Any memory-reduction number that includes it is wrong.
4. **`interleave: true`** — verify which 6 blocks carry MoE before writing hooks. `layer_idx` in telemetry should be the **MoE-layer index**, with the block index recorded alongside.
5. **top-k = 5, not 2.** The frequency term in `I(e|d,b,l)` normalises against 5 selections per token. The k-sweep in Step 5.2 should span k ∈ {5, 8, 12, 16, 24, 32, 48} — k must stay ≥ top-k or the router cannot fill its top-k.
6. **Diff-MoE (ICML 2025) is the nearest prior work** and must be positioned against in Related Work: they *train* time-aware experts; we *extract* time-banded subsets post-hoc from a time-agnostic model, training-free. If they already report band-wise expert divergence, our diagnostic figure 2 is confirmatory rather than novel — check their figures before Week 4's GO/NO-GO.

---

## 6. Sources

- [DiT-MoE GitHub](https://github.com/feizc/DiT-MoE) · [models.py](https://github.com/feizc/DiT-MoE/blob/main/models.py) · [arXiv 2407.11633](https://arxiv.org/abs/2407.11633)
- [feizhengcong/DiT-MoE HF file tree](https://huggingface.co/feizhengcong/DiT-MoE/tree/main)
- [kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE) · [ICML 2025 proceedings](https://proceedings.mlr.press/v267/cheng25d.html)
- [yhlleo/EfficientMoE](https://github.com/yhlleo/EfficientMoE) · [arXiv 2512.01252](https://arxiv.org/abs/2512.01252)
- [YHLLEO/DSMoE-S-E48](https://huggingface.co/YHLLEO/DSMoE-S-E48) (config + checkpoint verified via HF API)
