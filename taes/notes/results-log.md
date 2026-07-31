# TAES — Results & Decision Log

Append-only. One entry per run, per decision, per failure. Especially failures.

---

## 2026-07-31 — DECISION: backbone switched from DiT-MoE to DSMoE-S-E48

**Phase 0, Step 0.1 → 0.2.**

DiT-MoE fails the N decision rule. The advertised `dit_moe_s_16E2A.pt` is a dead link — not present in the HuggingFace file tree. Every *usable* released DiT-MoE checkpoint is N=8, top-k=2; the sole N=16 release is G/2 at 33 GB, which Kaggle cannot host.

Step 0.2 fallback search: Diff-MoE (ICML 2025) has no released checkpoints. **EfficientMoE (arXiv 2512.01252) does**, including `DSMoE-S-E48` — **N=48, top-k=5, depth 12, 256×256, 1.11 GB**.

**Decision: DSMoE-S-E48 is the backbone.** N=48 clears the N≥32 bar, so **memory mode is the primary claim**. Full reasoning and consequences for hooks/sampler/scoring: `backbone-survey.md` §4–§5.

**Open items created by this decision:**
- [x] Confirm which 6 of 12 blocks carry MoE (`interleave: true`) from `models_DSMoE.py` — **done 31 Jul, see below**
- [ ] Rewrite the DiT-MoE assumptions in `REFERENCE-GUIDE.md` / `research-plan-v5-8WEEK.md`
- [ ] Sampler is rectified flow, not DDIM — Step 1.2 timing and Step 1.3 FID command change
- [ ] Read Diff-MoE's figures before the Week-4 GO/NO-GO; if they already show band-wise expert divergence, our diagnostic 2 is confirmatory, not novel

---

## 2026-07-31 — Step 1.2 source audit of `models_DSMoE.py` (pre-load)

Read from `EfficientMoE/DSMoE/models/models_DSMoE.py` at HEAD, cross-checked against
`config_2025-11-08T22-26-38.yaml`. All findings are from source, not from a loaded model.

### F1 — MoE block indices: **1, 3, 5, 7, 9, 11** (0-indexed)

```python
use_moe_flag = [True] * depth
if self.MoE_config.interleave:
    use_moe_flag = [i % 2 == 1 for i in range(depth)]   # line 298
```
`skip_first2: false`, `skip_last2: false` ⇒ no further modification. depth=12 ⇒ odd blocks only.

**Mapping for telemetry:** MoE-layer index `l ∈ {0..5}` ↔ block index `2l + 1`.
Record both, per Phase-0 carry-over item 4.

### F2 — Router is **sigmoid**, not softmax; group routing is a no-op

`route_tokens_to_experts` (line 120) does `router_logits.sigmoid()`. Scores are independent
per-expert sigmoids in [0,1], *not* a distribution summing to 1.
`n_group: 1`, `topk_group: 1` ⇒ the DeepSeek group-limited routing reduces to plain top-5 of 48.

### F3 — ⚠️ `norm_topk_prob=True` — the returned gate weights are NOT a confidence signal

Hardcoded `norm_topk_prob=True` at construction (line 211), overriding nothing:

```python
topk_weights = router_logits.gather(1, topk_indices)      # line 139  <- raw sigmoid
if self.top_k > 1 and self.norm_topk_prob:
    topk_weights /= (topk_weights.sum(-1, keepdim=True) + 1e-20)
topk_weights = topk_weights * self.routed_scaling_factor  # x2.5
```

Returned `topk_weights` **always sum to 2.5 per token**, whatever the router's actual
confidence. Using them as the gate term in `I(e|d,b,l)` would measure only relative share
among the chosen 5 — the absolute confidence information is destroyed by the normalisation.

**This contradicts the DiT-MoE assumption recorded in `backbone-survey.md` §1**
(`norm_topk_prob=False` there, so raw probs were directly usable). The scoring design
inherited that assumption and must be amended.

**Decision:** `hooks.py` records the **pre-normalisation** value
`router_logits.gather(1, topk_indices)` — raw sigmoid in [0,1].
Consequence: we cannot hook `DSMoE.forward`'s output alone; the quantity we need is
internal to `route_tokens_to_experts`. Hook must target `TopkRouter` (`.gate`) output and
recompute selection, or monkeypatch `route_tokens_to_experts`. Decide in Step 2.1.

### F4 — Selection uses a bias that weighting does not

```python
router_logits_for_choice = router_logits + self.gate.e_score_correction_bias  # line 121
topk_indices  = topk(scores_for_choice, k=top_k)   # biased   -> WHICH experts
topk_weights  = router_logits.gather(1, topk_indices)  # unbiased -> HOW MUCH
```

`e_score_correction_bias` is a learned load-balancing correction that shifts *selection*
only. It is a pretraining artefact sitting directly on the frequency term of our importance
score. **Dump this vector when the model loads and save it with the telemetry** — it is the
concrete handle on the Phase-4 NO-GO question "is timestep-invariance an artefact of the
load-balancing loss?" (WORKOUT-PLAN Step 4.1).

### F5 — No standalone sampling script

Only `sample_ddp_feature.py` (DDP, feature-dumping). WORKOUT-PLAN Step 1.2.3's "run the
repo's own sampling script unmodified" is not literally executable. Plan: drive it
single-process, else write a minimal RF sampler against `DiT.forward_with_cfg`.
**This is the main 1.2 risk against the 3-day rule (clock started 31 Jul → fallback 3 Aug).**

### F6 — Deviations from the WORKOUT-PLAN install list (deliberate)

| Pinned | Doing | Why |
|---|---|---|
| `torch==2.6.0` | keep stock `2.10.0+cu128` | downgrade on a cu128 image is slow and breaks the CUDA stack; pin may be incidental. Revisit only on a version-specific failure. |
| `tensorflow==2.15.0` | **skip entirely** | only needed for the ADM/TF eval suite. We locked `pytorch-fid` (Step 1.3), so TF is dead weight and the most conflict-prone pin in the list. |

FID implementation **locked: `pytorch-fid`**, reference stats to be computed once from ADM's
`VIRTUAL_imagenet256_labeled.npz` and stored in the Kaggle Dataset. Never changes again.

### F7 — Hook paths (RESOLVED)

`DiTBlock.__init__` swaps `self.mlp` by flag — `DSMoE` when `use_moe`, plain `Mlp` otherwise.
The MoE module therefore sits at the ordinary `.mlp` attribute:

| Target | Module path | Type |
|---|---|---|
| MoE block | `blocks.{2l+1}.mlp` | `DSMoE` |
| Router | `blocks.{2l+1}.mlp.gate` | `TopkRouter` |
| Routed experts | `blocks.{2l+1}.mlp.experts.blocks[e]` | `MLP`, e ∈ 0..47 |
| Shared expert | `blocks.{2l+1}.mlp.shared_experts` | `MLP` — **never prune** |

for `l ∈ {0..5}` ⇒ blocks 1, 3, 5, 7, 9, 11.

### F8 — Shared expert: present, one routed-expert wide, and it *is* the residual

`self.shared_experts = MLP(hidden_size, moe_intermediate_size)` = `MLP(384, 128)` —
**identical geometry to each of the 48 routed experts** (unlike DiT-MoE, where it was
`2 * embed_dim` wide; `backbone-survey.md` §1).

⇒ 49 expert-units per MoE layer, 1 unprunable. **Hard memory floor = 1/49 = 2.04% of
routed params.** Use this denominator in every memory-reduction figure.

Structural point:

```python
if self.use_shared_expert:
    hidden_states = hidden_states + self.shared_experts(residuals)
else:
    hidden_states = hidden_states + residuals
```

With the shared expert enabled there is **no identity path inside `DSMoE`** — the shared
expert replaces it. Pruning it would sever the residual connection, not merely degrade
quality. "Never prune the shared expert" is a structural constraint, not a convention.

### F9 — ⚠️ Expert outputs are pre-weighted — `I(e|d,b,l)` double-counts as written

`DSNaiveMoE.forward` line 70:
```python
current_hidden_states = self.blocks[expert_idx](current_state) * top_k_weights[top_x, idx, None]
```
The gate weight is applied **inside** the expert loop. A forward hook on
`experts.blocks[e]` output therefore captures `gate × f_e(x)`, so
`I = freq × gate × L2` would multiply the gate term in twice.

**Resolve in Step 2.1**, options: (a) hook the expert `MLP` module output directly — it is
pre-multiplication, since the scaling happens in the caller, not the module; or (b) divide
out the recorded weight; or (c) redefine the contribution term as gate-weighted L2 and drop
the separate gate factor. (a) is cleanest and is the default plan.

### F10 — MODEL LOADS. Step 1.2 load gate PASSED (31 Jul, day 1 of 3)

Kaggle P100, stock `torch 2.10.0+cu128` (no downgrade), TF never installed.
Only source change required: stub the unconditional `fla.layers.kda` import in
`models/modules.py` (`use_kda=False` throughout, so it is dead code).

```
MoE flags: [F, T, F, T, F, T, F, T, F, T, F, T]     <- F1 confirmed at runtime
KDA flags: all False
built. total params: 69.2M
ckpt top-level keys: ['model', 'ema', 'opt', 'args']
-> using ckpt['ema']                                <- see below
sd entries: 1001
missing: 0 []      unexpected: 0 []
```

**Use `ckpt['ema']`, never `ckpt['model']`.** Both are present. The paper's FID-50K 14.81
is an EMA number; silently loading raw weights would bias every comparison in the project.
Pin this in `sample.py` and `fid.py`.

**69.2M stored vs 30M active** (paper). All 48 experts held in memory, 5 route per token —
that gap is precisely what memory-mode pruning targets.

### F11 — ⚠️ P100 cannot run this stack. **Use T4 ×2.**

First GPU draw was a P100 (Pascal, **sm_60**). Every CUDA call failed with
`no kernel image is available for execution on the device`.
**PyTorch dropped Pascal support at 2.9**; the stock Kaggle wheel `2.10.0+cu128` ships no
sm_60 kernels. This is a hardware/wheel mismatch, not a config error.

**This vindicates the repo's `torch==2.6.0` pin** — 2.6 is the last release with sm_60.
F6's "the pin is probably incidental" call was wrong, though not for the reason the repo
implies. Resolution: switch accelerator to **T4 ×2** (sm_75) rather than downgrade torch.
T4 also does fp16 at ~65 TFLOPS vs P100's ~19, so it is the faster card for this workload.

**Standing requirement: every Kaggle session must be T4. Guard asserts `sm >= 7.0`.**

### F12 — VAE decode OOMs at batch 64; chunk it

Transformer handles batch 64 fine at 10.76 GB peak. `vae.decode` on 64 latents tries to
allocate 4 GiB in one conv and dies on a 14.56 GB card. Decode in chunks of 8.
The VAE is a **fixed cost pruning never touches**, so timing is reported split — lumping
them together would understate every Phase-5 speedup and flatten the Pareto plot.

### F13 — ✅ Step 1.2 COMPLETE. Reference timing.

```json
{"gpu": "Tesla T4", "torch": "2.10.0+cu128", "weights": "ema",
 "steps": 25, "cfg": 1.5, "batch": 64, "decode_chunk": 8, "precision": "fp32",
 "transformer_s": 19.53, "vae_s": 9.11, "total_s": 28.64,
 "sec_per_image": 0.4475, "transformer_sec_per_image": 0.3052,
 "peak_vram_gb": 10.76}
```

**0.4475 s/image end-to-end** (transformer 0.3052, VAE 0.1423) at 25 RF steps, batch 64,
CFG 1.5, fp32, EMA weights, Tesla T4.

**Compute budget derived from this — governs Phase 5 scheduling:**

| Quantity | Value |
|---|---|
| FID-10k, one config | **75 min ≈ 1.25 GPU-h** — fits one session, no resume needed |
| Kaggle weekly quota | ~30 GPU-h ⇒ **~24 configs/week ceiling** |
| Phase 5 priority 1 (B=1 vs B=4, k=24, 2 domains) | 4 runs ≈ 5 h ✅ |
| Phase 5 priority 2 (k-sweep 7 × 2 B × 2 domains) | 28 runs ≈ **35 h — exceeds one week's quota** |
| Phase 6 FID-50k × 4 configs | ~25 h |

⇒ The k-sweep must drop to one domain, or use the second T4 in parallel.
**Do not reorder the WORKOUT-PLAN Step 5.2 priority list** — it is now quota-critical.

### F14 — ✅ `grid.png` verified. Sampler conventions correct.

16/16 recognisable, correctly class-conditioned ImageNet samples (dogs, bird on branch,
strawberry, lizard, knitwear, mosquito net). Confirms end-to-end: rectified-flow `t`
direction, `sample_steps`/`cfg` wiring, the 0.18215 latent scale, VAE decode, EMA weights.
2–3 malformed samples out of 16 is consistent with a 30M-active model at FID ≈ 15 — not a bug.

**⇒ Step 1.2 COMPLETE, day 1 of the 3-day budget. `DSMoE-B-E48` fallback not needed.**

### Snapshot gotcha

`kaggle datasets version -d` is `--delete-old-versions`, **not** dir-mode. First push silently
skipped `results/` and `patches/` ("Skipping folder: ...; use '--dir-mode'").
Correct invocation: `--dir-mode zip`. Check the upload list names every folder.

---

## 2026-07-31 — ✅ Step 1.3 COMPLETE. **Baseline FID-10k = 23.30**

`results/baseline/fid10k.json`:

```json
{"fid10k": 23.300021078466273, "n_images": 10000, "batch": 64, "seed_base": 0,
 "sampler": "RectifiedFlow.sample(mode=euler)", "steps": 25, "cfg": 1.5,
 "weights": "ckpt['ema'] of YHLLEO/DSMoE-S-E48 checkpoints/0700000.pt",
 "precision": "fp32", "decode_chunk": 8,
 "fid_impl": "pytorch-fid", "fid_impl_version": "0.3.0",
 "reference": "ADM VIRTUAL_imagenet256_labeled.npz (10k)",
 "gpu": "Tesla T4", "torch": "2.10.0+cu128", "pruned": false,
 "moe_blocks": [1,3,5,7,9,11], "wall_min": 89.3}
```

**Reading the gap to the paper (14.81):** theirs is FID-**50k** via the TF/ADM evaluator;
ours is FID-**10k** via `pytorch-fid`. FID is biased upward at smaller sample counts, and the
two implementations differ. A gap of ~8.5 is expected and healthy. Step 1.3 required the
number to be *stable*, not equal — it is, and `grid.png` independently confirms the sampler.

**⇒ 23.30 is the anchor. Every pruning result is reported as a delta from it.**

### 🔒 LOCKED FOR THE PROJECT — changing any of these invalidates all comparisons

| | |
|---|---|
| FID impl | `pytorch-fid` 0.3.0 |
| Reference stats | `fid_stats_imagenet256.npz` from ADM `VIRTUAL_imagenet256_labeled.npz` (10k) |
| Weights | `ckpt['ema']` — **never** `ckpt['model']` |
| Sampler | `RectifiedFlow.sample(mode='euler')`, 25 steps, CFG 1.5 |
| Precision | fp32 |
| GPU | T4 (sm_75+); P100 unusable |
| Seeding | `seed = seed_base * 1000003 + image_index` |

### Wall-clock note

89.3 min for FID-10k vs the 75 min predicted from the batch-64 timing — Inception activations
and per-chunk saves account for the difference. **Use ~90 min / 1.5 GPU-h per config** for
Phase 5 planning, not 75. Revised ceiling: **~20 configs/week**, not 24.

Kaggle's console stopped streaming output partway through the run while the kernel continued
normally. `act_partial.npy` mtime is the reliable liveness signal, not the console. Phase 2
sampling code should log every chunk with `flush=True`.

---

## Run log

---

## Run log

| Date | Run ID | Config | FID | Memory | Latency | Observation |
|---|---|---|---|---|---|---|
| 2026-07-31 | `baseline_unpruned_seed0` | DSMoE-S-E48 ema, N=48 k=5, 25 RF steps, CFG 1.5, fp32 | **23.30** (FID-10k) | 69.2M params / 10.76 GB peak | 0.4475 s/img (tf 0.3052 + vae 0.1423) | Reference anchor. Paper reports 14.81 @ FID-50k/ADM-TF — gap expected. |
