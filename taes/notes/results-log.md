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

## 2026-08-22 — ✅ Step 2.1 COMPLETE. Router telemetry hooks (`src/hooks.py`)

**Design, resolved from live source inspection (not guessed):**

- Confirmed `DSMoE.forward`: `router_logits = self.gate(hidden_states)` (raw, pre-sigmoid) →
  `topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)` → passed straight
  into `self.experts(hidden_states, topk_indices, topk_weights)`. Nothing is stored as an
  attribute, so `route_tokens_to_experts` itself never needed touching.
- **F3 resolved without monkeypatching.** A forward hook on `mlp.gate` captures raw
  `router_logits`; a forward **pre**-hook on `mlp.experts` receives `topk_indices` as a plain
  call argument (from `DSNaiveMoE.forward(self, hidden_states, top_k_index, top_k_weights)`).
  Pre-norm gate score recomputed as `router_logits.sigmoid().gather(1, topk_indices)` —
  mathematically identical to the internal `topk_weights` local variable *before* the
  `norm_topk_prob` block runs, confirmed against the pasted `route_tokens_to_experts` source.
- **F9 resolved as predicted (option a).** `DSNaiveMoE.forward` source confirms the gate weight
  (`top_k_weights[top_x, idx, None]`) multiplies `self.blocks[expert_idx](current_state)` —
  i.e. the expert `MLP`'s own output — so a forward hook on `mlp.experts.blocks[e]` directly is
  genuinely pre-multiplication. No correction factor needed.
- **F4 folded in.** `mlp.gate.e_score_correction_bias` dumped per layer at `register()` time,
  saved alongside telemetry.
- **Context contract, not per-token attribution.** Per-expert MLP hooks see only the routed row
  subset, no token identity — so `hooks.py` requires the caller to set
  `telemetry.set_context(domain_idx, t)` once before each forward pass, with every image in
  that call sharing domain and timestep. This matches how Step 3.1 calibration actually runs
  (one stratified `(image, t)` point — or a domain/bin-homogeneous batch — per forward call,
  not a full 25-step `rf.sample()` trajectory), so the contract costs nothing in practice.
- Accumulates **running sums only** (`freq`, `gate_sum`, `l2_sum`, `l2_count`) over
  `(domain, timestep_bin=50, layer=6, expert=48)` — no per-token dumps, per WORKOUT-PLAN 2.1.

**Smoke test (10 images, single direct `model(x, t, y)` call, domain=0, t=0.5):**

```
registered 300 hooks (6 gates + 6 expert-groups + 288 experts)
output shape: torch.Size([10, 8, 32, 32])
has NaNs: False
freq summed over experts, per layer: [12800]*6
l2_count summed over experts, per layer: [12800]*6   <- exact match to freq, every layer
t range seen: 0.5 0.5
```

`12800 = N(10) × T(256 tokens/image, 32×32 latent @ patch_size 2) × top_k(5)` — exact.
`freq == l2_count` in every layer confirms the router-side (F3) and expert-side (F9) hooks are
counting the identical set of routed tokens, with no double-count and no drop. **Output gate met.**

**Open item carried forward:** `T_MIN, T_MAX = 0.0, 1.0` in `hooks.py` is the assumed RF `t`
range for the 50-bin discretisation — untested beyond the single smoke value `t=0.5`. Verify
against real stratified values once Step 3.1's calibration sampler is written; adjust the
constants if RF actually runs on a different scale (e.g. integer-like `[0,1000]`, per the
`timestep_start/end` fields in `config_2025-11-08T22-26-38.yaml` noted in `backbone-survey.md`
§3.2, which conflicts with the "continuous t ∈ [0,1]" text in the same file).

---

## 2026-08-22 — ✅ Step 2.3 COMPLETE. Domain definitions (`configs/domains.json`)

Done ahead of Step 2.2 (out of WORKOUT-PLAN's written order) because latent caching needs the
class→domain mapping to know which images to cache — the plan lists 2.2 before 2.3, but 2.3 is
the actual dependency of 2.2, not the other way round.

**Image source:** Kaggle competition dataset `imagenet-object-localization-challenge` (official
ILSVRC2012 mirror), attached via Add Input → Competitions (required accepting the competition
rules once). Mounts at
`/kaggle/input/competitions/imagenet-object-localization-challenge`.
`LOC_synset_mapping.txt` — 1000 lines, line `i` = class index `i`'s WNID + label, confirmed
`n01440764 tench` at index 0, matching the standard DiT/ADM/torchvision convention (consistent
with Phase 1's `grid.png` already producing recognizable standard-ImageNet categories from
`torch.randint(0,1000)` labels). `train/<wnid>/` folders confirmed, ~1300 images/class.

**Class selection — keyword matching on synset labels, not an assumed index range.**
First attempt used plain substring search and was **wrong**: `"car"` matched inside Latin
binomial names (`carassius`, `carcharodon`, `carduelis`...), `"bus"` matched inside
`erythrocebus` — false-positived dozens of animal classes into the vehicle list. Fixed with
word-boundary regex (`\bkw\b`). Final: **0 overlap** between domains.

| Domain | Keyword matches | Selected | Selection method |
|---|---|---|---|
| animals  | 317 | **100** | `np.random.default_rng(0).choice(317, 100, replace=False)` |
| vehicles | 69  | **69**  | all matches used — see note below |

**⚠️ Vehicles cannot reach ~100 classes — ImageNet-1k does not contain that many.** 69 is the
real ceiling under a defensible keyword definition (broadened once: added `wing`,
`shopping cart`, `parachute` on top of the initial ~65-keyword list, +3 classes). ImageNet-1k is
heavily animal-skewed (120+ dog breeds alone) vs. a small, non-contiguous set of transportation
classes. **This is a hard dataset property, not a bug** — WORKOUT-PLAN 2.3 says "aim ~100," not
"require 100." Carrying forward as a named limitation for Step 6.4 (paper Limitations section):
*domain class-count is asymmetric (100 animal classes vs. 69 vehicle classes) because ImageNet-1k's
label taxonomy itself is animal-skewed.*

**Calibration images:** 500/domain, distributed as evenly as possible across each domain's
classes (`divmod` base + seeded-permutation remainder assignment), fixed seeds (animals=0,
vehicles=1), sampled without replacement from each class's train folder. Verified: 500/500,
correct class↔wnid↔path pairing (spot-checked `n01440764_31715` → class 0 tench;
`n02690373_1074` → class 404 airliner).

**Output:** `taes/configs/domains.json` (class lists) + `taes/configs/calibration_images.json`
(500+500 `{image_id, path, class_id, wnid, label}` records) written on Kaggle at
`/kaggle/working/taes/configs/`. Not yet pushed to the Dataset — will go out with the end-of-phase
snapshot (bootstrap doc §3).

**Then:** Step 2.2 — cache VAE latents for these 1000 calibration images.

---

## 2026-08-22 — ✅ Step 2.2 COMPLETE. VAE latent cache

Encoded all 1000 calibration images (500 animals + 500 vehicles from Step 2.3) through
`stabilityai/sd-vae-ft-mse` once. Preprocessing: `Resize(256) → CenterCrop(256) → ToTensor →
×2−1` (SD-VAE `[-1,1]` convention, matches the decode-side `/0.18215` scale already locked in
`SESSION-BOOTSTRAP.md`). Encoded with `posterior.mode() * 0.18215` — **deterministic**, not
`.sample()` — so calibration latents carry no VAE-encoder sampling noise; the only randomness
downstream will be RF forward-noising at Step 3.1, applied per stratified `t`.

```
[animals]  500/500  32.4 s
[vehicles] 500/500  29.6 s
animals  latents: torch.Size([500, 4, 32, 32])  8.2 MB
vehicles latents: torch.Size([500, 4, 32, 32])  8.2 MB
index entries: 1000
```

**Output:** `results/telemetry/latents/{animals,vehicles}.pt` (raw latent tensors) +
`results/telemetry/latents/index.json` mapping `image_id → {latent_path, latent_index, class_id,
domain}`, exactly the WORKOUT-PLAN 2.2 spec. **Size: 16.4 MB total** — trivial, no storage
concern, no reason to ever re-encode these before the project ends.

**⇒ Phase 2 (Instrumentation) COMPLETE — Steps 2.1, 2.3, 2.2 all done** (2.3 done before 2.2;
see note above). Next: Phase 3, Step 3.1 (run calibration through the model with hooks live).

---

## 2026-09-19 — ✅ PHASE 2 REDO COMPLETE (via Kaggle CLI, not notebooks)

**Why redone:** original Phase 2 (2026-08-22) was never mirrored off Kaggle and the Dataset copy
was lost. Redone in one session using `kaggle kernels push` (T4 ×2, script kernels) — the working
agreement changed from paste-a-cell to CLI-driven; see CLAUDE.md and SESSION-BOOTSTRAP §0b.
Order 2.1 → 2.3 → 2.2 as planned. Every gate below was read from the kernel log, not assumed.

### Step 2.1 ✅ `taes/src/hooks.py` — output gate reproduced exactly
```
registered 300 hooks (6 gates + 6 expert-groups + 288 experts)
e_score_correction_bias all-zero per layer: [True]*6          <- F4 reconfirmed
output shape: torch.Size([10, 8, 32, 32])    has NaNs: False
freq summed over experts, per layer:     [12800]*6
l2_count summed over experts, per layer: [12800]*6            freq == l2_count everywhere: True
```
Design as recorded (F3 pre-norm sigmoid via gate hook + experts pre-hook; F9 hook on expert `MLP`;
context contract `set_context(domain_idx, t)`). Written from source, not from the lost original, so
API details (accumulator shapes `(D=2, B=50, L=6, E=48)`, `state()`/`save()`) are new but the
semantics are the recorded ones. Mean pre-norm gate score per selected slot, layer 0, random
inputs, t=0.5: 0.535.

### F15 — ✅ RESOLVED: RF timestep range is `t ∈ [0,1]`
Source: `rectified_flow.py` — sampler uses `t = i / sample_steps`, training `t ~ U(0,1)`,
`zt = t·z1 + (1−t)·z0` (**t=0 is pure noise, t=1 is data**). The config's `timestep_start/end: 0/1000`
is not used by the RF path. `T_MIN, T_MAX = 0.0, 1.0` in `hooks.py` is correct — closes the open
item from the first Phase 2. **Bin 0 = noisiest, bin 49 = cleanest** — say so in every timestep figure.

### Step 2.3 ✅ `taes/configs/domains.json` + `calibration_images.json`
| | Matches | Selected |
|---|---|---|
| animals | **397** | 100 (`default_rng(0).choice`) |
| vehicles | **69** | all 69 (matches the original run's count exactly) |
Overlap 0. 500/500 images per domain, 100 / 69 classes used, class↔wnid↔path asserted per record.
Spot-check: `n01440764_31715` → class 0 tench (same record as the original run); `n02687172_12159` →
403 aircraft carrier.
- **Animal match count differs from the original (397 vs ~317)** — the original keyword list was
  never recorded. The new one reproduces ImageNet's animal index range 0–397 exactly (397 = all of
  it minus the ambiguous `crane`). Nothing had consumed the old draw. **Both keyword lists,
  exclusion lists, counts and seeds are now inside `domains.json`.**
- Word-boundary regex used throughout. Per-domain exclusion phrases were needed on top of keywords
  (e.g. `hot dog`, `coral reef`, `snake fence`, `car mirror`, `tank suit`, `garden cart`).
- **`crane` (classes 134 bird / 517 machine) is ambiguous and dropped from both domains.**
- Paper limitation §6.4 stands: 100 vs 69 classes (vehicles are a hard ImageNet ceiling).

### Step 2.2 ✅ latent cache
```
[animals]  500/500  30.1s  torch.Size([500, 4, 32, 32])  8.2 MB  mean=0.070 std=0.826
[vehicles] 500/500  30.7s  torch.Size([500, 4, 32, 32])  8.2 MB  mean=0.046 std=0.828
index entries: 1000
```
`posterior.mode() * 0.18215`, Resize(256)→CenterCrop(256)→ToTensor→×2−1. In the Dataset at
`results/telemetry/latents/`. **Never re-encode.**

### F16 — `kaggle kernels output` downloads all of `/kaggle/working`
The first bootstrap put the 1.1 GB checkpoint + repo clone in `/kaggle/working`; the output pull hung.
Bootstrap now uses `/tmp` for both. Pull selectively: `--file-pattern`.

### F17 — Dataset versioning by CLI can silently drop files
A new version's file set is exactly the folder pushed. **Always `datasets download --unzip` the current
version into a staging dir, add to it, then push** — otherwise `results/baseline/fid_stats_imagenet256.npz`
disappears. Also: the CLI builds a temp filename from the absolute path, so push from a **short path**
(e.g. `D:\tstage`); long scratchpad paths fail with `[Errno 2]`. Confirmed final listing has
`results/`, `patches/`, `taes-src/`, `taes-configs/`.

### F18 — ⚠️ credentials: the machine's `~/.kaggle/kaggle.json` belonged to another account
It authenticated as `zahidhussainlone`. Replaced with a `mohammedsarim` API token in the gitignored
project `.env` (`KAGGLE_API_TOKEN`), loaded per command with `set -a; . ./.env; set +a`. `kaggle.json`
left untouched.

## 2026-09-19 — ✅ Step 3.1 COMPLETE. Calibration telemetry (`kernels/p3-01-calibrate`)

Every one of the 500 latents per domain × every one of the 50 t-bins = 25,000 forwards/domain, ~2.7 min/domain
on T4 (far under the 3 h budget, so **full coverage, not a 10-image-per-bin partition**). Design:
- `zt = t·z1 + (1−t)·z0`, `t = (bin + U(0,1))/50` jittered inside the bin per image; `z0` seeded per (domain, bin) (`1000·di + b`); batch 100.
- **Conditional pass only, with the image's true class label; no CFG / no null-label half.** Sampling under CFG also routes the y=1000 half, which is domain-agnostic; it is *not* in these statistics. Decision recorded; revisit if Phase-5 FID looks off vs. the scores (F19 below).
- One `RouterTelemetry(n_domains=1)` per domain → `results/telemetry/{animals,vehicles}.pt` (shape `(1,50,6,48)`, domain_idx 0 in each file).

Gate (from log): `freq` per (bin,layer) = **640000** = 500×256×5 exactly, all bins/layers; `freq == l2_count` everywhere; `t_seen` 0.0028…0.9981 (whole [0,1]).
Mean pre-norm gate score: animals 0.476, vehicles 0.483.

### F19 — dead experts in animals
Animals: **1 expert with zero selections across all 50 bins in layers 4 and 5** (layers 0–3: none); vehicles: none. Never routed under calibration conditions (cond., 500 imgs) — free pruning candidates; check they aren't just domain-rare (they aren't picked by vehicles either only if same index — check in 3.2).

## 2026-09-19 — ✅ Step 3.2 COMPLETE. Banded importance (`taes/src/scoring.py`)

`I = freq × (gate_sum/freq) × (l2_sum/l2_count)`, normalised to sum 1 over experts per (domain, band, layer). Bands sum the **raw accumulators**
of the 50 fine bins, then form the ratios (never average ratios); edges `round(i·50/B)` (B=8 → uneven 6/7-bin bands). Band 0 = noisiest (F15).
Run locally on CPU (pure post-processing of the pulled telemetry) → `taes/results/scores/{animals,vehicles}_B{1,2,4,8}.pt`, each `{I:(B,L,E), B, edges, moe_blocks}`; all rows sum to 1.

**F19 resolved:** the never-routed experts are *domain-specific, not dead* — animals: layer 4 expert 33, layer 5 expert 21; vehicles route to both.
Not free-pruning candidates for a two-domain union. Quick B=1 look: top-5 overlap animals vs vehicles per layer = [4,3,4,3,5,5] of 5.

## 2026-09-19 — ✅ Step 3.3 COMPLETE. Diagnostic figures (`taes/src/figures.py` → `taes/results/figures/diagnostic_{1,2,3}.png`)

Run locally on the pulled telemetry. Fig 1: importance, expert (sorted by overall importance) × 50 bins, per layer/domain. Fig 2: Jaccard(S_b,S_b'),
B=8, k=24, plus mean off-diagonal Jaccard vs k with random baseline. Fig 3: |∪_b S_b|/N vs k, B=4 solid / B=8 dashed.

### F20 — banding is real and smooth in t (pre-GO/NO-GO reading; decision belongs to Phase 4)
- **Overlap (k=24, mean off-diag Jaccard; random = 0.333):** B=4 animals [.67 .57 .49 .58 .56 .64], vehicles [.67 .57 .51 .65 .63 .61];
  B=8 animals [.69 .61 .57 .62 .59 .61], vehicles [.72 .60 .57 .70 .66 .65]. Well below 1 and above chance. Structure is a **band-distance gradient**:
  adjacent bands ≈0.8–1.0, extreme bands (noise vs data) ≈0.1–0.3, i.e. *below* the random baseline in layers 1, 2, 4 — noise-end and data-end use different experts.
- **Fig 1:** importance is concentrated in ~10 experts/layer; individual experts ramp up toward the data end or fade toward it, plus a few sharp noise-end and data-end spikes. Smooth drift, not a step.
- **Union (k=24, N=48):** B=4 [.69 .75 .83 .75 .75 .71] / [.67 .77 .79 .69 .71 .73]; B=8 up to .85. **Union/N ≈ 0.7–0.85 ⇒ per-band subsets cover most of the pool ⇒ memory mode saves little at k=24;
  compute mode is the primary claim** (§5.4 rule). Memory mode only pays at small k (check fig 3 low-k end).
- Caveats: (i) no noise floor measured — tail experts at k=24 carry tiny importance, so tail-driven Jaccard is sampling-sensitive (a split-half-of-images null is needed before quoting Jaccard in the paper);
  (ii) animals and vehicles look alike in band structure, so this is timestep dependence, not domain dependence; (iii) calibration is conditional-only, no CFG (3.1 decision).

## 2026-09-19 — ✅ Step 3.4 + PHASE 3 COMPLETE
`taes/paper/main.tex` (article class, journal template TBD): Method section complete (setting, calibration, banded importance, selection, compute/memory modes, diagnostics) with the three figures (copied to `taes/paper/figures/`). **Not compiled** — no LaTeX on this machine; compile on Overleaf or install MiKTeX before relying on it.
Method text uses t∈[0,1] (F15); REFERENCE-GUIDE §5.1's "[0,1000]" is superseded. Dataset version pushed (`--dir-mode zip`, staged from current contents per F17): now holds `results/telemetry/{animals,vehicles}.pt`, `results/scores/*`, `results/figures/*`, `taes-src/{hooks,scoring,figures}.py`; fid_stats and latents intact. Note `kaggle datasets files` is paginated — use `--page-size 100`.
**Open for Phase 4 before quoting overlap numbers:** split-half noise-floor null for Jaccard (two independent calibration runs on disjoint 250-image halves / different noise seeds).

## 2026-09-19 — ✅ Step 4.1: noise-floor null (`kernels/p4-01-splithalf`, `taes/src/noisefloor.py`, `taes/results/figures/noisefloor.png`)

Per domain: two **disjoint 250-image halves** (perm seed 4000+di), independent noise seeds (10000+1000·di+100·h + bin), otherwise identical to Step 3.1 (cond., no CFG, 50 bins, jittered t). Plus one **unconditional** pass (y=1000, all 500 latents, seed 20000+…). All gates held (`freq` = n·256·5 per bin/layer, `freq == l2_count`). 85 s/half, 170 s/500-image pass on T4. Files in `results/telemetry_p4/`.
Floor = Jaccard(top-k half0 band b, top-k half1 band b). Across-band = same statistic between *different* bands, still half0-vs-half1 (both sides carry the same cross-half noise). Mean over 6 layers:

| domain, B | k | floor (same band) | extreme bands (0 vs B−1) | all off-diag | random |
|---|---|---|---|---|---|
| animals B4 | 24 | 0.90 | 0.36 | 0.57 | 0.33 |
| vehicles B4 | 24 | 0.94 | 0.41 | 0.61 | 0.33 |
| animals B8 | 24 | 0.90 | 0.30 | 0.59 | 0.33 |
| vehicles B8 | 24 | 0.93 | 0.36 | 0.65 | 0.33 |
| animals B4 | 12 | 0.92 | 0.24 | 0.48 | 0.14 |
| vehicles B4 | 12 | 0.94 | 0.22 | 0.44 | 0.14 |
| animals/vehicles B4 | 5 | 0.94 / 0.99 | 0.17 / 0.16 | 0.40 / 0.41 | 0.055 |

Extreme-band Jaccard < floor in **6/6 layers in every (domain, B, k) cell** (72/72). Gate met.

### F21 — the band structure is far above the noise floor
Sampling noise costs only ~0.06–0.10 Jaccard (floor 0.90–0.99 even at 250 images, so the 500-image numbers in F20 are, if anything, slightly cleaner). Distant bands share 0.16–0.41 depending on k (k=24: 0.30–0.41 vs chance 0.33 — i.e. roughly chance; k=12: 0.17–0.24 vs 0.14; k=5: 0.12–0.17 vs 0.055), so noise-end and data-end sets are close to independent draws. F20's smooth gradient stands; F20 caveat (i) is closed. Quote floor next to every Jaccard in the paper.

### F22 — mechanism / robustness
- **Load-balancing artefact: no.** `e_score_correction_bias` is all-zero (F4), so there is no aux-loss-free balancing bias that could impose or hide timestep structure; selection and weighting rank experts identically. (Not proof about the *training* loss, but no bias-term confound exists in the checkpoint.)
- **Depth:** effect present in all 6 layers (72/72 cells). Depth-dependent strength: extreme-band J at k=24 (B=4) is lowest in layer 2 (0.20 animals / 0.26 vehicles) and layer 1, highest in layer 0 (0.52 / 0.55); layer 0 is the most timestep-stable, mid layers the least.
- **Domain:** animals and vehicles agree (F20 caveat ii) — this is timestep dependence, not domain dependence.
- **CFG:** conditional-only calibration vs. unconditional (y=1000) routing, same band, B=4, mean over bands: k=24 J = 0.87–1.0 (animals), 0.89–0.98 (vehicles); k=12: 0.78–1.0 / 0.80–1.0; k=5: 0.73–1.0 / 0.75–1.0. Routing under the null label is similar but not identical, most different in layers 2–4 at small k. **Decision: no separate CFG-branch run.** Real sampling uses both branches; if Phase-5 FID at small k looks worse than the scores predict, the first thing to try is pooling cond+uncond telemetry (files exist: `{d}_uncond.pt`), not a new calibration.
- Caveat: floor uses 250-image halves, so it is *pessimistic* (noisier) relative to the 500-image scores.

## 2026-09-19 — 🟢 DECISION: **GO**

**Diagnostic:** `taes/results/figures/noisefloor.png` (floor vs adjacent vs extreme bands, B=4, k=12/24) and `diagnostic_2.png`.
Timestep-banded expert importance is real: the top-k sets of the noisiest and cleanest bands overlap at 0.12–0.41 (≈chance at k=24) while two independent 250-image calibrations of the *same* band overlap at 0.9–0.99, in every layer, both domains, B∈{4,8}, k∈{5,12,24}. The structure is a smooth band-distance gradient (adjacent bands ≈0.8–1.0), not a step. Hypothesis alive; proceed to Phase 5 with the plan's Step 5.2 order untouched. Whether the structure translates into better FID-vs-budget than a global subset is the Phase 5 question, and the honest outcome could still be "no better".

**Compute vs memory (union size |∪_b S_b|/N, mean over 6 layers, from the saved scores):**

| k | 5 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|
| B=4 animals / vehicles | 0.20 / 0.20 | 0.32 / 0.31 | 0.42 / 0.45 | 0.56 / 0.54 | 0.75 / 0.73 |
| B=8 animals / vehicles | 0.23 / 0.22 | 0.35 / 0.34 | 0.48 / 0.48 | 0.59 / 0.59 | 0.80 / 0.76 |

- **k=24 (priority-1 headline): compute mode.** Union 0.73–0.80 ⇒ physical slicing keeps ~75% of routed experts (~25% routed-param saving at best, before the shared-expert floor), while a global k=24 memory-mode subset saves 50%. TAES's k=24 claim is therefore about *quality at equal per-band active-expert budget*, not memory.
- **Memory mode pays at k ≲ 12:** union 0.42–0.48 ⇒ 52–58% routed-param reduction at k=12, ~68% at k=8, ~80% at k=5 (k=5 = router top-k, per-band sets are then a rigid assignment, expect heavy quality loss). Memory-mode claim, if any, lives in k∈{8,12,16}; the compare-against-global-at-equal-memory framing (B=4 union at k vs B=1 at k·union) must be used for the Pareto plot's x-axis (Step 5.3).
- The §5.4 "N≥32 ⇒ memory primary" rule is **not** borne out at the priority-1 k; record both modes, compute mode as headline at k=24, memory mode via the union-size x-axis across the k-sweep.

Run log addition below.

---

## Run log

| Date | Run ID | Config | FID | Memory | Latency | Observation |
|---|---|---|---|---|---|---|
| 2026-07-31 | `baseline_unpruned_seed0` | DSMoE-S-E48 ema, N=48 k=5, 25 RF steps, CFG 1.5, fp32 | **23.30** (FID-10k) | 69.2M params / 10.76 GB peak | 0.4475 s/img (tf 0.3052 + vae 0.1423) | Reference anchor. Paper reports 14.81 @ FID-50k/ADM-TF — gap expected. |
| 2026-09-19 | `p4_splithalf` | split-half (2×250) + uncond passes, animals & vehicles, cond., 50 bins | — | — | 85 s/half | Noise floor Jaccard 0.90–0.99 vs extreme-band 0.2–0.4, 72/72 cells ⇒ GO (F21/F22). |

## 2026-09-20 — ✅ Step 5.1 COMPLETE. Pruning (`taes/src/prune.py`, kernel `p5-01-prune`)

**Design.** `masks` = per-layer bool `(B, E_l)`. **Compute mode** (`Pruner`): model pre-hook derives the band from RF `t` (`int(t·50)` fine bin → `edges`, same binning as `RouterTelemetry`); gate forward-hook sets logits of disallowed experts to −inf ⇒ sigmoid 0 ⇒ router's top-5 is taken from the allowed set only. **Memory mode** (`slice_experts`): keep union_b S_b per layer, slice expert `ModuleList`, router rows, `e_score_correction_bias`, `num_experts`/`n_routed_experts`; shared expert untouched; masks re-indexed.
**Decision — k vs top-5:** the k allowed experts are the candidate pool; the router still picks its usual 5 of them by gate score (weights renormalised over the 5 as before). Hard floor k ≥ 5 (asserted). `n_group=topk_group=1` in this config, so group routing is a no-op.

**Gate results (k=24, B=4, animals; T4):**
- all-ones mask ≡ unpruned model, bit-exact.
- Full 25-step CFG sampling, 16 imgs, telemetry: **masked-expert activations = 0** (allowed = 6,144,000; every (bin, layer) still routes exactly 5 experts/token). Negative control (unmasked): 390,598 out-of-mask activations ⇒ the check has power.
- Memory mode kept experts/layer = [33,36,40,36,36,34] → **routed fraction 0.747**, routed params 31.70M of 42.47M, **state_dict 264.5 → 223.3 MB**. Max |compute − memory| output = **0.0** (t ∈ 0…0.96).
- Pruned vs unpruned latent rel. change 0.160 (same seed); grid in `kernels/p5-01-prune/out/results/p5_01/`.

### F23 — memory mode is functionally identical to compute mode with the same masks
Dropped experts are in no band's set, so they never fire ⇒ one FID run per (B,k,domain) gives both modes' quality; memory mode only changes the size axis (union). Halves Phase-5 cost — the k-sweep needs one run per config, not two.

### Open before priority-1 (needs a decision)
Phase-1 baseline FID (23.30) samples **all 1000 classes** vs full-ImageNet stats. Pruning is domain-calibrated, so evaluating a domain needs domain-class generation **and a domain reference** (ImageNet-train images of the domain's classes, minus the 500 calibration images) — full-ImageNet stats would measure domain shift, not pruning damage. This means new reference stats (not a regeneration of the locked ones) plus an unpruned per-domain baseline (+2 runs ≈ 3 GPU-h). The exact Phase-1 sampling loop (per-chunk seeding) also exists only in the old notebook; `sample.py`/`fid.py` still to be rebuilt.

## 2026-09-20 — ✅ Step 5.2a: domain FID references (`kernels/p5-02-domainref`, `taes/src/fid.py`, `taes/src/sample.py` written)
Per domain: 10,000 real ImageNet-**train** images, stratified equal per class (100 animal / 69 vehicle classes), the 500 calibration images excluded, ADM `center_crop_arr` to 256, uint8 → `pytorch-fid` pool3 (same Inception path as the locked stats). Stats: `results/fid_ref/{animals,vehicles}_train10k.npz` (pulled to `kernels/p5-02-domainref/out/results/fid_ref/`; **not yet in the Dataset**).
Gates: real-vs-real interleaved 5k/5k FID = **8.21 animals / 7.45 vehicles** (the sampling-noise scale of the reference; 10k-vs-10k will be lower). Preprocessing check: 10 real train imgs/class × 1000 classes vs the locked ADM stats = **7.27** (same scale ⇒ pipeline matches the ADM reference; not 0 because different images). First attempt used a class-ordered split (FID 100) — test bug, not a pipeline bug.
Domain-FID absolute values are NOT comparable with 23.30; every domain result is reported vs its own unpruned baseline.
`sample.py::run_config` = Phase-1 sampler re-created (seed = seed_base·1000003 + chunk start, z then y from one generator, y ∈ domain classes or all 1000 when `cls=None`); uint8 conversion `round((x+1)·127.5)` is an assumption about the Phase-1 notebook — checked by the full-class reproduction below.
**Next (5.2b):** unpruned baselines: full-class FID-10k (must reproduce 23.30 — validates sampler + uint8 path; 1.5 GPU-h) + animals + vehicles domain baselines (3 GPU-h), as parallel kernels.

## 2026-09-20 — ✅ Step 5.2b: unpruned baselines (`kernels/p5-03-base-{full,animals,vehicles}`)

| Run ID | Classes | Reference | FID-10k | Wall |
|---|---|---|---|---|
| `unpruned_full_seed0` | all 1000 | locked ADM stats | **23.310** (Phase 1: 23.300) | 75.5 min |
| `unpruned_animals_seed0` | 100 animal | animals_train10k | **19.141** | ~75 min |
| `unpruned_vehicles_seed0` | 69 vehicle | vehicles_train10k | **24.279** | ~75 min |

### F24 — baselines and infrastructure
- Rebuilt sampler reproduces the Phase-1 anchor to 0.01 (23.310 vs 23.300) ⇒ seeding, uint8 conversion (`round((x+1)·127.5)`), and `pytorch-fid` path confirmed. Assumptions in `sample.py` are now validated.
- **Domain FIDs are on their own scales** (own reference, own class set): every domain pruned result is reported as Δ from its baseline (animals 19.141, vehicles 24.279). Sample grids show correct class content (animals: birds/dogs/primates/fish; vehicles: tank, boat, bus, balloon, parachute).
- **Kaggle mount paths vary per container**: one container mounted the Dataset at `/kaggle/input/taes-artifacts` (and ImageNet at `/kaggle/input/imagenet-object-localization-challenge`), crashing the vehicles kernel on a missing zip. `_bootstrap.py` now probes both layouts for `DS` and `IMAGENET_ROOT`.
- Kaggle allows **2 concurrent batch GPU sessions**; a third push is refused ("Maximum batch GPU session count of 2 reached").
- `run.py` only prints the log at completion; liveness = `kaggle kernels status`.

| Date | Run ID | Config | FID | Memory | Latency | Observation |
|---|---|---|---|---|---|---|
| 2026-09-20 | `unpruned_full_seed0` | sampler re-created, all classes | 23.310 | 3.1 GB peak (bs 64, allocated) | 3058 s sampling / 10k | Reproduces 23.300. |
| 2026-09-20 | `unpruned_animals_seed0` | 100 animal classes vs animals ref | 19.141 | — | — | Domain baseline. |
| 2026-09-20 | `unpruned_vehicles_seed0` | 69 vehicle classes vs vehicles ref | 24.279 | — | — | Domain baseline. |

## 2026-10-02 — Step 5.2 priority 1 (3/4): B=1 vs B=4, k=24

| Run ID | FID-10k | Δ vs baseline | Union frac | Sanity |
|---|---|---|---|---|
| `compute_animals_B1_k24_seed0` | 22.221 | +3.080 (19.141) | 0.500 | n=10000, wall ~77min, grid ok |
| `compute_animals_B4_k24_seed0` | **21.277** | **+2.136** (19.141) | 0.747 | n=10000, wall 77.2min, grid matches baseline content, ok |
| `compute_vehicles_B4_k24_seed0` | 26.621 | +2.342 (24.279) | 0.726 | n=10000, wall 83.7min, from a prematurely-launched run (chain script bug) — not yet grid-checked |
| `compute_vehicles_B1_k24_seed0` | pending | — | 0.500 (by construction) | — |

### F25 — TAES beats global at equal k, but not at equal memory
Animals, k=24: TAES (B4) FID 21.277 < global (B1) FID 22.221 — closes ~31% of the degradation gap from baseline (19.141). Real, modest win in the "quality at equal candidate-pool-size k" framing. **Not a memory-equal comparison**: B4's union is 0.747 of routed experts vs B1's exact 0.500 — TAES spends 50% more experts than B1 to get there. A fair memory claim needs global B1 run at k≈36 (0.747·48) to match TAES's actual footprint; not yet run. This is the headline caveat for the paper: k=24 supports a compute/candidate-pool claim, not yet a memory claim (consistent with the 2026-09-19 GO decision's compute-vs-memory split).

### F26 — Kaggle reliability notes (see SESSION-BOOTSTRAP §6.10-13)
`animals_B4_k24` hung `RUNNING` with no progress for ~6h on two separate launches (identical config, masks checked clean — no NaNs, exactly k=24/band, structurally identical to vehicles_B4 which completed normally both times it ran). Third launch completed normally (77.2 min). No code-level cause found; most likely Kaggle GPU-queue contention — unrelated kernels in this same window also sat `QUEUED` for hours. Mitigation adopted going forward: fast canary run (n≈500) before each full FID-10k commit, and sanity-check every `COMPLETE` result (n, wall time, sample grid) before logging — "COMPLETE" alone is not sufficient evidence of a good run.

## 2026-10-02 — ✅ Step 5.2 priority 1 COMPLETE: B=1 vs B=4, k=24, both domains

| Run ID | FID-10k | Δ vs baseline | Union frac | Wall | Sanity |
|---|---|---|---|---|---|
| `compute_animals_B1_k24_seed0` | 22.221 | +3.080 (19.141) | 0.500 | 77.2min equiv (prior entry) | ok |
| `compute_animals_B4_k24_seed0` | 21.277 | +2.136 (19.141) | 0.747 | 77.2min | ok |
| `compute_vehicles_B1_k24_seed0` | 27.644 | +3.365 (24.279) | 0.500 | 78.7min | ok — grid correct vehicle classes |
| `compute_vehicles_B4_k24_seed0` | 26.621 | +2.342 (24.279) | 0.726 | 83.7min | ok (recovered run, F25) |

### F25 (extended) — replicates across both domains
TAES (B4) beats global (B1) at equal k=24 in **both** domains, by a similar margin: animals 21.277 vs 22.221 (Δ 0.944, closes 31% of the degradation gap), vehicles 26.621 vs 27.644 (Δ 1.023, closes 30%). Consistent with F22 (animals ≈ vehicles). **Still not a memory-equal comparison** — B4 keeps 0.726–0.747 of routed experts vs B1's exact 0.500. Headline claim supported by this data: **banding improves quality at equal candidate-pool size k** (a compute/pool-restriction framing), not yet a memory-efficiency claim. Open: global B1 at k≈35–36 (matching TAES's actual union) for a true equal-memory test — candidate for priority 2's k-sweep, or a dedicated extra point.

**Priority 1 (headline) gate: passed — the core hypothesis predicts a real, replicated effect, not noise.** Proceed to Step 5.2 priority 2 (k-sweep, k ∈ {5,8,12,16,24,32,48}) per the unchanged WORKOUT-PLAN order.

## 2026-10-03 — Step 5.2 priority 2 started (animals only; one-domain cut). Kernel `p5-05-sweep` (parametrised; build with extras `taes/src/fid.py prune.py sample.py`)
Cut: animals only, B∈{1,4}, k∈{5,8,12,16,32} + B1 k=36. k=48 skipped (all-ones mask ≡ unpruned = 19.141 by construction). Order: B1k36, B4k8, B1k16, B4k12, B1k12, B4k5, B1k5, B4k16, B4k32, B1k8, B1k32. Canary (n=500) before every full run.
Union fractions (animals, local from scores): B1 = k/48; B4 k=5/8/12/16/24/32 → 0.205/0.319/0.424/0.559/0.747/0.868.

| Run ID | FID-10k | Δ vs 19.141 | Union frac | Sanity |
|---|---|---|---|---|
| `compute_animals_B1_k36_seed0` | **19.654** | **+0.513** | 0.750 | n=10000, wall 75.8min, grid ok |

### F27 — at equal memory, global beats TAES-k24 by a wide margin
Global B1 at k=36 (union 0.750) FID 19.654 (+0.51) vs TAES B4 k=24 (union 0.747) FID 21.277 (+2.14). At matched memory the global pruner is ~1.6 FID **better**; F25's equal-k win is a pool-size effect, not a memory win. The memory claim for TAES at this operating point is not supported. Remaining sweep decides whether banding wins anywhere on the memory axis (low k, where B4 union ≈ 0.32–0.42 vs B1 interpolated at k≈16–20).

### Handoff 2026-10-03 ~19:50 IST (pause until next morning)
In flight on Kaggle, nothing further to be launched tonight:
- `mohammedsarim/taes-p5-05-sweep` = **full** animals B4 k=8 (n=10000), pushed 13:14 UTC, expect done ~14:35 UTC. Pull: `kaggle kernels output mohammedsarim/taes-p5-05-sweep -p kernels/p5-05-sweep/out --file-pattern "result\.json|samples16\.png" -o`; sanity-check n, wall (~75–85 min), grid; then log.
- `mohammedsarim/taes-p5-06-sweep` = **canary** animals B1 k=16 (n=500), second concurrent lane (kernels/p5-06-sweep, same step.py with its own slug). If it passed (check log for `[union] routed fraction = 0.333`), set N=10000 in `kernels/p5-06-sweep/step.py` and launch the full run.
Remaining queue after those: B4k12, B1k12, B4k5, B1k5, B4k16, B4k32, B1k8, B1k32, then priorities 3–6. Two lanes (p5-05-sweep, p5-06-sweep) can run concurrently; canary before each full run; pass extras `taes/src/fid.py taes/src/prune.py taes/src/sample.py` to run.py.

| `compute_animals_B4_k8_seed0` | 46.570 | +27.429 (19.141) | 0.319 | n=10000, wall 70.6min, grid ok (same compositions, degraded texture) |
| `compute_animals_B1_k16_seed0` | 29.153 | +10.012 | 0.333 | n=10000, wall 79.1min, grid ok |
| `compute_animals_B4_k12_seed0` | 31.735 | +12.594 | 0.424 | n=10000, wall 82.6min, grid ok |

### F28 — global beats TAES at matched memory at low k too
B1 k=16 (union 0.333): 29.15. B4 k=8 (union 0.319): 46.57. B4 k=12 (union 0.424): 31.74. Global with *less* memory (0.333) beats TAES with more (0.424), and beats TAES at matched memory (0.333 vs 0.319) by ~17 FID. With F27 (k=24/k=36) that is two memory-matched points, high and low, both lost by TAES. Banding's only win so far is equal-k (F25), a pool-size effect that costs memory. Note B4's union at fixed k is large (band sets differ), which is what kills it on the memory axis.
Process: `run.py` poller crashed (IndexError on an empty `kaggle kernels status` reply) while both kernels completed fine; pulled manually with `kaggle kernels output`. Two concurrent lanes (`p5-05-sweep`, `p5-06-sweep`) worked.
| `compute_animals_B8_k24_seed0` | 21.522 | +2.381 | 0.795 | n=10000, wall 73.8min, grid ok |
| `compute_animals_B2_k24_seed0` | 21.747 | +2.606 | 0.663 | n=10000, wall 85.8min, grid ok |

### F29 — band count B∈{2,4,8} at k=24 does not beat global at matched memory
Animals, k=24: B2 21.747 (union 0.663), B4 21.277 (0.747), B8 21.522 (0.795), vs B1 22.221 (0.500). All banded variants land at ~21.3–21.7, within ~0.5 of each other (no monotone trend in B; single seed, so differences below ~0.5 are inside plausible noise), while paying 0.16–0.30 more union than B1 k=24.
Matched-memory read: B1 k=36 (0.750) = 19.654 vs B4 (0.747) 21.277; B1 k=32 (0.667) not yet run — linear interpolation between B1 k=24 and k=36 gives ≈20.5 at 0.667 (FID is convex in k, so the true value is likely ≤20.5) vs B2 21.747 ⇒ global ahead by ≥~1.2. **Interpolated, not measured — run B1 k=32 to confirm.**
Verdict so far (F27–F29): across B∈{2,4,8} and k∈{8,12,24}, no banded config beats global at matched memory; banding only wins at equal k. Not tuning chasing a result; the honest framing is "banding helps equal-k quality, not memory" unless priority 3–6 or another domain says otherwise.

### Handoff 2026-10-03 23:25 IST (end of tonight's session)
Nothing in flight. Both lanes idle. Completed tonight: B1k36, B4k8, B1k16, B4k12, B2k24, B8k24 (all animals, all logged + committed). Queue for tomorrow, in order: **B1 k=32** (confirm F29 matched-memory interpolation), then priority 3 (random removal, global k=24 and k=16), priority 4 (frequency-only, B1 and B4 k=24), remaining k-sweep points (B4 k=5, B1 k=5, B1 k=12, B4 k=16, B4 k=32, B1 k=8) as quota allows, priority 6 optional. Two lanes via `kernels/p5-05-sweep` / `p5-06-sweep` (edit CONFIG line in step.py; canary N=500 first; extras `taes/src/fid.py taes/src/prune.py taes/src/sample.py`). Note: `run.py` poller can crash on an empty status reply while the kernel is fine — pull manually with `kaggle kernels output <slug> -p <dir>/out --file-pattern "result\.json|samples16\.png|\.log$" -o`. Random removal and frequency-only need new code (not in `prune.py`/`scoring.py` yet).

## 2026-10-04 — Decision point: method framing, pre-registered kill criterion (written BEFORE the runs)
**New framing (post-hoc, disclosed as such):** F27–F29 killed the *union-memory* framing. A second deployment exists: batched sampling uses one timestep per batch (`Pruner.band_of` reads `t.flatten()[0]`; masked-expert activations were 0 over full trajectories, so t is shared), so only the **current band's k experts need be GPU-resident**; the rest sit in CPU RAM and are swapped in at the B−1 band boundaries. Peak resident routed memory is then **k/48 for both B1 and B4**, so the equal-k comparison (F25) *is* the equal-resident-memory comparison. This is a reframing after a negative result and must be reported as such; it is only credible if (i) the equal-k win holds at low k, where memory matters, and (ii) the swap overhead and peak memory are measured.
**Kill criterion:** run B1 k=12 and B4 k=16 (pairing existing B4 k=12 = 31.735 and B1 k=16 = 29.153), then B1 k=8 (pairs B4 k=8 = 46.570). If B4 does **not** beat B1 at equal k by **> 0.5 FID** at *both* k=12 and k=16, the method framing is dead: stop GPU use, write up as analysis / arXiv only. If it holds, run the measurement kernel (peak GPU memory + swap wall-clock, n≈64) and consider a second backbone (`DSMoE-B-E48` is released; L-E48 too large for quota). Do not move this threshold after seeing the data.
**Venue honesty:** even if positive, a 69M model is too small for anyone to need offloading; IEEE Transactions tier (TNNLS/TIP/TCSVT) almost certainly needs a second, larger backbone. Without it, IEEE Access is the realistic IEEE target.

## 2026-10-04 — External review (3 independent verdicts) reconciled
**Unanimous:** no-go as a method/compression paper (equal-memory data refutes it; per-band residency is a post-hoc rescue, no measurement, wrong model scale; correct baseline would be lossless offloading of the *unpruned* model). Conditional-go only as a soundness-first **analysis / negative-result paper**, TMLR the best fit; TPAMI/IJCV/JMLR reject. Mid-tier (Neural Networks, Neurocomputing, Pattern Recognition; IEEE TNNLS rated reject→major-revision, one verdict more optimistic) only after baselines + seeds + a second backbone. No verdict named IEEE Access; that is my own addition as the realistic IEEE fallback.
**Mandatory before any venue:** random-removal floor, frequency-only (+REAP-style) scoring, seed σ, measured B1 k=32, second backbone (DSMoE-B-E48), FID-50k on a few headline configs, real memory/latency measurement if any efficiency claim stays. Skip: DERN merging, quantisation, caching.
**Checked by me (web search 2026-10-04):** MoECa (arXiv 2606.15615, Jun 2026) is real — training-free expert-*branch caching* across timesteps in DiT-MoE, up to 2.83× speedup: the closest post-hoc/DiT-MoE/timestep-axis competitor and it has a real speedup. MosaicDiff (ICCV 2025, arXiv 2510.11962) is real — training-free timestep-staged pruning of *dense* DiT/SDXL. Diff-ES (arXiv 2603.05105) appears on alphaxiv. **"DIET" (reported by one verdict as a Sep-2026 training-free video-diffusion expert-pruning paper) could not be found — unverified, do not cite.** Timestep-dependent expert capacity exists in trained diffusion LMs; training-free dynamic top-k exists for LLMs (Elbow routing, LExI) — so post-hoc timestep-dependent active-k on an *image* diffusion MoE looks open but adjacent work is dense.
**CFG/uncond concern (validity):** calibration is conditional-only; Phase 4 measured cond-vs-uncond Jaccard 0.73–1.0 (worst layers 2–4 at small k) and recorded pooled cond+uncond calibration as the fallback; never tested in FID. Worth one run at the end.
**Amendment to the pre-registered kill criterion (written before k=12/k=16 results exist):** the >0.5 FID threshold stands, but a margin only counts if it also exceeds 2σ of seed noise; σ comes from 2 extra generation seeds (seed_base 1, 2) of B1 k=24 and B4 k=24. If the margin is <2σ the result is treated as null.
**The one flip experiment, from two verdicts (new method direction, not yet started):** per-band reduced *active* top-k (router top-k 5→3/4 at bands that tolerate it, renormalise) vs uniform top-k reduction at equal total activations. Success = ≥30% fewer active expert activations at ≤0.5 FID cost *and* beating the uniform-top-k baseline. This gives a real FLOPs benefit, unlike the current method. Needs new code (gate top-k hook; DSMoE source is on Kaggle only — clone locally to read `TopkRouter`).
**Cost correction:** one verdict priced DSMoE-B-E48 at 1.4 GPU-h/run (7 runs ≈ 9.8 h); another estimated 3–4× S-model cost. Treat B-E48 as ~4–5 h/run until measured.

### F30 — active-top-k route has a hard FLOP ceiling of 18% on this backbone (measured 2026-10-04, CPU, random weights from the real config, `torch.utils.flop_counter`)
Full-forward FLOPs, DSMoE-S-E48 (batch 2, no CFG): top-5 20.108 G; top-4 −4.5%; **top-3 −9.0%**; top-2 −13.5%; **top-1 −18.0%** (the ceiling: one routed expert everywhere). Routed experts are ~22% of per-token FLOPs; attention, the dense blocks and the shared expert dominate. Sampling wall-clock is further diluted by the VAE decode (0.142 of 0.4475 s/img), so a top-3 schedule is ≈ 6% of transformer time and ≈ 4% end-to-end, before any FID cost. DSMoE.top_k is a plain attribute read at call time and topk_weights are renormalised to routed_scaling_factor (2.5), so the schedule would have been a one-line hook — but the ceiling makes it not worth a journal claim, and MoECa (arXiv 2606.15615) already reports up to 2.83× training-free speedup on DiT-MoE.
**Consequence:** with the union-memory framing refuted (F27–F29), per-band residency unmeasurable on a model that needs no offloading, and active-k capped at ≤18% FLOPs, no remaining TAES variant on DSMoE-S gives an efficiency result a journal would call real. Decision pending with Sarim; recommendation: no further GPU spend beyond the two runs already in flight.

## 2026-10-04 — DECISION: target TMLR (empirical study); pivot gate shelved
Sarim chose TMLR after the three external verdicts and the F30 FLOP ceiling. TMLR policy checked directly (jmlr.org/tmlr acceptance-criteria + editorial-policies, fetched 2026-10-04): acceptance = technical soundness with "accurate and convincing evidence" + audience interest; novelty of the method explicitly not required; modest contributions acceptable; reproducibility studies welcome (Reproducibility Certification for added baselines/analysis/ablations); rejects papers that wrongly claim novelty. Negative results are not explicitly addressed (not forbidden).
**Paper claim, scaled to the evidence:** timestep bands in a diffusion MoE have distinct, reproducible expert preferences (split-half Jaccard 0.90–0.99 vs 0.12–0.41), banding helps at equal candidate-pool size k, but post-hoc banded pruning does not beat global pruning at equal memory (union accounting). Per-band residency is a clearly labelled secondary analysis, not the headline (post-hoc, unmeasured, model too small to need it). Related work must cover MoECa (2606.15615), MosaicDiff (ICCV 2025), DiffPruning, ALTER, Remix-DiT, REAP, EASY-EP, DERN, OLED-MoE — each verified before citing; do not cite "DIET".
**Kill criterion re-purposed:** B1 k=12 / B4 k=16 (in flight) now feed the equal-k low-k section; the >0.5 FID and 2σ rule still defines whether an equal-k advantage is claimed as real.
**Experiment plan (≈70 GPU-h ≈ 2.5–3 weeks of free quota; quota is per account, so two lanes do not add hours):**
Week 1: (1) gen seeds 1,2 for B1 k=24, B4 k=24, B1 k=36 (6 runs, ~8 h) → σ; (2) measured B1 k=32 (1.4 h); (3) random removal global k=24 and k=16 (2 runs, new mask builder) and frequency-only B1/B4 k=24 (2 runs, new scoring path); (4) CFG-pooled calibration (cond+uncond telemetry) B4 k=24 (1 run; telemetry files exist); (5) one alternate-calibration-set resample for B1/B4 k=24 (new 500-image telemetry + scoring, 2 runs).
Week 2: second backbone DSMoE-B-E48 — telemetry, split-half Jaccard, scores, then B1 vs B4 at 2–3 configs (cost per run to be measured; budget 4–5 GPU-h); vehicles replication of the k=16 and k=36 pairs (4 runs).
Week 3: FID-50k on 2–3 headline configs (≈7 GPU-h each); peak-memory measurement of physical slicing; LaTeX (TMLR template) and figures; verify every citation.
Cuts if quota runs short, in order: vehicles replication, alternate-calibration resample, FID-50k on a third config. Never cut: seeds, B1 k=32, random + frequency-only, second backbone.

## 2026-10-04 — Week-1 baseline machinery built (CPU only; no GPU spent)
- `scoring.importance(state, B, terms)` gains `terms ∈ {"full","freq","reap"}`; default unchanged (asserted equal to the stored score files). `reap` = mean gate × mean l2 without frequency = REAP-style (REAP averages over tokens routed to the expert; our telemetry gives exactly that mean).
- `taes/src/baselines.py` generates `taes/src/baseline_scores.py` (embedded tensors, no Dataset push needed; pass as an extra to `run.py`): `<domain>_B{1,4}_{freq,reap}` and `animals_B1_rand_s{0,1,2}`. Checked: shapes (B,6,48), exactly k allowed per band at k=24/12/5, random seeds independent (pairwise Jaccard 0.30/0.37/0.33 vs 0.33 expected).
- Lane `step.py` now takes `SCORE` and `SEED` in its CONFIG line (`DOMAIN, B, K, N, SCORE, SEED`); run id keeps the legacy name for std/seed0, else `..._k{K}_{SCORE}_seed{SEED}`. Launch extras now: `taes/src/fid.py taes/src/prune.py taes/src/sample.py taes/src/baseline_scores.py`.
- Selected-set overlap with the full TAES score (mean Jaccard over layers, top-24 / top-12): freq-only 0.86/0.85 (animals B1), 0.83/0.82 (B4), vehicles 0.86/0.81 (B1), 0.88/0.82 (B4); REAP-style 0.32/0.15 (animals B1), 0.38/0.14 (B4), vehicles 0.40/0.17 (B1), 0.46/0.18 (B4). Freq-only differs little from the full score (expect a small FID difference); REAP-style selects very different experts (expect a clear difference, direction unknown). These are set overlaps, not results.
- CFG-pooled calibration scores added (`<domain>_B{1,4}_cfgpool`: cond + uncond raw sums pooled 1:1, from `kernels/p3-01-calibrate/out/.../telemetry` + `kernels/p4-01-splithalf/out/.../telemetry_p4`). `baselines.py` now takes `<telemetry_dir> <uncond_dir>`; 15 tensors. Selected-set Jaccard vs the standard scores (k24/k12): animals B1 0.95/0.97, B4 0.96/0.94; vehicles B1 0.99/0.90, B4 0.96/0.95 — expect a very small FID change; one run (animals B4 k=24, SCORE="cfgpool") answers the reviewer's CFG-validity point.

### H1 (hypothesis, not a finding) — kept importance mass predicts FID degradation across B (2026-10-04, CPU only)
Kept mass M = mean over (band, layer) of the importance mass inside the selected set (scores are normalised per row; computed from calibration telemetry only, no sampling). Animals, 8 FID-10k configs: Spearman(M, ΔFID) = −0.976 (union −0.905, k −0.939). Ordered: B4k8 M=0.785 ΔFID +27.4 | B4k12 0.861 +12.6 | B1k16 0.868 +10.0 | B1k24 0.934 +3.08 | B2k24 0.955 +2.61 | B4k24 0.959 +2.14 | B8k24 0.960 +2.38 | B1k36 0.983 +0.51. Vehicles: B1k24 0.949 +3.36, B4k24 0.971 +2.34. Reading: at equal k banding keeps more mass (0.959 vs 0.934) — consistent with its equal-k win — and it pays for that mass with union.
**Cautions:** n=8, in-sample, k is itself monotone with M and ΔFID (k alone gives Spearman −0.94); a rough log-interpolation of the three B1 points under-predicts the banded configs' ΔFID at M≈0.96 (predicted ~1.2–1.4 vs actual 2.1–2.6), i.e. banded may lose ~1 FID at equal M — unresolved with only three B1 points. **Test:** the pending B1 k=12 / k=32 and B4 k=16 runs, then a held-out fit (fit f(M) on B1 points, predict the banded points). Only promote to a finding if it predicts held-out points better than k alone.

## 2026-10-04 — Fact-check of a fourth AI's "analysis contribution" write-up (Consensus search)
Agreed with it: TMLR judges soundness + "interest", not method novelty (policy pages fetched); shape "X is real but doesn't matter for Y"; a negative/corrective result needs fair reproduction, boundary conditions and a constructive lesson.
**Corrections (do not copy these into the paper):**
1. **"First quantitative evidence that a pretrained diffusion MoE has timestep-banded preferences" is FALSE as stated.** DiT-MoE (arXiv 2407.11633, Fei et al. 2024, 65 cites) already reports that expert selection prefers denoising timestep and that specialisation is more concentrated early and more uniform after half. Our real contribution on this axis is narrower: a split-half noise-floor validation (Jaccard 0.90–0.99 vs 0.12–0.41), band-resolved structure on a *pretrained, fine-grained* DSMoE (48 routed + shared expert), domain invariance (animals ≈ vehicles), and replication on a second backbone. DiT-MoE must be cited and the claim worded as extending/quantifying it.
2. **Its Findings 2 and 3 are one result stated twice** (equal-k advantage vs equal-union reversal). Count it once.
3. **Straw-man risk:** "the field assumes timestep specialisation implies pruning efficiency" needs named sources. Closest: DiffPruning, ALTER, Remix-DiT, MosaicDiff build timestep-specific subnets but fine-tune or prune dense nets; none claims post-hoc *expert* banding pays. State precisely what is being tested.
4. **"Report union, not k" holds only under static residency.** Under per-band offload, equal-k *is* equal peak memory. The recommendation must be scoped to its metric assumption, otherwise it is wrong for offloaded deployments.
5. **"Four TMLR-shaped contribution types" table: not on the policy pages I fetched** (acceptance criteria, editorial policies). Unverified; do not cite as TMLR policy unless found on a TMLR page.
6. **"DIET-style baseline": DIET could not be found; drop.** Use REAP-style (verified) and frequency-only.
7. It lists "one backbone" as an acceptable limitation; the three external verdicts treat that as a major reviewer objection — the second backbone (DSMoE-B-E48) stays in the plan. It also omits the CFG check, FID-50k, the F30 FLOP ceiling and H1 (kept-mass diagnostic), which are our extra analysis content.
Related work to add from the search: DiT-MoE (2407.11633), Switch-DiT (2403.09176), DiffMoE (2503.14487), Race-DiT (2503.16057), ProMoE (2510.24711), EC routing for diffusion LMs (2604.01622), TEAM (2602.08404); verify each before citing.

## 2026-10-04 — B1 k=12 and B4 k=16 complete (kill-criterion runs); H1 held-out test
| Run ID | FID-10k | Δ vs 19.141 | Union | Sanity |
|---|---|---|---|---|
| `compute_animals_B1_k12_seed0` | 36.736 | +17.595 | 0.250 | n=10000, wall 71.4min, grid ok (more degraded than k=16) |
| `compute_animals_B4_k16_seed0` | 25.779 | +6.638 | 0.559 | n=10000, wall 76.6min, grid ok |
(`run.py` poller survived this time; pulled with `kaggle kernels output`.)

### F31 — the equal-k advantage of banding GROWS as k shrinks (pre-registered kill criterion: PASSED)
Animals, B1 (global) minus B4 (TAES) at equal k: k=24 **+0.94**, k=16 **+3.37** (29.153 vs 25.779), k=12 **+5.00** (36.736 vs 31.735). Pre-registered rule (written before these runs): B4 must beat B1 at equal k by >0.5 FID at both k=12 and k=16 → margins 5.0 and 3.4, passed. The extra "> 2σ" condition is still pending σ from the seed runs (generation-seed σ not yet measured; margins are large but unverified against it).
**Memory axis unchanged:** at matched union global still wins (B1 k=16, union 0.333: 29.15 vs B4 k=8, 0.319: 46.57; B4 k=12, union 0.424: 31.74 vs B1 interpolated at 0.417 ≈ 25.5; B4 k=16, union 0.559: 25.78 vs B1 interpolated at 0.5625 ≈ 21.4). So banding wins strongly under equal-k (= equal per-band resident memory under band-wise offload) and loses under static union memory; both scopes must be reported.

### H1 update — kept importance mass M, held-out test (fit f(M) on the four B1 points, predict the banded points; log-linear interpolation, in-range points only)
| Config | M | actual ΔFID | pred from M | pred from k (= B1 at same k) |
|---|---|---|---|---|
| B4 k12 | 0.861 | 12.59 | 10.76 | 17.59 |
| B4 k16 | 0.908 | 6.64 | 4.92 | 10.01 |
| B2 k24 | 0.955 | 2.61 | 1.43 | 3.08 |
| B4 k24 | 0.959 | 2.14 | 1.25 | 3.08 |
| B8 k24 | 0.960 | 2.38 | 1.20 | 3.08 |
B4 k8 (M=0.785) is below the B1 range: no prediction. Reading: M predicts the banded points far better than k at low k (errors −1.8/−1.7 vs +5.0/+3.4) but at k=24 it is no better than k (errors −0.9…−1.2 vs +0.5…+0.9). **Systematic sign: actual > predicted for all five points, i.e. banded configs lose ~1–2 FID relative to global at equal kept mass.** So M is a good but not sufficient predictor; banding carries an extra penalty (candidate causes, untested: discontinuities at band boundaries, router re-normalisation effects). Four B1 points, in-sample interpolation, single seed — hypothesis-level; stays H1, not a finding. B1 k=32 and seeds will tighten it.

## 2026-10-04 — Seed-1 runs (generation seed_base 1) for B1 k=24 and B4 k=24
| Run ID | FID-10k | seed-0 value | seed difference | Union | Sanity |
|---|---|---|---|---|---|
| `compute_animals_B1_k24_seed1` | 21.693 | 22.221 | −0.528 | 0.500 | n=10000, wall 71.0min, grid ok |
| `compute_animals_B4_k24_seed1` | 20.846 | 21.277 | −0.431 | 0.747 | n=10000, wall 70.6min, grid ok |

### F32 — generation-seed noise is ~0.3–0.4 FID; the paired banded-vs-global gap is stable
Absolute FID moves by 0.53 (B1) and 0.43 (B4) between seeds, i.e. a crude per-config σ of ≈0.37 and ≈0.31 (two seeds each: **very low degrees of freedom**; pooled σ ≈ 0.34, 2σ ≈ 0.68). The two configs shift together because every config sees the same noise/class draws for a given seed, so the **paired** gap B1−B4 at k=24 is stable: 0.944 (seed 0), 0.847 (seed 1), mean 0.90. Consequences: (a) the equal-k margins at k=16 (3.37) and k=12 (5.00) exceed 2σ by a wide margin, so the pre-registered "> 0.5 FID and > 2σ" kill criterion is satisfied (those two pairs are still single-seed; σ comes from the k=24 pairs); (b) the k=24 equal-k advantage (~0.9) is real but only ~2.5σ in absolute terms — report the paired gap across seeds; (c) differences below ~0.7 FID in absolute terms are not claimable: this covers the B∈{2,4,8} comparison (21.75/21.28/21.52), consistent with "no clear trend"; (d) the equal-memory gap B1 k=36 (19.654) vs B4 k=24 (21.277) is 1.6, ≈4σ, but k=36 has one seed only. Third seed and B1 k=36 seeds still needed to firm up σ.

## 2026-10-11 — Seed-2 runs (generation seed_base 2) for B1 k=24 and B4 k=24
| Run ID | FID-10k | Union | Sanity |
|---|---|---|---|
| `compute_animals_B1_k24_seed2` | 21.755 | 0.500 | n=10000, wall 75.9min, grid ok |
| `compute_animals_B4_k24_seed2` | 21.039 | 0.747 | n=10000, wall 76.1min, grid ok |
(Pollers died while idle; pulled manually. No canary run for these two: both configs had already completed full runs.)

### F33 — three-seed σ and paired gap at k=24
B1 k=24 over seeds 0/1/2: 22.221, 21.693, 21.755 (mean 21.890, sd 0.289). B4 k=24: 21.277, 20.846, 21.039 (mean 21.054, sd 0.216). Paired B1−B4 gap: 0.944, 0.847, 0.716 (mean 0.836, sd 0.114). The gap is positive in all three seeds; absolute σ ≈ 0.25 (2σ ≈ 0.5), so F32's "~0.7 not claimable" threshold was conservative: differences below ~0.5 FID absolute remain unclaimable, the paired k=24 gap (0.84) is claimable. Equal-k margins at k=16 (3.37) and k=12 (5.00) are single-seed but ≫ 2σ.

## 2026-10-11 — B1 k=36 seed 1 and measured B1 k=32 (seed 0)
| Run ID | FID-10k | Union | Sanity |
|---|---|---|---|
| `compute_animals_B1_k36_seed1` | 19.299 | 0.750 | n=10000, wall 77.7min, grid ok |
| `compute_animals_B1_k32_seed0` | 20.117 | 0.667 | n=10000, wall 73.2min, grid ok |

### F34 — equal-union gap confirmed with a second seed; B1 k=32 measured
B1 k=36: 19.654 (seed 0), 19.299 (seed 1). Matched union 0.75 (B1 k=36 vs B4 k=24): global wins by 1.623 (seed 0) and 1.547 (seed 1), paired, consistent with F32(d) (≈4σ+). B1 k=32 (union 0.667) = 20.117 is the measured point replacing the F29/C10 interpolation; single seed. Baseline Δ vs 19.141: +0.98.
