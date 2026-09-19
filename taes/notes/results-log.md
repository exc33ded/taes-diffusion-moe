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

---

## Run log

| Date | Run ID | Config | FID | Memory | Latency | Observation |
|---|---|---|---|---|---|---|
| 2026-07-31 | `baseline_unpruned_seed0` | DSMoE-S-E48 ema, N=48 k=5, 25 RF steps, CFG 1.5, fp32 | **23.30** (FID-10k) | 69.2M params / 10.76 GB peak | 0.4475 s/img (tf 0.3052 + vae 0.1423) | Reference anchor. Paper reports 14.81 @ FID-50k/ADM-TF — gap expected. |
