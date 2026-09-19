# Phase 2 kickoff prompt — REDO

Phase 2 was completed on 2026-08-22 but the code was never mirrored off Kaggle and the
Dataset copy is gone. `taes/src/hooks.py` is 0 bytes and `taes/configs/` is empty.
Compute cost of redoing: minutes. This prompt **reproduces** the recorded decisions rather
than re-deriving them.

Paste into a fresh chat. Written 2026-08-29.

---

Phase 2 (REDO) — TAES: instrumentation. Router telemetry, domain definitions, latent cache.

**Working agreement, before anything else.** I write and run all code on **Kaggle** — there
is no local GPU or Python environment for this project. Give me Kaggle notebook cells,
**one block at a time**, then stop and wait for me to paste the output or traceback. Do not
write the whole notebook up front. Do not create local files or folders unless I ask. Assume
sessions die: everything resumable, everything to the Dataset. Kaggle's console stops
streaming on long cells, so log every chunk with `flush=True`.

**This is a redo of work that was already done and logged.** `taes/notes/results-log.md`
records the exact design decisions, the bugs hit, and the verification output. Reproduce
those decisions — do not silently invent different ones. Where the log specifies a seed, a
regex, or a preprocessing step, use exactly that.

Read these first, in order:

1. `CLAUDE.md` — working agreement, current state, locked settings, **end-of-phase ritual**.
2. `taes/notes/SESSION-BOOTSTRAP.md` — how to stand up a working Kaggle session. **Start by
   giving me the §1 bootstrap cell plus the §1a ImageNet/paths cell, and confirm
   `ready | 69.2M params | MoE blocks [1, 3, 5, 7, 9, 11]` before writing any Phase 2 code.**
   I am starting a new Kaggle notebook.
3. `taes/notes/results-log.md` — the three 2026-08-22 entries are the spec for this redo.
   Findings **F3, F7, F9** govern the hook design.
4. `WORKOUT-PLAN.md` Steps 2.1 → 2.3.

## Order of work

Do **2.1 → 2.3 → 2.2**, not the plan's written order. Latent caching (2.2) needs the
class→domain mapping from 2.3, so 2.3 is the real dependency.

---

## Step 2.1 — `src/hooks.py` (reproduce the recorded design)

`DSMoE.forward` stores nothing as an attribute, so `route_tokens_to_experts` never needs
monkeypatching. Three hook types, **300 total** (6 gates + 6 expert-groups + 288 experts):

- **Forward hook on `blocks.{2l+1}.mlp.gate`** → raw `router_logits` (pre-sigmoid).
- **Forward PRE-hook on `blocks.{2l+1}.mlp.experts`** → receives `topk_indices` as a plain
  call argument, since `DSNaiveMoE.forward(self, hidden_states, top_k_index, top_k_weights)`.
- **Forward hook on `blocks.{2l+1}.mlp.experts.blocks[e]`** → expert output **before** the
  gate multiplication. (F9: `DSNaiveMoE.forward` multiplies `top_k_weights` onto the
  `MLP`'s output in the *caller*, so hooking the `MLP` module itself is genuinely
  pre-multiplication. No correction factor.)

**F3:** recompute the pre-normalisation gate score as
`router_logits.sigmoid().gather(1, topk_indices)` — identical to the internal `topk_weights`
local *before* the `norm_topk_prob` block. Never use the returned `topk_weights`.

**F4:** dump `mlp.gate.e_score_correction_bias` per layer at `register()` time and save it
alongside the telemetry. (It was all-zeros in Phase 1; confirm again.)

**Context contract:** per-expert hooks see only the routed row subset with no token identity,
so the caller must call `telemetry.set_context(domain_idx, t)` once before each forward pass,
with every image in that call sharing domain and timestep. This matches how Step 3.1 runs.

Accumulate **running sums only** — `freq`, `gate_sum`, `l2_sum`, `l2_count` over
`(domain, timestep_bin=50, layer=6, expert=48)`. No per-token dumps.
`T_MIN, T_MAX = 0.0, 1.0` for the 50-bin discretisation — **flagged as unverified**; the
config's `timestep_start/end: 0/1000` conflicts with the "continuous t ∈ [0,1]" note in
`backbone-survey.md` §3.2. Carry the open item to Step 3.1.

**Output gate — must match exactly.** Smoke test on 10 images, single direct `model(x, t, y)`
call, domain=0, t=0.5:

```
registered 300 hooks (6 gates + 6 expert-groups + 288 experts)
output shape: torch.Size([10, 8, 32, 32])
has NaNs: False
freq summed over experts, per layer: [12800]*6
l2_count summed over experts, per layer: [12800]*6
```

`12800 = 10 images × 256 tokens (32×32 latent, patch_size 2) × top_k 5`.
`freq == l2_count` in every layer proves the router-side and expert-side hooks count the
identical routed set — no double-count, no drop.

---

## Step 2.3 — `configs/domains.json`

**Image source:** Kaggle competition dataset `imagenet-object-localization-challenge`,
attached via **Add Input → Competitions** (accept the rules once). Mounts at
`/kaggle/input/competitions/imagenet-object-localization-challenge`.
`LOC_synset_mapping.txt`: line `i` = class index `i`'s WNID + label; `n01440764 tench` at
index 0, the standard DiT/ADM/torchvision convention.

**Use word-boundary regex `\bkw\b`, never plain substring matching.** The first attempt used
substrings and was wrong: `car` matched `carassius`, `carcharodon`, `carduelis`; `bus` matched
`erythrocebus` — dozens of animals false-positived into vehicles. Verify **0 overlap** between
domains and print the count.

| Domain | Matches | Selected | Method |
|---|---|---|---|
| animals | ~317 | **100** | `np.random.default_rng(0).choice(n_matches, 100, replace=False)` |
| vehicles | ~69 | **all** | ImageNet-1k has no more |

Vehicle keywords must include `wing`, `shopping cart`, `parachute` (added in the original
broadening pass, +3 classes over the initial ~65-keyword list).

**⚠️ Vehicles cannot reach ~100.** ImageNet-1k is animal-skewed (120+ dog breeds) with a small
non-contiguous transportation set. This is a dataset property, not a bug — WORKOUT-PLAN says
"aim ~100," not "require 100." It is a named limitation for §6.4: *domain class-count is
asymmetric (100 vs 69) because ImageNet-1k's taxonomy is animal-skewed.*

**⚠️ The original animal keyword list was never recorded**, so the seeded 100-of-317 draw
cannot be reproduced bit-exactly. Nothing downstream has consumed it (Phase 3 never ran), so a
different-but-defensible set is acceptable — but **write both keyword lists into
`domains.json` itself this time**, along with the match counts and the seed, so this can never
recur.

**Calibration images:** 500/domain, spread as evenly as possible across each domain's classes
(`divmod` base + seeded-permutation remainder assignment), seeds **animals=0, vehicles=1**,
sampled without replacement from each class's `train/<wnid>/` folder.
Verify 500/500 and spot-check class↔wnid↔path pairing.

**Output:** `taes/configs/domains.json` + `taes/configs/calibration_images.json`
(500+500 `{image_id, path, class_id, wnid, label}` records).

---

## Step 2.2 — VAE latent cache

Encode all 1000 calibration images through `stabilityai/sd-vae-ft-mse` **once**.

- Preprocessing: `Resize(256) → CenterCrop(256) → ToTensor → ×2−1` (SD-VAE `[-1,1]`
  convention, matching the locked decode-side `/0.18215`).
- Encode with **`posterior.mode() * 0.18215`** — deterministic, **not** `.sample()`. Keeps
  calibration latents free of encoder sampling noise; the only randomness downstream is RF
  forward-noising at Step 3.1.

Expected: ~30 s/domain, `torch.Size([500, 4, 32, 32])`, 8.2 MB each, **16.4 MB total.**

**Output:** `results/telemetry/latents/{animals,vehicles}.pt` +
`index.json` mapping `image_id → {latent_path, latent_index, class_id, domain}`.

---

## Blocker rule

Three days maximum on any blocker, then tell me and we change approach.

## When Phase 2 is done

Run the **end-of-phase ritual in `CLAUDE.md`** in full. It is mandatory and it is what failed
last time. In particular, **step 1**: print the full contents of `taes/src/hooks.py`,
`taes/configs/domains.json` and `taes/configs/calibration_images.json` so I can commit them
locally — Kaggle-only code does not exist. Then push a Dataset version with `--dir-mode zip`,
update `results-log.md`, `SESSION-BOOTSTRAP.md`, the WORKOUT-PLAN checkboxes and `CLAUDE.md`'s
Current state table, confirm `taes/notes/phase3-prompt.md` is still accurate, and commit and
push to GitHub. Then remind me to open a new chat.
