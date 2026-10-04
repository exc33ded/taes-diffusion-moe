# Claims audit (TMLR rejects claims the evidence does not support)

Every claim in the paper must appear here with evidence, run IDs, and a status before submission.
Statuses: **supported** (measured, sanity-checked), **pending** (runs queued/needed), **drop-if-fails** (conditional), **do-not-claim**.
Sources are entries in `taes/notes/results-log.md` (F-numbers) unless stated. Update this file whenever a result lands.

| # | Claim (as worded in the paper) | Evidence | Status |
|---|---|---|---|
| C1 | Band-wise top-k expert sets are reproducible: split-half Jaccard 0.90–0.99 within a band vs 0.12–0.41 between distant bands (S-E48, both domains) | F21 (`p4-01-splithalf`, 72/72 cells) | supported on S-E48; **pending B-E48** |
| C2 | No load-balance-bias confound (`e_score_correction_bias` ≡ 0) | F22; SESSION-BOOTSTRAP §5 | supported |
| C3 | Animals and vehicles show the same band structure | F21/F22 | supported (S-E48) |
| C4 | This extends — not first reports — timestep preference of experts | DiT-MoE (arXiv 2407.11633) reports it qualitatively; Consensus search 2026-10-04 | **wording rule: never say "first"** |
| C5 | Compute mode and static-memory mode give identical outputs (max abs diff 0.0) | F23, `p5-01-prune` | supported |
| C6 | Full-ImageNet reproduction: FID-10k 23.310 vs 23.300 (independent implementation) | F24 | supported |
| C7 | Unpruned baselines: animals 19.141, vehicles 24.279 (own references); real-vs-real 8.21 / 7.45 | 5.2a/5.2b entries | supported |
| C8 | Equal k (animals): B4 beats B1 by 0.94 (k=24), 3.37 (k=16), 5.00 (k=12) | runs `compute_animals_B{1,4}_k{24,16,12}_seed0`; F25, F31 | supported single-seed; **pending σ** |
| C9 | Equal k (vehicles, k=24): 26.621 vs 27.644 | F25 | supported single-seed |
| C10 | At matched union, global beats banded in every pair: B1 k=36 (0.750) 19.654 vs B4 k=24 (0.747) 21.277; B1 k=16 (0.333) 29.153 vs B4 k=8 (0.319) 46.570; B1 k=16 vs B4 k=12 (0.424) 31.735 | F27, F28 | supported single-seed; **pending measured B1 k=32 and seeds** |
| C11 | Band count B∈{2,4,8}: no monotone trend at k=24 (21.75/21.28/21.52 vs 22.22) | F29 | supported single-seed; differences < ~0.5 are inside plausible noise → word as "no clear trend" |
| C12 | Seed σ small relative to the 1–5 FID equal-k margins | seed-1 runs (B1/B4 k=24) in flight; seed 2 queued | **pending** — do not state a σ until measured |
| C13 | Pooled cond+uncond calibration overlaps ours at Jaccard 0.90–0.99 | `taes/src/baselines.py` (CPU), results-log 2026-10-04 | supported (set overlap); **FID effect pending** |
| C14 | Random / frequency-only / REAP-style scoring comparison | code and scores built (`taes/src/baseline_scores.py`) | **pending runs**; set overlaps with the full score: freq 0.82–0.88, REAP-style 0.14–0.46 (not results) |
| C15 | Routed experts are ≈22% of forward FLOPs; top-1 everywhere saves ≤18% (top-3: 9%) | F30, CPU FlopCounterMode, random weights from the real config, batch 2 no CFG | supported for S-E48; **pending B-E48**; end-to-end share depends on VAE (0.142 of 0.4475 s/img, measured) |
| C16 | Kept mass M predicts ΔFID (Spearman −0.98 over 8 configs); out-of-sample at low k, banded runs ~1–2 FID worse than the global curve at equal M | H1 + held-out table in results-log 2026-10-04 | **drop-if-fails** — in-sample, 4 global points, k confounded with M; needs seeds + B-E48 |
| C17 | Per-band resident memory is k/N for banded and global alike; equal k = equal resident memory under band-wise offload | derivation; batched sampling uses one t per step (`Pruner.band_of` reads `t[0]`; 0 masked activations over full trajectories, F23) | supported as a *scope statement*; **swap latency/peak memory pending** — do not claim a speedup or saving until measured |
| C18 | Banding wins under per-band offload by up to 5 FID at k=12 | C8 + C17 | **conditional on C17 measurement**; word as "at equal per-band k" until then |
| C19 | Report union, not k, for static deployments | C10 + derivation | supported, scoped to static residency |
| C20 | A method that beats global pruning at equal static memory | none | **do-not-claim** |
| C21 | Generalises beyond DSMoE-S-E48 / ImageNet animals+vehicles | none yet | **do-not-claim** until B-E48 (and ideally vehicles at more k) |
| C22 | Any latency / throughput benefit | none measured | **do-not-claim** |

## Novelty guardrails (checked 2026-10-04)
- Prior: DiT-MoE (timestep preference, trained); Diff-MoE ICML 2025 (time-aware experts, trained); DiffPruning/ALTER/Remix-DiT (timestep subnets, trained or fine-tuned); MosaicDiff, Diff-ES (training-free timestep-staged pruning of *dense* models); MoECa (training-free caching in DiT-MoE); LLM expert-pruning scores incl. EASY-EP, REAP, DERN and the unified formulation of Liu et al. 2026 (arXiv 2606.15716).
- Our score = frequency × mean gate × mean output norm = the frequency-weighted member of the EASY-EP/REAP family; do not present the score as new.
- Not found / not cited: "DIET", "OLED-MoE" (see CITATIONS.md).
