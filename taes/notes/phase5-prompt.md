# Phase 5 continuation prompt

Paste into a fresh chat. Written 2026-10-02, mid-Phase-5 (priority 1 of Step 5.2 done).

---

Phase 5 — TAES: finish the configuration matrix (priority 2 onward).

**Working agreement.** Read `CLAUDE.md`. GPU code runs on Kaggle via the CLI
(`kernels/<name>/step.py` + `kernel-metadata.json`, `python kernels/run.py`, recipe
`SESSION-BOOTSTRAP.md` §0b, `set -a; . ./.env; set +a`, verify `kaggle config view` →
`mohammedsarim`). **One kernel at a time** — launch, watch it to completion, sanity-check,
log, only then launch the next. Do not chain or parallelize launches.

Read in order: `CLAUDE.md`; `taes/notes/SESSION-BOOTSTRAP.md` (**§6 items 10–13 are new —
Kaggle reliability gotchas from this session**); `taes/notes/results-log.md` (**F23–F25**, and
the 2026-09-19 GO decision); `WORKOUT-PLAN.md` Step 5.2.

## State entering this session

- **Step 5.1 done (F23).** `taes/src/prune.py` — compute mode (gate-mask hook) and memory mode
  (physical slice) verified bit-identical. `taes/src/sample.py` (`run_config`) and
  `taes/src/fid.py` rebuilt and validated: full-class FID-10k reproduces the Phase-1 anchor to
  0.01 (23.310 vs 23.300).
- **Domain references built:** `results/fid_ref/{animals,vehicles}_train10k.npz` (real
  ImageNet-train images, calibration images excluded). Baselines: full 23.310, **animals
  19.141**, **vehicles 24.279**. Every domain result is a delta from its own baseline, never
  from 23.30.
- **Step 5.2 priority 1 done (F25).** B=1 (global) vs B=4 (TAES), k=24, both domains:

  | Domain | B1 (global) | B4 (TAES) | Δ | Union frac (B4) |
  |---|---|---|---|---|
  | animals | 22.221 (+3.080) | **21.277 (+2.136)** | 0.944 | 0.747 |
  | vehicles | 27.644 (+3.365) | **26.621 (+2.342)** | 1.023 | 0.726 |

  TAES beats global at equal k in both domains, consistently (~1 FID, closes ~30% of the
  degradation gap). **This is not yet a memory-equal comparison** — B4 keeps 73–75% of routed
  experts vs B1's exact 50%. The headline claim this data supports is "banding improves quality
  at equal candidate-pool size k," not "banding saves memory." A fair memory test needs a global
  B1 run at k≈35–36 (matching TAES's actual union) — not yet run; consider adding it alongside
  priority 2.

## Kaggle reliability — read before launching anything (SESSION-BOOTSTRAP §6.10–13)

- The CLI gives **zero visibility into a `RUNNING` kernel** — no partial log, no partial output,
  regardless of print statements or `--file-pattern`. "Cells" don't fix this; it's a platform
  limit. Judge liveness only by `kaggle kernels list --mine` push time vs now, against the
  ~75–90 min normal wall-clock for a FID-10k run.
- `animals_B4_k24` hung `RUNNING` for ~6h on two separate launches with no code-level cause
  found (masks checked clean). Third launch completed normally. **If a run exceeds ~100 min with
  no sign of completion, kill it (`kaggle kernels delete -y <slug>`) and relaunch** rather than
  wait — don't assume it's just slow.
- **Run a fast canary (n=500, ~3–5 min) as its own separate kernel before every full n=10000
  run.** Confirm it completes normally (it will, by construction, report a high/noisy FID at
  n=500 — that's expected and not the point; the point is confirming it doesn't hang) before
  committing the full run.
- **`COMPLETE` is not proof of a good run.** Before logging any result: check `n` in
  `result.json` matches the request, `wall_s` is in the normal ~75–90 min range (too short is as
  suspicious as too long), and pull+view `samples16.png` to confirm real, class-appropriate
  images.
- Kaggle allows **2 concurrent batch GPU sessions** per account; a third push is refused.

## Goals, in order

1. **Step 5.2 priority 2 — k-sweep.** k ∈ {5, 8, 12, 16, 24, 32, 48} (hard floor k≥5), on both
   B=1 and B=4. Full matrix is 28 runs ≈ 42 GPU-h, which exceeds a week's quota — **cut to one
   domain** (animals, since priority 1 showed both domains behave alike — F22, F25) or spread
   across the two allowed concurrent sessions, and say explicitly which cut was made. This is
   where the real memory-efficiency claim lives: union/N pays off at k≲12 (F20/F22: union 0.42
   at k=12, 0.32 at k=8), so this sweep is the actual test of whether TAES is worth anything as
   a memory claim, not just a quality-at-equal-k claim.
   - While here, also run **global B1 at k≈35–36** (matching priority 1's TAES union at k=24) to
     close the memory-fairness gap flagged in F25.
2. **Step 5.2 priority 3 — random removal** (sanity floor).
3. **Step 5.2 priority 4 — frequency-only** scoring (justifies the gate/L2 terms in the
   importance formula).
4. **Step 5.2 priority 5 — B ∈ {2, 8}** (completes the band-count ablation).
5. **Step 5.2 priority 6 — DERN-style baseline.** Nice to have, cut without guilt if quota runs
   out.
6. Log every run (ID, FID, memory, one-line observation) and every failure in
   `results-log.md`, numbering findings from **F27**.

## Framing reminder (from an earlier literature check this phase)

No existing paper does post-hoc, training-free extraction of a timestep-banded expert subset
from a pretrained diffusion MoE — the closest work all trains/fine-tunes per-interval experts
(DiffPruning, Remix-DiT, ALTER) or prunes MoEs globally with no timestep axis (REAP, DERN). That
gap is real but narrow, and it is only earned if the k-sweep shows banding beating global at
*equal memory*, not just equal k. If the sweep shows the curves converging or crossing at low k,
report that honestly — a null or mixed result here is still a publishable, defensible finding
(arXiv + workshop tier at least); don't tune parameters chasing the result you want.

## Blocker rule

Three days maximum on any blocker, then say so and change approach.

## When Phase 5 is fully done

End-of-phase ritual in `CLAUDE.md` in full (mirror code, Dataset push `--dir-mode zip` staged
from current contents, results-log, WORKOUT-PLAN ticks, SESSION-BOOTSTRAP, Current state table,
next prompt, commit + push), then remind Sarim to open a new chat.
