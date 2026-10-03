Phase 5 — TAES: continue the configuration matrix (second continuation, 2026-10-04).

**Working agreement.** Read `CLAUDE.md`. GPU code runs on Kaggle via the CLI. Load credentials with
`set -a; . ./.env; set +a`, verify `kaggle config view` -> `mohammedsarim`. Recipe: `SESSION-BOOTSTRAP.md` §0b.
**Canary (n=500, own kernel) before every full n=10000 run. Sanity-check every COMPLETE** (n, wall 70-90 min, view
`samples16.png`) before logging. Kill and relaunch a run that is still RUNNING past ~100 min (F26).

Read in order: `CLAUDE.md`; `taes/notes/SESSION-BOOTSTRAP.md` (§6.10-13); `taes/notes/results-log.md`
(**F23-F29** and the two 2026-10-03 handoff notes at the bottom); `WORKOUT-PLAN.md` Step 5.2.

## Tooling that exists now
- `kernels/p5-05-sweep` and `kernels/p5-06-sweep`: two independent lanes (own slugs), so **two kernels can run
  concurrently** (Kaggle cap is 2). Each `step.py` has one `CONFIG` line: `DOMAIN, B, K, N = "animals", B, K, N`
  (N=500 -> canary, N=10000 -> full; run id and seed handled in the script).
- Launch: `python kernels/run.py p5-05-sweep taes/src/fid.py taes/src/prune.py taes/src/sample.py --pull 'result\.json|samples16\.png'`
  (the extras are required; without them the kernel dies on `NameError: np`).
- **`run.py`'s poller can crash on an empty `kaggle kernels status` reply while the kernel is fine.** If it dies,
  check `kaggle kernels status <slug>` and pull manually:
  `kaggle kernels output <slug> -p kernels/<dir>/out --file-pattern "result\.json|samples16\.png|\.log$" -o`.
- Union fractions for any (B, k) can be computed locally on CPU from `taes/results/scores/animals_B{1,2,4,8}.pt`
  with `prune.build_masks` (see the 2026-10-03 entry).
- Scoring and masks only exist for B-banded importance (`scoring.py`, `prune.build_masks`). **Random removal and
  frequency-only are not implemented** — they need new mask builders (random: a seeded random k-of-48 mask, global;
  frequency-only: rank by routing frequency alone, dropping the gate and L2 terms; check `scoring.py` for how
  the three terms are currently combined before changing anything).

## State entering this session (animals only; baseline FID 19.141)

| Config | Union | FID-10k |
|---|---|---|
| B1 k=16 | 0.333 | 29.153 |
| B1 k=24 | 0.500 | 22.221 |
| B1 k=36 | 0.750 | **19.654** |
| B2 k=24 | 0.663 | 21.747 |
| B4 k=8 | 0.319 | 46.570 |
| B4 k=12 | 0.424 | 31.735 |
| B4 k=24 | 0.747 | 21.277 |
| B8 k=24 | 0.795 | 21.522 |

Vehicles: B1 k=24 27.644, B4 k=24 26.621 (baseline 24.279).

**Findings (F27-F29):** banding beats global at *equal k* but pays 0.16-0.30 more union; at *equal memory* global wins
every pair tested (B1 k=36 vs B4 k=24: 19.65 vs 21.28; B1 k=16 vs B4 k=8: 29.15 vs 46.57). No trend across
B ∈ {2,4,8}. The B2-vs-global comparison at union 0.663 uses an **interpolated** B1 (~20.5), not a measured one.
Do not tune parameters chasing a better result; a null or mixed result is still a defensible finding.

## Goals, in order (animals only; vehicles only if quota remains)
1. **B1 k=32** (union 0.667): canary, then full. Confirms the F29 interpolation. Run concurrently with step 2's coding.
2. **Step 5.2 priority 3 — random removal** (global, seeded; k=24 and k=16 so they sit on B1's curve), then
   **priority 4 — frequency-only** (B1 and B4 at k=24). Write the mask builders locally, unit-check them
   (exactly k allowed per band, shared expert never pruned), then canary, then full.
3. **Remaining k-sweep points** as quota allows, most informative first: B4 k=5, B1 k=5, B1 k=12, B4 k=16,
   B4 k=32, B1 k=8. k=48 is skipped (all-ones mask = unpruned baseline 19.141).
4. **Priority 6 — DERN-style baseline**: nice to have, cut without guilt.
5. Log every run (ID, FID, union, wall, one-line observation) and every failure in `results-log.md`, numbering new
   findings from **F30**. Commit at natural checkpoints (no Claude co-author line — see global CLAUDE.md).

## Decision to make early
Once B1 k=32 and the priority 3/4 results are in, decide whether Phase 5's story is
"banding helps equal-k quality, not memory" (current evidence) and what is worth the remaining quota: a second
seed on the two or three load-bearing comparisons (B1 k=36 vs B4 k=24, B1 k=16 vs B4 k=8) is probably worth more than
more k points, since F29 notes differences below ~0.5 FID are inside plausible single-seed noise. Check Kaggle weekly
GPU quota first (about 30 GPU-h/week; each full run is ~1.4 GPU-h) before planning more than ~15 runs.

## Blocker rule
Three days maximum on any blocker, then say so and change approach.

## When Phase 5 is fully done
End-of-phase ritual in `CLAUDE.md` in full (mirror code, Dataset push `--dir-mode zip` staged from current
contents — remember `kernels/p5-0*/` sweep scripts and new mask builders under `taes/src/`, results-log, WORKOUT-PLAN
ticks, SESSION-BOOTSTRAP, Current state table, next prompt, commit + push), then remind Sarim to open a new chat.
