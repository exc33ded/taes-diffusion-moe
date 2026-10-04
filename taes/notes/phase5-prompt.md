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

## State entering this session (animals only; baseline FID 19.141) — **target is now TMLR (empirical study)**
Read the 2026-10-04 entries at the bottom of `results-log.md` (external verdicts, F30, the TMLR decision and plan). The method/compression framing is closed; do not reopen it unless the data forces it.

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
In flight at hand-off (check first): **B1 k=12** (lane `p5-05-sweep`) and **B4 k=16** (lane `p5-06-sweep`), launched 2026-10-04 ~11:26 IST. Pull manually if `run.py` died. Vehicles: B1 k=24 27.644, B4 k=24 26.621 (baseline 24.279).

**Findings F27-F30:** banding beats global at equal k, pays more union; global wins every matched-memory pair; no trend over B; active-top-k FLOP ceiling is 18% (not worth pursuing). Claim the paper may make: distinct reproducible band preferences + banding helps at equal k + does **not** beat global at equal memory. Nothing stronger.

## Goals, in order (TMLR plan; see results-log for rationale and cuts)
Week 1
1. Apply the kill rule to B1 k=12 / B4 k=16 (margin must exceed 0.5 FID **and** 2σ; σ comes from item 2).
2. Seeds: gen `seed_base` 1 and 2 for B1 k=24, B4 k=24, B1 k=36 (6 runs). Report mean ± std.
3. Measured B1 k=32 (replaces the interpolation).
4. Baselines: random removal (global, k=24 and k=16; new seeded mask builder) and frequency-only scoring (B1 and B4 k=24; check `scoring.py` for how the three terms combine before changing it). Unit-check masks first (exactly k per band, shared expert untouched).
5. CFG check: pool cond+uncond telemetry (`results/telemetry_p4/{d}_uncond.pt`), recompute scores, B4 k=24.
Week 2: second backbone `DSMoE-B-E48` (telemetry, split-half Jaccard, scores, then B1 vs B4 at 2-3 configs; measure cost per run first, budget 4-5 GPU-h); vehicles replication of the k=16 and k=36 pairs.
Week 3: FID-50k on 2-3 headline configs; peak-memory measurement of physical slicing; LaTeX in the TMLR template; verify every citation (MoECa 2606.15615, MosaicDiff ICCV 2025, DiffPruning, ALTER, Remix-DiT, REAP, EASY-EP, DERN, OLED-MoE; do **not** cite "DIET").
Cuts if quota runs short (in order): vehicles replication, alternate-calibration resample, third FID-50k config. Never cut: seeds, B1 k=32, random + frequency-only, second backbone.
Log every run (ID, FID, union, wall, sanity) and number new findings from **F31**. Commit at checkpoints (no Claude co-author line).

## Tooling
- Two independent lanes `kernels/p5-05-sweep` and `kernels/p5-06-sweep` (own slugs); each `step.py` has one `CONFIG` line `DOMAIN, B, K, N = ...` (N=500 canary, N=10000 full). Launch: `python kernels/run.py <lane> taes/src/fid.py taes/src/prune.py taes/src/sample.py --pull 'result\.json|samples16\.png'` (the extras are mandatory).
- `run.py`'s poller can crash on an empty status reply while the kernel is fine; check `kaggle kernels status <slug>` and pull with `kaggle kernels output <slug> -p kernels/<dir>/out --file-pattern "result\.json|samples16\.png|\.log$" -o`.
- Quota is per account (~30 GPU-h/week); two lanes shorten wall-clock but not GPU-hours. Canary (n=500, own kernel) before every full run; sanity-check n, wall (70-90 min) and the grid.
- Shelved: `kernels/g1-ditxl-timing`, `taes/notes/pivot-gate.md` (acceleration pivot).

## Blocker rule
Three days maximum on any blocker, then say so and change approach.

## When Phase 5 is fully done
End-of-phase ritual in `CLAUDE.md` in full (mirror code, Dataset push `--dir-mode zip` staged from current
contents — remember `kernels/p5-0*/` sweep scripts and new mask builders under `taes/src/`, results-log, WORKOUT-PLAN
ticks, SESSION-BOOTSTRAP, Current state table, next prompt, commit + push), then remind Sarim to open a new chat.
