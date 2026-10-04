Phase 5 — TAES: continuation (new chat, 2026-10-04 hand-off). **Phase 5 is NOT done.** Target venue: **TMLR, empirical-study framing.**

**Working agreement.** Read `CLAUDE.md` first. GPU code runs on Kaggle via the CLI (`kernels/<lane>/step.py` + `kernel-metadata.json`, `python kernels/run.py`, recipe `SESSION-BOOTSTRAP.md` §0b and §6.10–6.17). Load credentials with `set -a; . ./.env; set +a`, verify `kaggle config view` -> `mohammedsarim`. Canary (n=500, own kernel) before every full n=10000 run; sanity-check every COMPLETE (n, wall 70–90 min, view `samples16.png`) before logging. Never commit `.env`. No Claude/Anthropic attribution lines in commits (global CLAUDE.md).

Read in order: `CLAUDE.md`; `taes/notes/SESSION-BOOTSTRAP.md` (§6.10–6.17); `taes/notes/results-log.md` (**F23–F31, H1, the 2026-10-04 entries at the bottom**); `taes/paper/README.md`, `taes/paper/CLAIMS-AUDIT.md`, `taes/paper/CITATIONS.md`; this file.

## Where the project stands
TAES as a compression *method* is closed (F27–F30). The paper is an empirical study: band-wise expert preferences are real and reproducible; banding helps at equal per-band k (the gain grows as k shrinks, F31); it does **not** beat global pruning at equal *static* memory (union); per-band offload is the regime where equal-k matters (unmeasured); routed experts cap FLOP savings at ~18% (F30); kept importance mass is a candidate predictor of quality loss (H1, hypothesis only). Do not claim more than `CLAIMS-AUDIT.md` allows. Never say "first" for timestep expert preference (DiT-MoE, arXiv 2407.11633, reported it).

Animals results so far (baseline FID 19.141; seed 0; Δ = FID − baseline):

| B | k | union | FID | Δ |
|---|---|---|---|---|
| 1 | 12 | 0.250 | 36.736 | +17.60 |
| 1 | 16 | 0.333 | 29.153 | +10.01 |
| 1 | 24 | 0.500 | 22.221 | +3.08 |
| 1 | 36 | 0.750 | 19.654 | +0.51 |
| 4 | 8 | 0.319 | 46.570 | +27.43 |
| 4 | 12 | 0.424 | 31.735 | +12.59 |
| 4 | 16 | 0.559 | 25.779 | +6.64 |
| 4 | 24 | 0.747 | 21.277 | +2.14 |
| 2 | 24 | 0.663 | 21.747 | +2.61 |
| 8 | 24 | 0.795 | 21.522 | +2.38 |
Vehicles (baseline 24.279): B1 k=24 27.644, B4 k=24 26.621.
**Seed-1 runs (B1 k=24, B4 k=24) were launched 2026-10-04 ~12:54 IST on the two lanes — check them first** (see "First actions"). SEED_1_RESULTS_PLACEHOLDER

## Decisions already made (do not reopen without Sarim)
- TMLR empirical study; method/compression framing and the acceleration pivot are closed (`taes/notes/pivot-gate.md` is shelved; `kernels/g1-ditxl-timing` will not run).
- Pre-registered kill criterion was PASSED (F31): B4 beat B1 at equal k by >0.5 FID at k=12 and k=16. The extra "> 2σ" condition is still open pending σ.
- Scope: the lean plan ("option B") plus the per-band offload measurement is the default; the full plan ("option A") adds vehicles replication and a second extra seed. **Confirm scope with Sarim at the start of the new chat**; Sarim said the time (~2 weeks) felt long.
- Not cited, because unverified: "DIET", "OLED-MoE", "TimeAware MoE (IEEE)". Verify any new citation via the arXiv API/Consensus before adding.

## First actions in the new chat
1. `git log --oneline | head`, `git status`; read the notes above.
2. Check the seed-1 kernels: `kaggle kernels status mohammedsarim/taes-p5-05-sweep` (lane A = B1 k=24 seed 1) and `taes-p5-06-sweep` (lane B = B4 k=24 seed 1). If COMPLETE, pull manually: `kaggle kernels output <slug> -p kernels/<dir>/out --file-pattern "result\.json|samples16\.png|\.log$" -o`; run id `compute_animals_B{1,4}_k24_seed1`. Sanity-check, log, compute the seed-0 vs seed-1 differences and judge whether the F31 margins (5.0 at k=12, 3.4 at k=16) clearly exceed them; add to `CLAIMS-AUDIT.md` C12.
3. Launch the rest of week 1, one canary then one full run per lane at a time (2 concurrent sessions is the Kaggle cap).

## Queue (animals; `CONFIG` line = `DOMAIN, B, K, N, SCORE, SEED` in `kernels/p5-05-sweep/step.py` and `p5-06-sweep/step.py`; N=500 canary, N=10000 full)
1. Seed 2 for B1 k=24 and B4 k=24 (skip if option B uses a single extra seed); B1 k=36 seed 1 (and 2).
2. **Measured B1 k=32** (replaces the interpolation in C10).
3. Baselines at k=24: `SCORE="rand_s0"` (B1; also `rand_s1`, `rand_s2` if time), `"freq"` (B1 and B4), `"reap"` (B1 and B4), `"cfgpool"` (B4). Scores are embedded in `taes/src/baseline_scores.py` (extra required). Selected-set Jaccard vs the full score: freq 0.82–0.88, reap 0.14–0.46, cfgpool 0.90–0.99 — these are overlaps, not results.
4. Second backbone `DSMoE-B-E48` (`YHLLEO/DSMoE-B-E48`): measure per-run cost first (budget 4–5 GPU-h), then telemetry + split-half Jaccard + scores + unpruned baseline + B1 vs B4 at k=16 (two k if option A). Requires a new bootstrap config and a B-E48 domain reference/baseline; re-measure the FLOP ceiling (F30) on it.
5. Per-band offload measurement (peak GPU memory + swap latency) — the experiment that turns the equal-k result into a measured efficiency claim (ideally on B-E48).
6. If quota allows: vehicles replication of the k=16 and k=36 pairs; one alternate-calibration resample.
Cuts if short: vehicles replication, alternate-calibration resample, second extra seed. Never cut: σ from seeds, B1 k=32, random + frequency-only (+REAP-style), second backbone.
FID-50k on 2–3 headline configs, Precision/Recall and the memory measurement belong to Phase 6 (runbook Step 6.1–6.2) but are needed for the TMLR paper.

## Tooling (details in SESSION-BOOTSTRAP §6.14–6.17)
- Launch: `python kernels/run.py <lane> taes/src/fid.py taes/src/prune.py taes/src/sample.py taes/src/baseline_scores.py --pull 'result\.json|samples16\.png'`. The extras are mandatory (without them the kernel dies with `NameError: np`).
- Lanes: `kernels/p5-05-sweep` and `kernels/p5-06-sweep` have independent slugs and can run concurrently. `run.py` may crash on an empty status reply while the kernel is fine — check `kaggle kernels status` and pull manually.
- Run ids: `compute_<domain>_B<B>_k<k>[_<score>]_seed<s>`; canaries `canary_...` (seed_base 999).
- A background watcher (`Monitor` polling `kaggle kernels status`) is the reliable way to wait; it expires after 30 min and must be re-armed.
- CPU analysis helpers: `taes/src/scoring.py` (`importance(state,B,terms)`), `taes/src/baselines.py` (regenerates `baseline_scores.py` from local telemetry in `kernels/p3-01-calibrate/out/...` and `kernels/p4-01-splithalf/out/...`), kept-mass analysis recipe in the 2026-10-04 results-log entries.
- Paper: `taes/paper/` (see its README). No LaTeX is installed locally; upload to Overleaf to compile. `python taes/paper/tools/lint.py` must pass; `\todo` count must reach 0 before submission.

## Time and quota
~75–85 min per FID-10k run on the small model (~1.4 GPU-h); quota ~30 GPU-h/week per account (two lanes shorten wall-clock, not GPU-hours). Estimated Phase 5 remainder: ~2 weeks for the full plan, ~1 week for the lean plan; Phase 6 adds FID-50k (~7 GPU-h each) and writing.

## Blocker rule
Three days maximum on any blocker, then say so and change approach.

## When Phase 5 is fully done
End-of-phase ritual in `CLAUDE.md` in full: mirror code (`taes/src/*.py`, `taes/configs/*.json`, kernel step scripts), Dataset push with `--dir-mode zip` staged from the current contents (include `taes/src/baseline_scores.py`, the sweep outputs and the new B-E48 artifacts), results-log, WORKOUT-PLAN ticks, SESSION-BOOTSTRAP, Current state table, `phase6-prompt.md`, commit and push, then remind Sarim to open a new chat. Push to GitHub has **not** been done for any of the Phase 5 commits so far.
