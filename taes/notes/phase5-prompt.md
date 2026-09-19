# Phase 5 kickoff prompt

Paste into a fresh chat. Written 2026-09-19 at the end of Phase 4 (**GO**).

---

Phase 5 — TAES: implement pruning and run the configuration matrix.

**Working agreement.** Read `CLAUDE.md`. GPU code runs on Kaggle via the CLI (`kernels/<name>/step.py` + `kernel-metadata.json`, `python kernels/run.py`, recipe `SESSION-BOOTSTRAP.md` §0b, `set -a; . ./.env; set +a`, verify `kaggle config view` → `mohammedsarim`). One step per kernel, check the gate, stop for me.

Read in order: `CLAUDE.md`; `taes/notes/SESSION-BOOTSTRAP.md`; `taes/notes/results-log.md` (**F19–F22 and the 2026-09-19 GO decision**); `WORKOUT-PLAN.md` Steps 5.1–5.3.

## State entering Phase 5
- **GO.** Band structure is far above the split-half noise floor (F21: floor 0.90–0.99 vs distant-band 0.12–0.41, 72/72 cells). Mechanism: no load-balancing bias (F4); effect in all 6 layers; animals ≈ vehicles (F22).
- Scores `results/scores/{animals,vehicles}_B{1,2,4,8}.pt` = `{I:(B,L,E), B, edges, moe_blocks}` (band 0 = noisiest, F15); `scoring.topk_sets(I,k)` gives the masks. Uncond telemetry `results/telemetry_p4/{d}_uncond.pt` exists if small-k FID disagrees with the scores (pool cond+uncond before running anything new).
- **Compute vs memory:** union/N at k=24 ≈ 0.73–0.80 ⇒ **compute mode is the headline at k=24**; memory mode pays at k ≲ 12 (union 0.42–0.48 at k=12, 0.32 at k=8). For the Pareto x-axis use *active-expert budget* for compute mode and *union size (routed-param fraction)* for memory mode; compare TAES at k against B=1 at equal memory (k_global ≈ union·N).
- `taes/src/{prune,sample,fid}.py` are 0-byte stubs. Sampling/FID code lives only in the Phase-1 Kaggle notebook (baseline 23.30, seed_base 0) — **must be re-created as kernel code with identical settings** (`SESSION-BOOTSTRAP` §1, §4: EMA, euler, 25 steps, CFG 1.5, fp32, `seed = seed_base*1000003 + i`, FID via `pytorch-fid` vs `results/baseline/fid_stats_imagenet256.npz` — never regenerate). First gate: re-run the unpruned baseline on a small n and check activations/FID path against 23.30 (or a 2k-image sanity), before any pruned run.

## Goals, in order
1. **Step 5.1 — `src/prune.py`.** Compute mode: per-band gate mask at inference (the band index derives from the RF timestep `t`; mask logits of excluded experts to −inf/before top-5 so 5 experts are still selected from the allowed set; **verify masked experts never activate** using `RouterTelemetry`). Memory mode: slice expert weights, re-index the router, never touch the shared expert, verify on-disk size shrinks; report as fraction of routed params + MB. Unit test: pruned model runs and emits images. Decide and log how the k allowed experts interact with top-5 (k ≥ 5 hard floor).
2. **Step 5.2 — matrix, priority list unchanged.** Run ID `{mode}_{domain}_B{B}_k{k}_seed{s}`; priority 1 = B=1 vs B=4 at k=24, both domains (4 runs ≈ 6 GPU-h). **Do not reorder** the list (1 headline, 2 k-sweep {5,8,12,16,24,32,48}, 3 random removal, 4 frequency-only, 5 B∈{2,8}, 6 DERN). Priority 2 exceeds a week's quota — cut to one domain or use the second T4, and say so.
3. Log every run (ID, FID, memory, one-line observation) and every failure in `results-log.md`, numbering findings from F23.

## Blocker rule
Three days maximum on any blocker, then tell me and we change approach.

## When Phase 5 is done
End-of-phase ritual in `CLAUDE.md` in full (mirror code, Dataset push `--dir-mode zip` staged from current contents, results-log, WORKOUT-PLAN ticks, SESSION-BOOTSTRAP, Current state table, next prompt, commit + push), then remind me to open a new chat.
