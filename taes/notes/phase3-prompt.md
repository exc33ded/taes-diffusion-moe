# Phase 3 kickoff prompt

Paste into a fresh chat to start Phase 3. Written 2026-08-29, updated 2026-09-19 after the Phase 2 redo.

---

Phase 3 — TAES: calibration, banded importance scoring, and the diagnostic figures.
Also: start the paper.

**Working agreement, before anything else.** All code runs on **Kaggle through the Kaggle
CLI**, driven from this repo (CLAUDE.md "How we work"; recipe in `SESSION-BOOTSTRAP.md` §0b).
Write `kernels/<name>/step.py` + `kernel-metadata.json`, run `python kernels/run.py`, read the
log, check the step's output gate, then stop for me before the next step. Load the token with
`set -a; . ./.env; set +a`. Assume sessions die: everything resumable, everything to the
Dataset (stage-then-push, F17). Log every chunk with `flush=True`.

Read these first, in order:

1. `CLAUDE.md` — working agreement, current state, locked settings, end-of-phase ritual.
2. `taes/notes/SESSION-BOOTSTRAP.md` — CLI workflow (§0b), locked settings, gotchas.
   `kernels/_bootstrap.py` already prints the `ready | 69.2M params | MoE blocks [1, 3, 5, 7, 9, 11]`
   line; confirm the CLI still authenticates as `mohammedsarim` (`kaggle config view`) first.
3. `taes/notes/results-log.md` — findings F1–F20 and everything done so far. **F3, F9 and
   F7 govern how telemetry must be read**; Phase 2's `hooks.py` already accounts for them,
   so do not silently re-derive a different convention in `scoring.py`.
4. `WORKOUT-PLAN.md` Steps 3.1 → 3.4.

## State entering Phase 3

- Phase 1: baseline **FID-10k = 23.30**, unpruned, locked settings.
- Phase 2 complete:
  - `taes/src/hooks.py` — `RouterTelemetry(model, n_domains=2, n_bins=50, n_experts=48)`;
    `register()` → 300 hooks; **`set_context(domain_idx, t)` before every forward, one domain and
    one `t` per call**; accumulators `freq/gate_sum/l2_sum/l2_count` shaped `(D,B,L,E)`;
    `state()` / `save(path)`. Running sums only, 50 fine bins so band count **B stays a free
    parameter**. **`t ∈ [0,1]`, t=0 = pure noise, t=1 = data (F15) — bin 0 is the noisiest.**
    RF forward noising for calibration: `zt = t·z1 + (1−t)·z0`, `z1` = the cached latent.
  - `taes/configs/domains.json` — **100 animal classes (of 397 keyword matches), 69 vehicle
    classes**; keyword + exclusion lists and seeds stored inside. Vehicles
    cannot reach 100; ImageNet-1k does not contain that many. Named limitation for §6.4.
  - `taes/configs/calibration_images.json` — 500 images/domain, seeds animals=0, vehicles=1.
  - `results/telemetry/latents/{animals,vehicles}.pt` + `index.json` — 1000 cached VAE
    latents, 16.4 MB, encoded with `posterior.mode() * 0.18215` (deterministic, no encoder
    sampling noise). **Never re-encode these.**

## Goals, in order

1. **3.1 — Run calibration.** For each domain, for each of the 500 cached latents: apply RF
   forward noising stratified so every one of the 50 timestep bins gets coverage, run one
   forward pass with hooks live, `torch.no_grad()`. Output
   `results/telemetry/{domain}.pt`. Budget ~3 h/domain — checkpoint and resume.
2. **3.2 — Banded importance.** `src/scoring.py`:
   `I(e|d,b,l) = freq × mean_gate_score × mean_output_l2`, normalised within each layer.
   Aggregate the 50 fine bins into B ∈ {1,2,4,8} — B must cost nothing extra.
   Output `results/scores/{domain}_B{B}.pt`.
   **Do not double-count the gate term (F9) and use the pre-normalisation sigmoid (F3).**
3. **3.3 — The three diagnostic figures.** Importance heatmap (expert × timestep bin, one
   panel per layer); **band overlap — Jaccard between S_b and S_b′ per layer, the核
   diagnostic that decides Phase 4**; union-size curve |∪_b S_b|/N vs k, which directly
   measures §7 Risk 1. Output `results/figures/diagnostic_{1,2,3}.png`.
4. **3.4 — Start the paper.** `paper/main.tex` compiling with a complete **Methods**
   section. Write Methods first; it is the section that can be written with certainty now,
   and writing it exposes gaps in the definitions.

These figures are cheap and they decide the project. Do not spend GPU time on Phase 5
before looking at them.

## Compute budget

0.4475 s/image (transformer 0.3052 + VAE 0.1423), 25 steps, batch 64, T4.
Calibration is a **single forward pass per image**, not a full sample — far cheaper than
sampling, so 3.1 should come in well under the FID-10k figure of ~90 min per 10k passes.
Kaggle: ~30 GPU-h/week, ~20 FID configs/week. Do not reorder Step 5.2's priority list.

## Blocker rule

Three days maximum on any blocker, then tell me and we change approach.

## When Phase 3 is done

Run the **end-of-phase ritual in `CLAUDE.md`** in full — it is mandatory. In particular:
print `src/scoring.py` and any other Kaggle-written file so I can commit it locally
(Phase 2's `hooks.py` was left at 0 bytes in the repo for three weeks), push a Dataset
version with `--dir-mode zip`, update `results-log.md`, `SESSION-BOOTSTRAP.md`, the
WORKOUT-PLAN checkboxes and `CLAUDE.md`'s Current state table, write
`taes/notes/phase4-prompt.md`, then commit and push to GitHub. Then remind me to open a
new chat.
