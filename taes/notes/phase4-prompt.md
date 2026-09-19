# Phase 4 kickoff prompt

Paste into a fresh chat. Written 2026-09-19 at the end of Phase 3.

---

Phase 4 — TAES: GO/NO-GO decision.

**Working agreement.** Read `CLAUDE.md` (working agreement, current state, locked settings). Code that needs a GPU runs on Kaggle via the CLI
(`kernels/<name>/step.py` + `kernel-metadata.json`, `python kernels/run.py`, recipe `SESSION-BOOTSTRAP.md` §0b, `set -a; . ./.env; set +a`,
verify `kaggle config view` → `mohammedsarim`). CPU post-processing (scoring, figures) runs locally (`SESSION-BOOTSTRAP.md` §2b). One step, check the gate, stop for me.

Read in order: `CLAUDE.md`; `taes/notes/SESSION-BOOTSTRAP.md`; `taes/notes/results-log.md` (**F19, F20** and the Phase 3 entries are the inputs to this phase); `WORKOUT-PLAN.md` Step 4.1.

## State entering Phase 4
- Telemetry `results/telemetry/{animals,vehicles}.pt` `(1,50,6,48)` (500 imgs × 50 t-bins, conditional-only, no CFG); scores `results/scores/{domain}_B{1,2,4,8}.pt`; figures `results/figures/diagnostic_{1,2,3}.png`. All in the Dataset; local copies in `kernels/p3-01-calibrate/out/results/telemetry/`, `taes/results/{scores,figures}/`.
- F20: k=24 mean off-diagonal Jaccard 0.5–0.7 (random 0.33), gradient in band distance, extreme bands ≈0.1–0.3; union/N at k=24 ≈ 0.7–0.85 ⇒ compute mode primary. Animals/vehicles similar.
- `taes/paper/main.tex` has a Method draft (uncompiled — no LaTeX locally).

## Goals, in order
1. **Noise-floor null.** F20's Jaccard numbers have no sampling-noise baseline. Re-run calibration twice per domain on disjoint 250-image halves with different noise seeds (`kernels/p3-01-calibrate` is the template; ~3 min/half-run), score both at the same B and k, and report split-half Jaccard *within the same band* as the noise floor. Compare with across-band Jaccard. Gate: across-band overlap for distant bands is clearly below the split-half same-band overlap.
2. **Check the mechanism (WORKOUT-PLAN 4.1 / §7 Risk 5).** Is the timestep structure an artefact of the load-balancing loss? `e_score_correction_bias` is all-zero (F4), which argues no — state it. Also: uniform across depth? (per-layer numbers exist); calibration without CFG vs. real sampling — decide whether a CFG-branch check is worth a run.
3. **Write the dated GO/NO-GO entry** in `results-log.md` with the figure and one paragraph, incl. compute-vs-memory decision from figure 3 (also read the low-k end of the union curve: does memory mode pay at k ≈ 5–12?).
4. If GO, write `taes/notes/phase5-prompt.md`. **Do not reorder Step 5.2's priority list.**

## Blocker rule
Three days maximum on any blocker, then tell me and we change approach.

## When Phase 4 is done
Run the end-of-phase ritual in `CLAUDE.md` in full (mirror code, Dataset push with `--dir-mode zip` staged from current contents, results-log, WORKOUT-PLAN ticks, SESSION-BOOTSTRAP, Current state table, next prompt, commit + push), then remind me to open a new chat.
