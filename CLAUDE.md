# TAES — project instructions

Read this before doing anything. It overrides generic assumptions about what this
folder is for.

---

## What this is

An active research project: **TAES — Timestep-Aware Expert Selection**, post-hoc
training-free extraction of timestep-banded expert subsets from a pretrained diffusion
MoE. Backbone: `YHLLEO/DSMoE-S-E48`. Target: arXiv preprint + journal submission.

This is **not** a "survey the literature and write a paper" task. The plan already exists
and is being executed. `WORKOUT-PLAN.md` is the runbook; `REFERENCE-GUIDE.md` is the why;
`taes/notes/results-log.md` is the append-only record of everything done and decided.

---

## How we work — the working agreement

**All code runs on Kaggle, driven from this repo through the Kaggle CLI** (changed 2026-09-19;
previously paste-a-cell). There is no local GPU, no local ImageNet, and the local Python is only
for the CLI and kernel building.

Therefore:

1. **One step per kernel.** Write `kernels/<name>/step.py` + `kernel-metadata.json`, run
   `python kernels/run.py <name> [extras] --pull REGEX`, read the log, check the step's output gate,
   *then* move on. Do not chain steps on the assumption the first worked. Recipe:
   `SESSION-BOOTSTRAP.md` §0b. Load credentials first: `set -a; . ./.env; set +a`.
2. **Source of truth is the local repo.** `taes/src/`, `taes/configs/`, `kernels/` are written
   locally and pushed to Kaggle; whatever a kernel produces is pulled back with `--pull` and
   mirrored into the repo. `.env` (Kaggle token) is gitignored — never commit or print it.
3. **Notes stay current**: `taes/notes/results-log.md`, `SESSION-BOOTSTRAP.md`, `WORKOUT-PLAN.md`
   checkboxes, and the next phase's prompt.
4. **Assume sessions die.** Everything resumable, everything checkpointed to disk,
   everything pushed to the Kaggle Dataset `mohammedsarim/taes-artifacts`.
5. **Log every run and every decision to `results-log.md`**, especially failures. Findings
   are numbered (F1, F2, ...) and referenced by number in later phases.
6. **Three days maximum on any blocker**, then say so and change approach.
7. **One chat per phase.** At the end of a phase, remind Sarim to open a new chat.

## End-of-phase ritual — MANDATORY, no phase is "done" without it

A phase is not complete when the code runs. It is complete when it survives the session.
Work through this list explicitly and tell Sarim which steps he needs to run:

1. **Mirror code off Kaggle.** Print every file written on Kaggle this phase
   (`taes/src/*.py`, `taes/configs/*.json`) so Sarim can paste them into the local repo.
   Kaggle-only code does not exist. *This was missed at the end of Phase 2 — `hooks.py`
   sat at 0 bytes locally for three weeks.*
2. **Push a Kaggle Dataset version** with `--dir-mode zip` (see SESSION-BOOTSTRAP §3).
   Confirm every expected folder is named in the upload list.
3. **Update `results-log.md`** — findings, run-log row, decisions, failures.
4. **Tick the WORKOUT-PLAN checkboxes** for the completed steps.
5. **Update `SESSION-BOOTSTRAP.md`** with anything learned — new deps, new mounts, new
   gotchas, changed locked settings.
6. **Update the Current state table below.**
7. **Write `taes/notes/phase{N+1}-prompt.md`** in the same shape as the previous one.
8. **Commit and push to GitHub**, with the phase name in the message:
   ```powershell
   cd "D:\Research Paper"
   git add -A
   git commit -m "Phase N complete: <one-line result>"
   git push
   ```
9. **Then** remind Sarim to open a new chat.

Steps 1 and 8 are the ones that get skipped. Do not skip them.

## Tone

Sarim is running this project and knows it well. Be concise and direct. Skip preamble.
Flag real problems early and plainly — a wrong methodological choice caught now costs
minutes, caught in week 7 costs the paper. Do not pad with encouragement.

---

## Session start checklist

At the start of any new chat, before proposing work:

1. Read `taes/notes/SESSION-BOOTSTRAP.md` — how to stand up a working Kaggle session,
   locked settings, model topology, operational gotchas.
2. Read `taes/notes/results-log.md` — what is done, what was found, what is open. The
   findings F1–F20 are load-bearing; hook and scoring code depends on them.
3. Read the current phase prompt in `taes/notes/phase{N}-prompt.md`.
4. Check the **Current state** section below for where the project actually is.

Then give Sarim the bootstrap cell from SESSION-BOOTSTRAP.md §1 and wait for it to
confirm before writing any phase code.

---

## Current state

**Last updated: 2026-09-19 (end of Phase 3).**
*Whoever finishes a phase updates this section — it is the first thing a new chat reads.*

| Phase | Status |
|---|---|
| 0 — Viability / backbone choice | ✅ complete — `DSMoE-S-E48`, N=48, top-k=5 |
| 1 — Environment + smoke test + reference FID | ✅ complete — **baseline FID-10k = 23.30** |
| 2 — Instrumentation (hooks, latents, domains) | ✅ complete (redone 2026-09-19 via CLI) — `hooks.py` 300 hooks / counts 12800 per layer; 100 animal + 69 vehicle classes; 1000 latents. Mirrored locally **and** in the Dataset. |
| 3 — Calibration + scoring + diagnostics | ✅ complete — telemetry, scores B∈{1,2,4,8}, 3 figures, Method draft. F19/F20 in results-log. In the Dataset. |
| **4 — GO/NO-GO** | ⬅ **NEXT** — `taes/notes/phase4-prompt.md` |
| 5 — Main experiments | pending |
| 6 — Finalise + preprint | pending |
| 7 — Submission | pending |

**Next action:** Phase 4 — noise-floor null for band Jaccard, then the dated GO/NO-GO entry. Preliminary read (F20): bands differ smoothly with t → GO-leaning, union/N ≈ 0.7–0.85 at k=24 → compute mode primary. Open a **new chat** and paste `taes/notes/phase4-prompt.md`.

---

## Locked settings — changing any invalidates every comparison

| | |
|---|---|
| Backbone | `YHLLEO/DSMoE-S-E48` · N=48 · top-k=5 · depth 12 · rectified flow · 256×256 |
| Weights | `ckpt['ema']` — **never** `ckpt['model']` |
| Sampler | `RectifiedFlow.sample(mode='euler')` · 25 steps · CFG 1.5 |
| Precision | fp32 |
| FID | `pytorch-fid` 0.3.0 vs ADM `VIRTUAL_imagenet256_labeled.npz` (10k) |
| **Baseline** | **FID-10k = 23.30** (unpruned, seed_base 0) |
| Seeding | `seed = seed_base * 1000003 + image_index` |
| GPU | **T4 ×2** — P100 is sm_60 and cannot run torch 2.10 at all |
| MoE blocks | 1, 3, 5, 7, 9, 11 (MoE-layer `l` ↔ block `2l+1`) |

Full detail, including the model-topology cheat-sheet and the F3/F9 traps that hook code
must avoid, is in `taes/notes/SESSION-BOOTSTRAP.md`.

---

## Compute budget

0.4475 s/image (transformer 0.3052 + VAE 0.1423) at 25 steps, batch 64, T4.
FID-10k ≈ 90 min ≈ 1.5 GPU-h per config. Kaggle gives ~30 GPU-h/week ⇒ **~20 configs/week.**
Phase 5's k-sweep alone exceeds a week's quota — **do not reorder WORKOUT-PLAN Step 5.2's
priority list.**

---

## Repo layout

```
Research Paper/                 <- git root, pushed to GitHub as taes-diffusion-moe
  CLAUDE.md                     <- this file
  WORKOUT-PLAN.md               <- the runbook; check off steps as they complete
  REFERENCE-GUIDE.md            <- rationale; read §2 and §8 before each session
  research-plan-v*.md           <- historical, superseded by the two above
  kernels/                      <- Kaggle CLI kernels: _bootstrap.py, build.py, run.py, <step>/
  .env                          <- KAGGLE_API_TOKEN, gitignored
  taes/
    notes/
      SESSION-BOOTSTRAP.md      <- Kaggle setup, locked settings, gotchas
      results-log.md            <- append-only findings + run log  ** the memory **
      backbone-survey.md        <- Phase 0 evidence
      phase{N}-prompt.md        <- kickoff prompt for each phase
    src/                        <- mirrored from Kaggle at end of each phase
    configs/                    <- domains.json, calibration_images.json
    results/                    <- lives on Kaggle + the Dataset, not committed
    paper/                      <- LaTeX, from week 3
```

`results/` and large binaries stay on the Kaggle Dataset, not in git.
