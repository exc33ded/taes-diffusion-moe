# TAES — Workout Plan (Execution Runbook)

**The "what do I do right now" document.**
Companions: `REFERENCE-GUIDE.md` (why — read before each session) · `research-plan-v5-8WEEK.md` (schedule)

Every step has: **Goal → Do → Output → Then**. Do not skip the *Output* line; if you cannot produce it, the step is not done.

---

## Phase 0 — Viability check ✅ **COMPLETE (31 Jul 2026)**

> **Outcome.** DiT-MoE failed the N rule at N=8 — its advertised `S/2-16E2A` checkpoint does not exist on HuggingFace, and the only N=16 release (G/2) is 33 GB. Step 0.2 fired. **Backbone is now `YHLLEO/DSMoE-S-E48`: N=48, top-k=5, depth 12, ImageNet-256, rectified flow, 1.11 GB.** N≥32 ⇒ **memory mode is the primary claim**.
>
> Full evidence: `notes/backbone-survey.md`. Decision entry: `notes/results-log.md`.
>
> **Carried into Phase 1:**
> - Sampler is rectified flow, not DDIM
> - `interleave: true` ⇒ MoE in 6 of 12 blocks — confirm which before writing hooks
> - Shared expert is never prunable; exclude from N and all memory figures
> - k-sweep floor is k ≥ 5 (router top-k)
> - Read Diff-MoE (ICML 2025) figures before the week-4 GO/NO-GO

<details>
<summary>Original Step 0.1 / 0.2 procedure (kept for the record)</summary>

### Step 0.1 — Count the experts

**Goal.** Determine N (experts per layer) in available DiT-MoE checkpoints. This decides whether the project is viable as designed.

**Do.**
1. Open `https://github.com/feizc/DiT-MoE`. Read the README and the model-config section.
2. Find the config definitions — look for the model registry (often `models.py` or `model.py`) where variants like `DiT-MoE-S/2`, `B/2`, `L/2`, `XL/2` are declared. Note `num_experts` and `num_experts_per_tok` (top-k) for each.
3. Open the linked HuggingFace repo(s). List which checkpoints are **actually released** — not just which configs exist in code. *(This step is what caught the dead link. Never trust a README table.)*
4. Record for every released checkpoint: variant name, N, top-k, number of MoE layers, resolution, parameter count.

**Output.** A table in `notes/backbone-survey.md`.

**Then — apply the decision rule:**
- **N ≥ 32 released** → proceed to Step 1.1 with that checkpoint. Memory mode is primary. ← *where DSMoE-S-E48 landed*
- **16 ≤ N < 32** → proceed, run both modes, decide later from measured union sizes.
- **Only N = 8** → **stop and do Step 0.2 before any GPU work.** ← *where DiT-MoE landed*

### Step 0.2 — Backbone fallback (only if N < 16)

**Do.** In this order, stop at the first that works:
1. Check `kunncheng/Diff-MoE` (ICML 2025) for released checkpoints and their N. → *no checkpoints released; Related Work only*
2. Check `yhlleo/EfficientMoE` (*Efficient Training of Diffusion MoE: A Practical Recipe*, 2025) — it explicitly studies varying expert counts, so higher-N configs likely exist. → ✅ *DSMoE-S-E48, N=48*
3. **Fallback — train your own.** A small MoE-DiT on CIFAR-10 or CelebA-64, **with N set to 32**. ~15 GPU-hours. Weaker backbone, but you control N and sidestep §7 Risk 1 entirely. This is a legitimate choice, not a defeat. → *still the last resort if Phase 1 fails*

</details>

---

## Phase 1 — Environment and smoke test (Week 1)

### Step 1.1 — Kaggle project skeleton

**Goal.** A reproducible, resumable workspace. Kaggle sessions die; design for it.

> ✅ **Local skeleton already built at `taes/`** (31 Jul) with `notes/backbone-survey.md` and `notes/results-log.md` populated. Remaining: GitHub push, Kaggle notebook, Kaggle Dataset.

**Do.**
1. New Kaggle Notebook, GPU accelerator on, internet on.
2. Create this structure in your repo (mirror locally + GitHub so nothing lives only on Kaggle):

```
taes/
  configs/          # experiment configs, one YAML per run
  src/
    hooks.py        # router telemetry
    scoring.py      # banded importance
    prune.py        # extraction / masking
    sample.py       # generation
    fid.py          # metric computation
  notes/            # backbone-survey.md, results-log.md
  results/          # one subdir per run_id
  paper/            # LaTeX from week 3
```

3. Create a Kaggle Dataset for artifacts (latents, telemetry, checkpoints). **Every run writes there, never only to `/kaggle/working`.**

**Output.** Repo pushed to GitHub; empty Kaggle Dataset created; notebook attaches it.

**Then.** Step 1.2.

### Step 1.2 — Load and sample

**Goal.** One image out of the pretrained model. Nothing else matters until this works.

**Do.**
1. Clone `yhlleo/EfficientMoE` **and** its dependency `KlingTeam/DiffMoE`. Install: `torch==2.6.0`, `torchvision==0.21.0`, `peft==0.17.1`, `tensorflow==2.15.0`, `timm torchdiffeq diffusers transformers`, and the `LTH14/torch-fidelity` fork from git.
2. Download `YHLLEO/DSMoE-S-E48` → `checkpoints/0700000.pt` (1.11 GB) and the config YAML. VAE is `stabilityai/sd-vae-ft-mse`.
3. Run the repo's own sampling script unmodified, default settings. **Snapshot the working environment into a Kaggle Dataset the moment it runs** — reinstalling this stack after a dead session is a half-day you do not have.
4. Save a grid of 16 images.
5. Time it: seconds per image at 25 **rectified-flow** steps, batch 64. **Record this — your entire compute budget derives from it.**
6. While you are in the code: note which of the 12 blocks carry MoE (`interleave: true`) and the exact router module path. `hooks.py` needs both.

**Output.** `results/smoke/grid.png`, a recorded seconds-per-image figure, and the MoE block indices written into `notes/results-log.md`.

**Then.**
- Works → Step 1.3.
- Broken after **3 days** → `DSMoE-B-E48`, then the train-your-own fallback in Step 0.2.3. Do not debug into week 2.

### Step 1.3 — Reference FID

**Goal.** A number you trust, to anchor every later comparison.

**Do.**
1. Pick your FID implementation (`pytorch-fid` or `clean-fid`) and **never change it**.
2. Generate 10k images, unpruned, fixed seed.
3. Compute FID-10k against the ImageNet reference statistics for your resolution.
4. Compare to the backbone's reported figure: **DSMoE-S-E48 = FID-50K 14.81 @ CFG 1.5, IS 96.51** (700K steps, 30M active params). Ours is FID-10k at our own settings, so expect a sizeable gap — you need it *stable*, not identical.

**Output.** `results/baseline/fid10k.json` with FID, seed, sampler, steps, and the exact command.

**Then.** Step 2.1. **This is your Week-1 hard gate.**

---

## Phase 2 — Instrumentation (Week 2)

### Step 2.1 — Router telemetry hooks

**Goal.** Capture routing behaviour without touching the forward pass.

**Do.** In `src/hooks.py`, register a forward hook on every router module. Per call, record:

| Field | Why |
|---|---|
| `layer_idx` | per-layer analysis |
| `timestep` | the banding axis |
| `expert_idx` selected (top-k) | frequency term |
| `gate_score` per selected expert | confidence term |
| `output_l2` per selected expert | contribution term |
| `domain_label` | secondary axis |
| `seed` | reproducibility |

Accumulate **running sums, not per-token dumps** — full logs will blow past memory and disk. You need means and counts per `(domain, timestep_bin, layer, expert)`. Use fine timestep bins (say 50) at collection time and aggregate into bands later, so band boundaries stay a free parameter.

**Output.** `src/hooks.py` plus a smoke test on 10 images producing a populated tensor with no NaNs and frequencies summing to top-k per token.

**Then.** Step 2.2.

### Step 2.2 — Cache VAE latents

**Goal.** Never run the VAE twice.

**Do.** Encode all images for both domains once; save latents to the Kaggle Dataset with an index mapping `image_id → (latent_path, class, domain)`.

**Output.** Cached latents + index. Note the size.

**Then.** Step 2.3.

### Step 2.3 — Define the domains

**Goal.** Two reproducible ImageNet superclasses.

**Do.** Build explicit class-ID lists for **animals** and **vehicles** (aim ~100 classes each, non-overlapping). Save as `configs/domains.json`. Sample 500 calibration images per domain, fixed seed.

**Output.** `configs/domains.json` + calibration image ID lists. **Commit these — reviewers may ask, and you must be able to re-run identically.**

**Then.** Phase 3.

---

## Phase 3 — Calibration and scoring (Week 3) — *and start writing*

### Step 3.1 — Run calibration

**Do.** For each domain, for each of 500 calibration images: sample noisy latents across the full t range (stratified so every timestep bin gets coverage), run one forward pass with hooks live. No gradients — wrap in `torch.no_grad()`.

**Output.** `results/telemetry/{domain}.pt` — the aggregated statistics tensor. ~3 h/domain.

**Then.** Step 3.2.

### Step 3.2 — Compute banded importance

**Do.** In `src/scoring.py`, for each `(domain d, band b, layer l, expert e)`:

```
I(e | d,b,l) = freq(e|d,b,l) × mean_gate_score(e|d,b,l) × mean_output_l2(e|d,b,l)
```

Normalise within each layer so layers are comparable. Make **B a parameter**, computed by aggregating the fine bins — so B ∈ {1,2,4,8} costs nothing extra.

**Output.** `results/scores/{domain}_B{B}.pt` for each B.

**Then.** Step 3.3.

### Step 3.3 — The diagnostic figures

**Goal.** See whether banding does anything, *before* spending GPU time on it.

**Do.** Produce three plots:
1. **Importance heatmap** — expert (y) × timestep bin (x), one panel per layer. Visual test of whether importance shifts with t.
2. **Top-k set overlap across bands** — Jaccard similarity between S_b and S_b' for all band pairs, per layer. **This is the核 diagnostic.**
3. **Union size curve** — |∪_b S_b| / N versus k, per layer. **This directly measures §7 Risk 1.**

**Output.** `results/figures/diagnostic_{1,2,3}.png` — these go in the paper.

**Then.** Step 4.1. **These figures are cheap and they decide the project.**

### Step 3.4 — Start the paper (parallel, from now on)

**Do.** Set up `paper/` with the target journal's LaTeX template. Write **Methods first** — it is the section you can write with certainty right now, and writing it exposes gaps in your own definitions.

Draft in this order across the coming weeks: **Methods (wk3) → Related Work (wk5) → Experiments (wk6) → Results (wk7) → Intro + Discussion (wk7) → Abstract (last)**.

**Output.** `paper/main.tex` compiling, with a complete Methods section.

**Then.** Continue in parallel; never leave writing to the end.

---

## Phase 4 — GO / NO-GO (Week 4)

### Step 4.1 — The decision

**Look at diagnostic figure 2 (band overlap).**

**GO — bands select measurably different experts** (Jaccard well below 1.0 in a meaningful fraction of layers):
→ The core hypothesis is alive. Proceed to Step 5.1.
→ Check figure 3: if union/N is near 1.0, **compute mode is primary** (§5.4 rule). Record the choice in `notes/results-log.md`.

**NO-GO — bands select essentially the same experts** (Jaccard ≈ 1.0 nearly everywhere):
→ Timestep-banded importance is not distinct from global. **Pivot now**, do not push on.
→ New framing: *"Expert importance in diffusion MoE is timestep-invariant, contradicting the specialization structure reported by DiT-MoE."* (Still a valid framing on a DSMoE backbone — the DiT-MoE claim is about diffusion MoE routing generally, and DSMoE uses the same DeepSeek-style router. Say so explicitly.)
→ You still need a mechanism, not just the observation (§7 Risk 5). Investigate: is it uniform across depth? Does it hold per-domain? Is it an artefact of the load-balancing loss during pretraining?
→ Same schedule, shorter paper, honest result. Go to Step 5.1 with baselines only.

**Output.** A dated GO/NO-GO entry in `notes/results-log.md` with the figure and one paragraph of reasoning.

**Then.** Phase 5.

---

## Phase 5 — Main experiments (Weeks 5–6)

### Step 5.1 — Implement pruning

**Do.** In `src/prune.py`:
- **Compute mode** — a gate mask applied per band at inference. Verify masked experts genuinely never activate.
- **Memory mode — this is the primary claim** (N=48 ≥ 32, §5.4 rule). Physically slice expert weights from the state dict, re-index the router output projection, save a new checkpoint. **Verify on-disk size actually shrinks.** Never slice the shared expert. Report reduction as *fraction of routed params* as well as absolute MB — DSMoE's experts are narrow (`moe_intermediate_size: 128`), so absolute savings look smaller than the expert-count reduction suggests.

**Output.** Both paths working; a unit test confirming a pruned model runs and produces images.

**Then.** Step 5.2.

### Step 5.2 — Run the configuration matrix

**Do.** Establish a run ID convention: `{mode}_{domain}_B{B}_k{k}_seed{s}`. Every run writes `results/{run_id}/` containing config, FID, memory, latency, and 16 sample images.

Run order — **most informative first**, so a quota shortfall costs the least:

| Priority | Config | Why first |
|---|---|---|
| 1 | B=1 (global) vs B=4 (TAES), **k=24 (=N/2)**, both domains | **The headline result.** If this fails, nothing else matters. |
| 2 | k sweep on B=1 and B=4, **k ∈ {5, 8, 12, 16, 24, 32, 48}** (hard floor k ≥ 5 = router top-k) | Builds the Pareto curve |
| 3 | Random removal | Sanity floor |
| 4 | Frequency-only | Justifies the L2 and gate terms |
| 5 | B ∈ {2,8} | Completes the band ablation |
| 6 | DERN-style baseline | Nice to have; cut without guilt |

FID-10k throughout. Queue the next run before you start writing — **never leave the GPU idle**; Kaggle quota does not roll over.

**Output.** `notes/results-log.md` updated after each run: run ID, FID, memory, one-line observation.

**Then.** Step 5.3.

### Step 5.3 — The main plot

**Do.** Memory (or active params) on x, FID on y. One curve per method, error bars over 3 seeds, one panel per domain.

**Output.** `results/figures/pareto.png`.

**Then.** **Read it honestly.** TAES curve below global = the paper works. Curves overlapping = report it as such. Do not tune until you get the answer you want — that is how papers get retracted.

---

## Phase 6 — Finalise and preprint (Week 7)

### Step 6.1 — FID-50k
Run on exactly 4 configs: no-prune, random, B=1 best-k, B=4 best-k. ~40 GPU-hours. Start this **at the beginning of week 7** — it is the long pole.

### Step 6.2 — Precision / Recall
On the same 4 configs. Catches diversity collapse that FID alone can hide. Cheap once samples exist.

### Step 6.3 — Figures
Final versions of: 3 diagnostics, Pareto, qualitative grid (same seed and prompt, each method), union-size analysis. Vector format, readable at print size.

### Step 6.4 — Finish the paper
Write Results into the drafted skeleton, then Intro and Discussion. **Write Limitations explicitly**: one backbone, one dataset, two domains, N constrained. Naming your own limits pre-empts reviewers.

### Step 6.5 — arXiv
Post it. **Do not wait for polish.** Categories: `cs.CV` primary, `cs.LG` cross-list. This is the scoop insurance and the deliverable you can cite immediately.

**Output.** Live arXiv preprint.

---

## Phase 7 — Submission (Week 8)

1. Choose the journal. Read two recent MoE/efficiency papers from it to calibrate length and tone.
2. Reformat to their template; check page and figure limits.
3. Write the cover letter — 3 sentences on the contribution, 1 on fit.
4. Clean the GitHub repo: README with exact repro commands, `requirements.txt`, the config files, the domain class lists.
5. Submit. Log the date in `notes/results-log.md`.

---

## Standing rules

1. **GPU never idle.** Queue the next run before writing, reading, or thinking.
2. **Everything to the Kaggle Dataset.** Sessions die without warning.
3. **One run, one directory, one config file.** No untracked flag changes.
4. **Log after every run**, even failures — especially failures.
5. **Fixed seeds everywhere**, recorded in the config.
6. **Read `REFERENCE-GUIDE.md` §2 and §8 before each session.** If tempted to add scope, check the decision log first; if it is not there, add it with a decision rather than acting silently.
7. **Three days maximum on any blocker.** Then switch approach. Sprints die from stubbornness, not difficulty.
8. **Write weekly.** The paper is not a week-8 activity.

---

## Week-1 checklist (start here)

- [x] Step 0.1 — count experts, fill `notes/backbone-survey.md`
- [x] Apply the N decision rule → DiT-MoE failed at N=8, Step 0.2 done, **DSMoE-S-E48 (N=48)** selected
- [x] Step 1.1a — local repo skeleton at `taes/`
- [x] Step 1.1b — GitHub `taes-diffusion-moe` pushed; Kaggle Dataset `mohammedsarim/taes-artifacts` created + attached
- [x] Step 1.2 — loaded `0700000.pt` (**ema**), 16-image grid verified, **0.4475 s/img** @ 25 RF steps / batch 64 / T4
- [x] Confirm which 6 of 12 blocks carry MoE + the router module path → blocks **1,3,5,7,9,11**; router `blocks.{b}.mlp.gate`
- [x] Step 1.3 — **baseline FID-10k = 23.30**, `pytorch-fid` 0.3.0, saved to `results/baseline/fid10k.json`
- [x] Step 2.1 — `taes/src/hooks.py`, 300 hooks, `freq == l2_count == 12800` per layer (redone 2026-09-19)
- [x] Step 2.3 — `configs/domains.json` (100 animal / 69 vehicle classes) + `calibration_images.json` (500+500)
- [x] Step 2.2 — VAE latents cached, `(500,4,32,32)` ×2, in Dataset `results/telemetry/latents/`
- [x] Step 3.1 — calibration telemetry, 500 img × 50 bins per domain, `freq`=640000 per (bin,layer) (2026-09-19)
- [x] Step 3.2 — `src/scoring.py`, `results/scores/{domain}_B{1,2,4,8}.pt`
- [x] Step 3.3 — `results/figures/diagnostic_{1,2,3}.png`; reading = F20
- [x] Step 3.4 — `taes/paper/main.tex` with complete Method section (not yet compiled: no LaTeX on this machine)
- [ ] Install TooManyPapers, load the §9 bibliography
- [ ] Read DSMoE **(Do et al., LLM)** and Ban&Pick scoring functions in full — ⚠️ not DSMoE (Liu et al.), the backbone
- [ ] Read the DiT-MoE expert-specialization section in full
- [ ] Read EfficientMoE §4 (the expert-count study) — it is the backbone's own justification for N=48
- [ ] Read Diff-MoE (ICML 2025) figures — do they already show band-wise expert divergence?
