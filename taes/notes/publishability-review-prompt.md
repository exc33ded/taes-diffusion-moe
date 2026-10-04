# Prompt: independent publishability review of the TAES project (journal target)

Paste everything below the line into a deep-research / reviewer-grade model. It is self-contained; the
reviewer cannot see the repo. Ask it to **verify literature claims itself** (searches, dates) rather than
trusting the framing below — the framing is the author's and may be wrong or stale.

---

You are an experienced reviewer and associate editor for ML / computer-vision journals, plus a researcher
who tracks efficient diffusion models and mixture-of-experts (MoE) compression. I need a **blunt,
evidence-based verdict on whether this project is publishable in a journal**, and if so where and in what
form. Do not be encouraging for its own sake. A clear "not publishable as framed, publishable as X if Y" or
"not publishable" is a useful answer. Today is early October 2026; **search for recent work (2025–2026)**
and check novelty yourself.

## 1. The project

**TAES — Timestep-Aware Expert Selection.** Post-hoc, training-free extraction of *timestep-banded expert
subsets* from a pretrained diffusion MoE. Idea: in a diffusion MoE the router's expert preferences change
with the denoising timestep, so instead of pruning experts globally, keep a different small expert subset per
timestep band.

**Backbone (one only):** `YHLLEO/DSMoE-S-E48` — small DiT-style diffusion MoE, ~69M params, 12 blocks, MoE in
6 of them (blocks 1,3,5,7,9,11), 48 routed experts + 1 shared expert per MoE layer, router top-k = 5,
rectified flow, ImageNet 256×256 (latent, SD-VAE). Sampler: Euler, 25 steps, CFG 1.5, fp32.

**Method.**
1. Calibration: 500 real ImageNet images per *domain* (animals = 100 classes, vehicles = 69 classes),
   VAE-encoded, noised along the flow, **class-conditional only**. Router telemetry is recorded per
   (timestep bin, layer, expert): routing frequency, pre-normalisation gate sigmoid, and expert-output L2.
   50 fine timestep bins.
2. Score: importance I(band, layer, expert) combining frequency, gate, and L2 terms. Bands: B ∈ {1,2,4,8}
   contiguous groups of the 50 bins (B=1 = a single global ranking = the standard global-pruning baseline).
3. Selection: keep the top-k experts per (band, layer). The shared expert is never pruned.
4. Two deployments, which are functionally identical in output:
   - *compute mode*: a gate mask restricts the router to the band's k allowed experts (router still picks its
     top-5 among them, so **active FLOPs/token are unchanged by construction**; no latency benefit measured);
   - *memory mode*: physically drop experts that appear in **no** band's set; the memory saving is the
     **union** across bands / 48, not k/48.
   Verified bit-identical outputs of the two modes; every (bin, layer) still routes exactly 5 experts.

**Pre-experiment diagnostics (these are the genuinely positive findings):**
- Split-half noise floor for band-wise expert-set agreement (Jaccard) is 0.90–0.99, versus 0.12–0.41 between
  distant bands: bands have **real, reproducible, distinct expert preferences**, not noise.
- No load-balance-bias confound (e_score_correction_bias is all zeros).
- Union across bands is large: at k=24 of 48, B=4 union ≈ 0.73–0.80 of experts; union ≈ 0.42 at k=12,
  0.32 at k=8.

## 2. Experimental protocol

FID-10k (pytorch-fid 0.3.0), 10,000 generated images, **single seed (seed_base 0)**. Because pruning is
domain-calibrated, each domain is evaluated with domain-class generation vs a **domain reference** (10k real
ImageNet-train images of those classes, calibration images excluded). Reproduced the full-ImageNet baseline
(23.31 vs 23.30 published-style anchor) to validate the sampler. Domain baselines (unpruned): **animals
19.141, vehicles 24.279**. Results are reported as deltas from each domain's own baseline.
Compute is Kaggle free-tier T4s (~1.4 GPU-h per FID-10k run), which limits the matrix.

## 3. Results so far

**Priority 1 — k=24 (N/2), global (B1) vs banded (B4), both domains:**

| Domain | B1 (global) | B4 (TAES) | B4 union |
|---|---|---|---|
| animals | 22.221 (+3.08) | **21.277 (+2.14)** | 0.747 |
| vehicles | 27.644 (+3.37) | **26.621 (+2.34)** | 0.726 |

**Sweep, animals only (baseline 19.141):**

| Config | Union (memory fraction) | FID-10k |
|---|---|---|
| B1 k=16 | 0.333 | 29.153 |
| B1 k=24 | 0.500 | 22.221 |
| B1 k=36 | 0.750 | **19.654** |
| B2 k=24 | 0.663 | 21.747 |
| B4 k=8 | 0.319 | 46.570 |
| B4 k=12 | 0.424 | 31.735 |
| B4 k=24 | 0.747 | 21.277 |
| B8 k=24 | 0.795 | 21.522 |

**Honest reading by the author:**
- At **equal k**, banding beats global by ~1 FID in both domains (closes ~30% of the degradation gap) —
  but only because banding keeps ~50% *more* experts in memory (union 0.73–0.75 vs 0.50).
- At **equal memory** (union-matched), **global wins every pair tested**: B1 k=36 (0.750) = 19.65 vs B4 k=24
  (0.747) = 21.28; B1 k=16 (0.333) = 29.15 vs B4 k=8 (0.319) = 46.57; and B1 k=16 with *less* memory beats
  B4 k=12 with more (31.74).
- No monotone trend over B ∈ {2,4,8} (21.75 / 21.28 / 21.52 at k=24, differences within plausible
  single-seed noise).
- So the memory-efficiency claim that motivated the method is **not supported** by current data; what is
  supported is "banding improves quality at equal candidate-pool size k" and the diagnostic that bands have
  distinct, reproducible expert preferences.
- One matched-memory comparison (B2 at union 0.663 vs global) relies on an interpolated global point
  (~20.5 at k=32, not yet measured).

**Alternative framing now under test (post-hoc, after the negative result above — judge it skeptically):
per-band residency.** Batched sampling uses a single timestep per step for the whole batch, so only the
*current band's* k experts need to be GPU-resident; the other experts wait in CPU RAM and are swapped in at
the B−1 band boundaries per trajectory. Peak resident routed-expert memory is then k/48 for banded *and*
global pruning alike, so the equal-k result (banding ≈1 FID better at k=24, both domains) becomes an
equal-resident-memory result. Pre-registered kill criterion: banded must beat global at equal k by >0.5 FID at
both k=12 and k=16, otherwise the method framing is dropped. Still to measure: peak GPU memory and swap
overhead. Caveats: this reframing was chosen after seeing a negative result; a 69M model is too small to
*need* offloading (a larger released checkpoint, DSMoE-B-E48, exists but is untested); offloading adds a
PCIe transfer dependency. Please assess whether this framing is a legitimate contribution or an
after-the-fact rescue, and what a journal reviewer would make of it.

**Planned next (not yet run):** random-removal floor, frequency-only scoring ablation (justifies gate/L2
terms), remaining k-sweep points, possibly second seeds, a DERN-style baseline, FID-50k on a few headline
configs, possibly a second backbone / larger model.

## 4. Known weaknesses (the author's own list — add yours)

- One small backbone (69M). No evidence it generalises to larger diffusion MoEs.
- Sweep is animals-only; vehicles only at k=24.
- Single seed everywhere; no error bars. FID-10k (not 50k). FID on domain-restricted references; no
  Precision/Recall, no CLIP/class-conditional fidelity, no human or downstream metric.
- No measured speed/latency/FLOPs benefit (top-k routing unchanged), no real memory benchmark beyond the
  union fraction of expert parameters (expert params are only a part of total model memory).
- Calibration is domain-specific and class-conditional; unconditional/CFG handling not studied; no
  cross-domain transfer test of the selected sets.
- Absolute degradation is large at low memory, so a practical operating point is unclear.

## 5. Literature framing (author's belief — verify and correct it)

The author believes no paper does *post-hoc, training-free* extraction of timestep-banded expert subsets from
a *pretrained* diffusion MoE; closest works either train/fine-tune per-interval experts (DiffPruning,
Remix-DiT, ALTER) or prune MoEs globally with no timestep axis (REAP, DERN, and LLM MoE pruning in general).
Also check: timestep-dependent expert routing in diffusion MoEs, expert pruning/merging in diffusion MoE
(DiT-MoE, DSMoE, EfficientMoE and descendants), timestep-aware pruning/caching/quantisation for diffusion
(e.g. timestep-dependent mixed precision, step-dependent layer skipping, dynamic computation across steps),
and anything from 2025–2026 that would make the novelty claim false or weak.

## 6. What I need from you

Answer these, in this order, with evidence and links where you used sources:

1. **Novelty audit.** List the closest prior work with dates and exactly what each does/does not do relative
   to TAES. Is the "post-hoc, training-free, timestep-banded, pretrained diffusion MoE" niche genuinely open
   as of today? Anything that scoops or substantially overlaps?
2. **Contribution assessment.** Given the results above, what is the actual, defensible contribution? Rank
   candidates: (a) a method paper, (b) an analysis / negative-result paper ("timestep specialisation exists
   in diffusion MoEs but does not translate to memory savings under post-hoc pruning"), (c) a benchmark /
   empirical-study paper, (d) not a paper yet. Say which framing is honest and which would be rejected.
3. **Verdict by venue tier**, for a **journal** (not a conference) submission. For each tier give accept /
   major revision / reject as the likely outcome of the *current* package, and of the package after the
   planned experiments: top-tier (TPAMI, IJCV, JMLR) ; strong mid-tier (TMLR, Pattern Recognition,
   Neurocomputing, IEEE TNNLS, CVIU, Neural Networks, Machine Learning) ; lower tier. Also say whether a
   workshop paper / arXiv-only is the more rational outcome. Name 3–5 specific journals, with the reasons
   (scope fit, typical novelty bar, review speed, whether negative/analysis results are accepted — TMLR is
   known to judge soundness over novelty; confirm current policy).
4. **Reviewer simulation.** Write the 5 most damaging reviewer comments (Reviewer 2 style) on the current
   package, and for each say whether it is fatal, fixable with experiments, or fixable by reframing.
5. **Minimum experiment set** that would move the verdict one tier up, ordered by value per GPU-hour, given
   a hard constraint of ~30 free Kaggle T4-hours per week and ~1.4 GPU-h per FID-10k run. Be specific:
   which baselines are mandatory (random, frequency-only, global at matched memory, an existing MoE-pruning
   method adapted to diffusion), how many seeds, FID-10k vs 50k, extra metrics, second backbone/domain,
   whether a larger or different diffusion MoE is required, and whether a real latency/memory measurement is
   required to make any efficiency claim.
6. **Is there a different positive result in the data** that I am missing — e.g., the distinct band-wise
   expert preferences used for something other than pruning (interpretability, timestep-adaptive routing,
   per-band capacity allocation, caching, quantisation), or the equal-k advantage reframed in a setting where
   it is the right metric? Be skeptical; say if there is none.
7. **Go / no-go recommendation** with a one-paragraph justification, and the single experiment or fact whose
   outcome would flip it.

## 7. Output format and rules

- Start with a **one-paragraph verdict**, then the numbered sections above.
- Separate **verified facts (with sources)** from **your judgment**. Flag anything you could not verify.
- Do not assume the numbers above are cherry-picked in either direction; if something looks internally
  inconsistent or suspicious, say so.
- Calibrate to journal reality: reviewers reject "negative result with weak baselines on one small model"
  differently at TMLR than at TPAMI. Say where the line is.
- Keep it under ~1500 words unless the evidence requires more.
