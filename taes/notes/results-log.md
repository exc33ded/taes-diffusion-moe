# TAES — Results & Decision Log

Append-only. One entry per run, per decision, per failure. Especially failures.

---

## 2026-07-31 — DECISION: backbone switched from DiT-MoE to DSMoE-S-E48

**Phase 0, Step 0.1 → 0.2.**

DiT-MoE fails the N decision rule. The advertised `dit_moe_s_16E2A.pt` is a dead link — not present in the HuggingFace file tree. Every *usable* released DiT-MoE checkpoint is N=8, top-k=2; the sole N=16 release is G/2 at 33 GB, which Kaggle cannot host.

Step 0.2 fallback search: Diff-MoE (ICML 2025) has no released checkpoints. **EfficientMoE (arXiv 2512.01252) does**, including `DSMoE-S-E48` — **N=48, top-k=5, depth 12, 256×256, 1.11 GB**.

**Decision: DSMoE-S-E48 is the backbone.** N=48 clears the N≥32 bar, so **memory mode is the primary claim**. Full reasoning and consequences for hooks/sampler/scoring: `backbone-survey.md` §4–§5.

**Open items created by this decision:**
- [ ] Confirm which 6 of 12 blocks carry MoE (`interleave: true`) from `models_DSMoE.py`
- [ ] Rewrite the DiT-MoE assumptions in `REFERENCE-GUIDE.md` / `research-plan-v5-8WEEK.md`
- [ ] Sampler is rectified flow, not DDIM — Step 1.2 timing and Step 1.3 FID command change
- [ ] Read Diff-MoE's figures before the Week-4 GO/NO-GO; if they already show band-wise expert divergence, our diagnostic 2 is confirmatory, not novel

---

## Run log

| Date | Run ID | Config | FID | Memory | Latency | Observation |
|---|---|---|---|---|---|---|
| | | | | | | |
