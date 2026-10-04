# Pivot gate — can a free T4 host a credible diffusion-acceleration paper? (2026-10-04)

**Why:** TAES is closed as a method (F27–F30). Sarim wants a journal paper with something real, not an analysis paper, on any diffusion-efficiency topic. Consensus scan (2026-10-04): training-free DiT acceleration is saturated (TaylorSeer, HiCache, PrediT, Spectrum, OmniCache, SODA, MosaicDiff, RALU, JiT, MrFlow, Q&C…, claims 3–12× on FLUX/Hunyuan). A new method has low odds; the unverified open angle is a matched-protocol, commodity-GPU study of acceleration methods (composability, rank stability). This gate decides, cheaply, whether a T4 can host either.
**Budget:** ≤ ~9 GPU-h, ≤ 2 days. Pre-registered rules below; do not move them after seeing data.

## Stages (one kernel at a time per lane; canary/timing before any 10k run)
1. **Timing probe** — `kernels/g1-ditxl-timing` (written, not yet run). DiT-XL/2-256 via diffusers, DPM-Solver, CFG 1.5, fp16 and fp32, 25 and 50 steps; s/img (transformer vs total), peak VRAM, projected FID-10k GPU-h. ~15 min.
   **Rule:** if projected FID-10k (fp16, 25 steps) > 5 GPU-h → switch the testbed to DSMoE-S (pipeline already validated, FID-10k = 75 min) or cut n; otherwise continue with DiT-XL/2.
2. **Unpruned baseline FID-10k** on the chosen testbed against the locked ADM stats (DiT-XL/2: expect FID-10k roughly 5–12; no anchor to the paper's 2.27 because FID-50k + different sampler; use as internal reference only).
3. **One caching baseline reproduced, 2 runs** — FORA/TaylorSeer-style block-output caching, refresh interval N=2 and N=3 (own ~40-line hook, or TaylorSeer's public code after checking its licence). Measure real T4 wall-clock speedup and ΔFID-10k.
   **Rule (go):** measured speedup ≥ 1.5× at ΔFID ≤ 0.5 (N=2) ⇒ T4 can host credible work ⇒ choose between (A) matched-protocol study + composability and (B) a new method, with data in hand.
   **Rule (stop):** speedup < 1.2× or baseline infeasible ⇒ the commodity-GPU setting cannot support the published claims; write that up in the log, and stop (or scope down), no further GPU.

## Standing caps for the whole pivot
- Three weeks maximum from 2026-10-04; at the end of the gate decide go/stop explicitly.
- Publishable shape must contain a measured, non-interpolated efficiency result (wall-clock on a stated GPU) — no FLOPs-only claims.
- Every number logged with n, wall time, sanity check (as in SESSION-BOOTSTRAP §6.13).
