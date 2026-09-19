"""Phase 4 noise-floor null. python noisefloor.py <p3_telemetry_dir> <p4_telemetry_dir> <figdir>
Floor = Jaccard(top-k of half0 band b, top-k of half1 band b) (disjoint images, independent noise).
Across-band = same measure between DIFFERENT bands, half0 vs half1 (so both sides carry the same cross-half noise)."""
import sys, os, torch
from scoring import importance, topk_sets
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p3, p4, figdir = sys.argv[1:4]; os.makedirs(figdir, exist_ok=True)
def jac(a, b): return (a & b).sum(-1).float() / (a | b).sum(-1).float()   # (...,E) bool -> (...)
def imp(path, B): return importance(torch.load(path), B)[0]                # (B,L,E)
E = 48; res = {}
for d in ["animals", "vehicles"]:
    for B in (4, 8):
        for k in (5, 12, 24):
            A, Bh = (topk_sets(imp(f"{p4}/{d}_h{h}.pt", B), k) for h in (0, 1))   # (B,L,E)
            J = jac(A[:, None], Bh[None])                                          # (B,B',L)
            idx = torch.arange(B)
            floor = J[idx, idx].mean(0)                                            # (L,)
            far = (J[0, B - 1] + J[B - 1, 0]) / 2                                  # extreme bands
            offd = J[~torch.eye(B, dtype=bool)].mean(0)                            # mean across all other pairs
            res[d, B, k] = (floor, far, offd, J)
            f = lambda x: " ".join(f"{v:.2f}" for v in x)
            print(f"{d:8s} B={B} k={k:2d} | floor [{f(floor)}] mean {floor.mean():.2f} | extreme [{f(far)}] mean {far.mean():.2f} | "
                  f"all-offdiag mean {offd.mean():.2f} | layers extreme<floor: {int((far < floor).sum())}/6", flush=True)
print("random baseline k/(2N-k):", {k: round(k / (2 * E - k), 3) for k in (5, 12, 24)})
# cond (500 imgs, run p3) vs uncond (500 imgs), same band, B=4
for d in ["animals", "vehicles"]:
    for k in (5, 12, 24):
        c = topk_sets(imp(f"{p3}/{d}.pt", 4), k); u = topk_sets(imp(f"{p4}/{d}_uncond.pt", 4), k)
        print(f"cond-vs-uncond {d} k={k} B=4 same-band J per layer (mean over bands):", [round(float(x), 2) for x in jac(c, u).mean(0)])
# figure: B=4, k=24 and k=12 - floor vs extreme vs adjacent, per layer, both domains
fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
for r, d in enumerate(["animals", "vehicles"]):
    for c, k in enumerate((12, 24)):
        floor, far, _, J = res[d, 4, k]
        adj = torch.stack([(J[i, i + 1] + J[i + 1, i]) / 2 for i in range(3)]).mean(0)
        x = torch.arange(6)
        for j, (v, n) in enumerate([(floor, "same band (noise floor)"), (adj, "adjacent bands"), (far, "extreme bands 0 vs 3")]):
            ax[r, c].bar(x + (j - 1) * 0.27, v, 0.27, label=n)
        ax[r, c].axhline(k / (96 - k), ls=":", c="k"); ax[r, c].set_title(f"{d}, B=4, k={k}"); ax[r, c].set_xlabel("MoE layer")
ax[0, 0].set_ylabel("cross-half Jaccard"); ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{figdir}/noisefloor.png", dpi=130)
