"""Step 3.3 diagnostics. python figures.py <telemetry_dir> <figures_dir>. Band 0 = noisiest (t~0)."""
import sys, itertools, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scoring import importance, topk_sets

src, dst = sys.argv[1:3]
DOM = ["animals", "vehicles"]
S = {d: torch.load(f"{src}/{d}.pt") for d in DOM}
L, E = 6, 48

# 1. importance heatmap: expert (sorted by overall importance) x fine bin, per layer
fig, ax = plt.subplots(2, L, figsize=(20, 8), sharex=True)
for r, d in enumerate(DOM):
    I = importance(S[d], 50)[0]                                    # (50,L,E)
    order = importance(S[d], 1)[0, 0].argsort(-1, descending=True)  # (L,E)
    for l in range(L):
        ax[r, l].imshow(I[:, l, order[l]].T.numpy(), aspect="auto", cmap="viridis")
        ax[r, l].set_title(f"{d} · layer {l}", fontsize=9)
        if l == 0: ax[r, l].set_ylabel("expert (sorted by overall importance)")
        if r == 1: ax[r, l].set_xlabel("t bin (0 = noise → 49 = data)")
plt.tight_layout(); plt.savefig(f"{dst}/diagnostic_1.png", dpi=130); plt.close()

# 2. band overlap: Jaccard between S_b and S_b', B=8, k=24, per layer/domain; + mean off-diag vs k
def jac(a, b): return float((a & b).sum() / (a | b).sum())
def offdiag_mean(I, k):                                             # I (B,L,E) -> (L,)
    m = topk_sets(I, k); B = m.shape[0]
    return np.array([np.mean([jac(m[i, l], m[j, l]) for i, j in itertools.combinations(range(B), 2)]) for l in range(L)])
B, K = 8, 24
fig, ax = plt.subplots(2, L + 1, figsize=(24, 7))
for r, d in enumerate(DOM):
    I = importance(S[d], B)[0]; m = topk_sets(I, K)
    for l in range(L):
        M = np.array([[jac(m[i, l], m[j, l]) for j in range(B)] for i in range(B)])
        im = ax[r, l].imshow(M, vmin=0, vmax=1, cmap="magma"); ax[r, l].set_title(f"{d} · layer {l}", fontsize=9)
    ks = list(range(2, 47, 2))
    for l in range(L):
        ax[r, L].plot(ks, [offdiag_mean(I, k)[l] for k in ks], label=f"L{l}")
    ax[r, L].plot(ks, [k / (2 * E - k) * 0 + (lambda o: o / (2 * k - o))(k * k / E) for k in ks], "k--", label="random")
    ax[r, L].set_title(f"{d}: mean off-diag Jaccard vs k (B={B})", fontsize=9); ax[r, L].set_xlabel("k"); ax[r, L].legend(fontsize=6)
fig.colorbar(im, ax=ax[:, :L], shrink=0.6, label=f"Jaccard (k={K}, band 0 = noisiest)")
plt.savefig(f"{dst}/diagnostic_2.png", dpi=130); plt.close()

# 3. union-size curve |U_b S_b| / N vs k, per layer; B=4 solid, B=8 dashed
fig, ax = plt.subplots(1, L, figsize=(22, 3.6), sharey=True)
ks = list(range(1, 49))
for l in range(L):
    for d, c in zip(DOM, ["C0", "C1"]):
        for Bn, ls in [(4, "-"), (8, "--")]:
            I = importance(S[d], Bn)[0]
            ax[l].plot(ks, [topk_sets(I[:, l], k).any(0).sum().item() / E for k in ks], ls, color=c, label=f"{d} B={Bn}")
    ax[l].plot(ks, [k / E for k in ks], "k:", label="identical bands")
    ax[l].set_title(f"layer {l}", fontsize=9); ax[l].set_xlabel("k")
ax[0].set_ylabel("|∪_b S_b| / N"); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{dst}/diagnostic_3.png", dpi=130); plt.close()

# numbers for the log
print("k=24 mean off-diag Jaccard per layer  (random baseline = 12/36 = 0.333):")
for d in DOM:
    for Bn in (2, 4, 8): print(f"  {d:8s} B={Bn}", np.round(offdiag_mean(importance(S[d], Bn)[0], 24), 3))
print("union/N at k=24 per layer:")
for d in DOM:
    for Bn in (4, 8):
        I = importance(S[d], Bn)[0]; print(f"  {d:8s} B={Bn}", [round(topk_sets(I[:, l], 24).any(0).sum().item() / E, 3) for l in range(L)])
