"""Banded importance scoring — TAES Step 3.2.
I(e|d,b,l) = freq x mean_gate x mean_l2, normalised to sum 1 over experts within each (d,b,l).
Bands aggregate the RAW SUMS of the 50 fine bins (never average ratios), so B is free.
Band 0 = noisiest (t~0), band B-1 = cleanest (F15). Telemetry per F3 (pre-norm sigmoid) / F9 (pre-gate expert output).
"""
import torch


def band_edges(B, n_bins=50):
    return [round(i * n_bins / B) for i in range(B + 1)]


def importance(state, B):
    """state: RouterTelemetry.state() dict, tensors (D,50,L,E). Returns (D,B,L,E), rows sum to 1 over E."""
    e = band_edges(B, state["n_bins"])
    def agg(k): return torch.stack([state[k][:, e[i]:e[i + 1]].double().sum(1) for i in range(B)], 1)
    freq, gate, l2, cnt = agg("freq"), agg("gate_sum"), agg("l2_sum"), agg("l2_count")
    f = freq.clamp(min=1)
    I = freq * (gate / f) * (l2 / cnt.clamp(min=1))          # zero-freq -> 0 (freq factor)
    return I / I.sum(-1, keepdim=True).clamp(min=1e-30)


def topk_sets(I, k):
    """I (..., E) -> bool mask (..., E) of the top-k experts."""
    return torch.zeros_like(I, dtype=torch.bool).scatter_(-1, I.topk(k, -1).indices, True)


if __name__ == "__main__":
    import sys, os
    src, dst = sys.argv[1], sys.argv[2]                       # telemetry dir, scores dir
    os.makedirs(dst, exist_ok=True)
    for d in ["animals", "vehicles"]:
        s = torch.load(f"{src}/{d}.pt")
        for B in (1, 2, 4, 8):
            I = importance(s, B)[0]                            # (B,L,E)
            assert torch.allclose(I.sum(-1), torch.ones(1, dtype=I.dtype)), I.sum(-1)
            torch.save({"I": I.float(), "B": B, "edges": band_edges(B), "moe_blocks": s["moe_blocks"]},
                       f"{dst}/{d}_B{B}.pt")
            print(f"{d} B={B} {tuple(I.shape)} ok", flush=True)
