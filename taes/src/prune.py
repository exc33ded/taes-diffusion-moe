"""TAES pruning — Step 5.1. Import-free module (prepended after the bootstrap, like hooks.py; needs torch, bisect).

masks: list over MoE layers l of bool (B, E_l) — True = expert allowed in that band.
  build_masks(I, k)   scores I (B,L,E) -> masks (per-band top-k). Any other selector (random, frequency-only, DERN)
                      just has to produce the same shape.
  Pruner              COMPUTE mode: forward hooks; band from the RF timestep; logits of disallowed experts -> -inf, so
                      sigmoid=0 and the router's top-5 is taken from the allowed set only (k >= 5 is a hard floor).
  slice_experts       MEMORY mode: physically keep union_b S_b per layer, re-index router + masks. Shared expert untouched.
Memory mode + the returned masks is bit-for-bit the same function as compute mode with the original masks (dropped experts
are in no band's set, so they never fire) — one FID run covers both; memory mode only changes the size reported.
"""
import bisect
import torch

TOP_K = 5      # router top-k: an allowed set smaller than this cannot fill the top-k
N_FINE = 50    # fine t-bins used for calibration; bands are unions of them (scores' `edges`)


def build_masks(I, k):
    """I (B,L,E) importance -> list over L of bool (B,E) with exactly k allowed experts per band."""
    assert k >= TOP_K, f"k={k} < router top-k={TOP_K}"
    top = torch.zeros_like(I, dtype=torch.bool).scatter_(-1, I.topk(k, -1).indices, True)
    return [top[:, l].clone() for l in range(I.shape[1])]


class Pruner:
    """Compute mode. masks[l]: (B,E_l) bool. edges: band edges over the 50 fine bins (scores' `edges`)."""

    def __init__(self, model, masks, edges):
        self.model, self.edges, self.masks = model, list(edges), masks
        self.blocks = [i for i, b in enumerate(model.blocks) if getattr(b, "use_moe", False)]
        assert len(masks) == len(self.blocks)
        assert all(m.shape[0] == len(edges) - 1 and int(m.sum(-1).min()) >= TOP_K for m in masks)
        self.band, self._h = 0, []

    def band_of(self, t):
        fine = min(max(int(float(t.flatten()[0]) * N_FINE), 0), N_FINE - 1)   # same binning as RouterTelemetry
        return bisect.bisect_right(self.edges, fine) - 1

    def attach(self):
        def pre(mod, args):
            self.band = self.band_of(args[1])
        self._h.append(self.model.register_forward_pre_hook(pre))
        for l, bi in enumerate(self.blocks):
            def hook(mod, inp, out, l=l):
                return out.masked_fill(~self.masks[l][self.band].to(out.device), float("-inf"))
            self._h.append(self.model.blocks[bi].mlp.gate.register_forward_hook(hook))
        return self

    def detach(self):
        for h in self._h:
            h.remove()
        self._h.clear()


@torch.no_grad()
def slice_experts(model, masks):
    """Memory mode, in place: keep only union_b masks[l][b] per layer. Returns the re-indexed masks (list of (B, E_l')).
    Router rows / e_score_correction_bias are sliced to match; the shared expert is not touched."""
    new_masks = []
    for l, bi in enumerate(i for i, b in enumerate(model.blocks) if getattr(b, "use_moe", False)):
        mlp, m = model.blocks[bi].mlp, masks[l]
        keep = m.any(0).nonzero().flatten()
        mlp.experts.blocks = torch.nn.ModuleList([mlp.experts.blocks[int(e)] for e in keep])
        mlp.experts.num_experts = mlp.n_routed_experts = len(keep)
        mlp.gate.n_routed_experts = len(keep)
        mlp.gate.weight = torch.nn.Parameter(mlp.gate.weight[keep.to(mlp.gate.weight.device)].clone(),
                                             requires_grad=False)
        mlp.gate.e_score_correction_bias = mlp.gate.e_score_correction_bias[keep.to(mlp.gate.weight.device)].clone()
        new_masks.append(m[:, keep].clone())
    return new_masks


def routed_fraction(model, n_full=48):
    """Kept routed experts / original routed experts (shared expert excluded from both)."""
    kept = sum(len(b.mlp.experts.blocks) for b in model.blocks if getattr(b, "use_moe", False))
    L = sum(1 for b in model.blocks if getattr(b, "use_moe", False))
    return kept / (L * n_full)


def routed_params(model):
    return sum(p.numel() for b in model.blocks if getattr(b, "use_moe", False)
               for p in b.mlp.experts.parameters())
