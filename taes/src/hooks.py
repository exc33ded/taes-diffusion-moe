"""RouterTelemetry — TAES Step 2.1. Running-sum router/expert telemetry for DSMoE-S-E48.

Design (results-log 2026-08-22, findings F3/F4/F7/F9):
  - forward hook on  blocks.{2l+1}.mlp.gate                -> raw router_logits (pre-sigmoid)
  - forward PRE-hook on blocks.{2l+1}.mlp.experts          -> topk_indices (call arg)
  - forward hook on  blocks.{2l+1}.mlp.experts.blocks[e]   -> expert MLP output, pre gate-multiply (F9)
  - F3: gate score = router_logits.sigmoid().gather(1, topk_indices)  (pre-norm_topk_prob; never
        use the returned topk_weights, which always sum to 2.5)
  - F4: mlp.gate.e_score_correction_bias dumped per layer at register()
Context contract: caller sets set_context(domain_idx, t) before each forward; every image in
that call shares one domain and one t. t in [0,1] (RF: t=i/steps, zt = t*z1 + (1-t)*z0).
Accumulators over (domain, t_bin, layer, expert): freq, gate_sum, l2_sum, l2_count.
"""
import torch

T_MIN, T_MAX = 0.0, 1.0


class RouterTelemetry:
    def __init__(self, model, n_domains=2, n_bins=50, n_experts=48, device="cuda"):
        self.model, self.D, self.B, self.E = model, n_domains, n_bins, n_experts
        self.blocks = [i for i, b in enumerate(model.blocks) if getattr(b, "use_moe", False)]
        self.L = len(self.blocks)
        shp = (self.D, self.B, self.L, self.E)
        self.freq = torch.zeros(shp, dtype=torch.long, device=device)
        self.gate_sum = torch.zeros(shp, dtype=torch.float64, device=device)
        self.l2_sum = torch.zeros(shp, dtype=torch.float64, device=device)
        self.l2_count = torch.zeros(shp, dtype=torch.long, device=device)
        self.e_score_bias = {}
        self.t_seen = [float("inf"), float("-inf")]
        self._ctx = None
        self._logits = {}
        self._handles = []

    # ---- context ---------------------------------------------------------
    def set_context(self, domain_idx, t):
        t = float(t.flatten()[0]) if torch.is_tensor(t) else float(t)
        b = min(max(int((t - T_MIN) / (T_MAX - T_MIN) * self.B), 0), self.B - 1)
        self._ctx = (int(domain_idx), b)
        self.t_seen = [min(self.t_seen[0], t), max(self.t_seen[1], t)]

    # ---- hooks -----------------------------------------------------------
    def register(self):
        n = {"gate": 0, "experts": 0, "expert": 0}
        for l, bi in enumerate(self.blocks):
            mlp = self.model.blocks[bi].mlp
            self.e_score_bias[l] = mlp.gate.e_score_correction_bias.detach().cpu().clone()
            self._handles.append(mlp.gate.register_forward_hook(self._gate_hook(l))); n["gate"] += 1
            self._handles.append(mlp.experts.register_forward_pre_hook(self._route_hook(l))); n["experts"] += 1
            for e, blk in enumerate(mlp.experts.blocks):
                self._handles.append(blk.register_forward_hook(self._expert_hook(l, e))); n["expert"] += 1
        total = sum(n.values())
        print(f"registered {total} hooks ({n['gate']} gates + {n['experts']} expert-groups + "
              f"{n['expert']} experts)", flush=True)
        return total

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _gate_hook(self, l):
        def hook(mod, inp, out):
            self._logits[l] = out.detach()
        return hook

    def _route_hook(self, l):
        def hook(mod, args):
            assert self._ctx is not None, "call set_context(domain_idx, t) before forward"
            d, b = self._ctx
            idx = args[1]                                               # (N*T, top_k)
            gate = self._logits.pop(l).sigmoid().gather(1, idx)         # F3: pre-norm gate score
            flat = idx.flatten()
            self.freq[d, b, l] += torch.bincount(flat, minlength=self.E)
            self.gate_sum[d, b, l] += torch.zeros(self.E, dtype=torch.float64, device=flat.device)\
                .scatter_add_(0, flat, gate.flatten().double())
        return hook

    def _expert_hook(self, l, e):
        def hook(mod, inp, out):
            d, b = self._ctx
            self.l2_sum[d, b, l, e] += out.detach().float().norm(dim=-1).sum().double()
            self.l2_count[d, b, l, e] += out.shape[0]
        return hook

    # ---- output ----------------------------------------------------------
    def state(self):
        return {"freq": self.freq.cpu(), "gate_sum": self.gate_sum.cpu(),
                "l2_sum": self.l2_sum.cpu(), "l2_count": self.l2_count.cpu(),
                "e_score_bias": self.e_score_bias, "moe_blocks": self.blocks,
                "n_bins": self.B, "t_range": (T_MIN, T_MAX), "t_seen": self.t_seen}

    def save(self, path):
        torch.save(self.state(), path)
