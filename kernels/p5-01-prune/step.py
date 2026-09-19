# ---- Step 5.1 unit test: compute mode (masked experts never fire), memory mode (== compute mode, smaller on disk), images ----
import zipfile, copy, io
SD = f"{DS}/results/scores"
if not os.path.isdir(SD):
    zipfile.ZipFile(f"{DS}/results.zip").extractall("/tmp/res"); SD = "/tmp/res/scores"
OUT = f"{RES}/p5_01"; os.makedirs(OUT, exist_ok=True)
K, DOM, B = 24, "animals", 4
sc = torch.load(f"{SD}/{DOM}_B{B}.pt"); masks = build_masks(sc["I"], K)
NB = 50

def sd_mb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.tell() / 2**20

# --- (1) all-ones masks reproduce the unpruned model exactly
x = torch.randn(4, 4, 32, 32, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
y = torch.tensor([1, 2, 3, 4], device="cuda")
with torch.no_grad():
    t = torch.full((4,), 0.3, device="cuda"); ref = model(x, t, y)[0]
    p = Pruner(model, [torch.ones_like(m) for m in masks], sc["edges"]).attach()
    assert torch.equal(model(x, t, y)[0], ref); p.detach()
print("(1) all-ones mask == unpruned: exact", flush=True)

# --- (2) compute mode: telemetry over a full 25-step CFG sampling run
def run_tel(pr):
    tel = RouterTelemetry(model, n_domains=1, n_bins=NB); tel.register()
    h = model.register_forward_pre_hook(lambda mod, a: tel.set_context(0, a[1]))
    if pr: pr.attach()
    lat, yy = sample_latents(16, seed=7)
    if pr: pr.detach()
    h.remove(); tel.remove(); return tel.state(), lat, yy

pr = Pruner(model, masks, sc["edges"])
st, lat_c, yy = run_tel(pr)
fr = st["freq"][0]                                           # (50 bins, L, E)
viol, allowed_hits = 0, 0
for b in range(NB):
    band = bisect.bisect_right(sc["edges"], b) - 1
    for l in range(6):
        m = masks[l][band].cpu()
        viol += int(fr[b, l][~m].sum()); allowed_hits += int(fr[b, l][m].sum())
        # every token still gets exactly 5 experts (cond + uncond call = 2 forwards of 16x256 tokens)
        assert int(fr[b, l].sum()) == fr[b, l].sum() and int(fr[b, l].sum()) % (256 * 5) == 0
print(f"(2) compute k={K} B={B} {DOM}: masked-expert activations = {viol} (must be 0), allowed = {allowed_hits}", flush=True)
assert viol == 0 and allowed_hits > 0
# negative control: unmasked run must activate some out-of-mask experts (proves the check has power)
st0, lat_u, _ = run_tel(None)
viol0 = sum(int(st0["freq"][0][b, l][~masks[l][bisect.bisect_right(sc["edges"], b) - 1].cpu()].sum()) for b in range(NB) for l in range(6))
print(f"    negative control (unpruned): out-of-mask activations = {viol0} (must be > 0)", flush=True)
assert viol0 > 0

# --- (3) memory mode == compute mode, and the file shrinks
mm = copy.deepcopy(model)
size_full = sd_mb(model)
new_masks = slice_experts(mm, masks)
size_sl = sd_mb(mm)
frac = routed_fraction(mm)
print(f"(3) memory mode: kept routed experts/layer = {[len(b.mlp.experts.blocks) for b in mm.blocks if getattr(b,'use_moe',False)]} "
      f"| routed fraction {frac:.3f} | routed params {routed_params(mm)/1e6:.2f}M of {routed_params(model)/1e6:.2f}M "
      f"| state_dict {size_full:.1f} MB -> {size_sl:.1f} MB", flush=True)
assert size_sl < size_full
pc, pm = Pruner(model, masks, sc["edges"]).attach(), Pruner(mm, new_masks, sc["edges"]).attach()
worst = 0.0
with torch.no_grad():
    for tv in (0.0, 0.2, 0.5, 0.8, 0.96):
        t = torch.full((4,), tv, device="cuda")
        worst = max(worst, (model(x, t, y)[0] - mm(x, t, y)[0]).abs().max().item())
pc.detach(); pm.detach()
print(f"    max |compute - memory| over t in 0..0.96 = {worst:.2e}", flush=True)
assert worst < 1e-4

# --- (4) images from the pruned models (compute-mode already sampled above): decode a grid, compare to unpruned
import torchvision
torch.save({"masks": masks, "k": K, "B": B, "domain": DOM, "frac": frac, "mb_full": size_full, "mb_sliced": size_sl}, f"{OUT}/masks_animals_B4_k24.pt")
g = torch.cat([decode(lat_u[:8]), decode(lat_c[:8])])
torchvision.utils.save_image((g.clamp(-1, 1) + 1) / 2, f"{OUT}/grid_unpruned_top_pruned_bottom.png", nrow=8)
print("    classes:", yy[:8].tolist(), "| latent rel. change pruned vs unpruned:",
      f"{((lat_c - lat_u).norm() / lat_u.norm()).item():.3f}", flush=True)
print("STEP 5.1 DONE", flush=True)
