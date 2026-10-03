# ---- Step 5.2 priority 2 sweep: one parametrised kernel; edit the CONFIG line before each push ----
import zipfile
DOMAIN, B, K, N = "animals", 1, 16, 500   # CONFIG (N=500 -> canary)
def find(rel, zipname, zsub):
    for base in [DS, "/kaggle/input/taes-artifacts"]:
        p = f"{base}/{rel}"
        if os.path.exists(p): return p
    zipfile.ZipFile(f"{DS}/{zipname}").extractall("/tmp/ds_" + zipname); return f"/tmp/ds_{zipname}/{zsub}"
cd_ = find("taes-configs/domains.json", "taes-configs.zip", "domains.json")
cls = json.load(open(cd_))["domains"][DOMAIN]["class_ids"]
r = np.load(find(f"results/fid_ref/{DOMAIN}_train10k.npz", "results.zip", f"fid_ref/{DOMAIN}_train10k.npz"))
ref = (r["mu"], r["sigma"])
sc = torch.load(find(f"results/scores/{DOMAIN}_B{B}.pt", "results.zip", f"scores/{DOMAIN}_B{B}.pt"))
masks = build_masks(sc["I"], K)
union_frac = float(sum(m.any(0).sum().item() for m in masks) / (48 * len(masks)))
pr = Pruner(model, masks, sc["edges"])
canary = N < 10000
res = run_config(f"{'canary_' if canary else 'compute_'}{DOMAIN}_B{B}_k{K}" + ("" if canary else "_seed0"), ref, cls=cls, n=N,
                 seed_base=999 if canary else 0, pruner=pr,
                 meta={"mode": "compute", "domain": DOMAIN, "B": B, "k": K, "union_frac": union_frac, "canary": canary})
print(f"[union] routed fraction = {union_frac:.3f}  FID={res['fid']:.3f} n={res['n']} wall={res['wall_s']:.0f}s", flush=True)
print("SWEEP STEP DONE", flush=True)
