# ---- Step 5.2 priority 1: DOMAIN=vehicles B=B1 k=24 (memory mode == compute mode output, F23; report union fraction) ----
import zipfile
DOMAIN, B, K = "vehicles", 1, 24
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
res = run_config(f"compute_{DOMAIN}_B{B}_k{K}_seed0", ref, cls=cls, n=10000, seed_base=0,
                  pruner=pr, meta={"mode": "compute", "domain": DOMAIN, "B": B, "k": K, "union_frac": union_frac})
print(f"[union] routed fraction (memory-mode size) = {union_frac:.3f}", flush=True)
print("STEP 5.2-P1 DONE", flush=True)
