# ---- canary: vehicles B1 k=24, n=500, fast sanity check before the full 10k run (SESSION-BOOTSTRAP §6.13) ----
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
pr = Pruner(model, masks, sc["edges"])
res = run_config(f"canary_{DOMAIN}_B{B}_k{K}", ref, cls=cls, n=500, seed_base=999, pruner=pr,
                  meta={"mode": "compute", "domain": DOMAIN, "B": B, "k": K, "canary": True})
print(f"[canary] n=500 FID={res['fid']:.2f} wall={res['wall_s']:.0f}s", flush=True)
print("CANARY DONE", flush=True)
