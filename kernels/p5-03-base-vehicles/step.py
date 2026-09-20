# ---- Step 5.2b: unpruned baseline FID-10k, DOMAIN=vehicles ----
import zipfile
DOMAIN = "vehicles"
def find(rel, zipname, zsub):
    p = f"{DS}/{rel}"
    if os.path.exists(p): return p
    zipfile.ZipFile(f"{DS}/{zipname}").extractall("/tmp/ds_" + zipname); return f"/tmp/ds_{zipname}/{zsub}"
if DOMAIN == "full":
    cls, ref = None, find("results/baseline/fid_stats_imagenet256.npz", "results.zip", "baseline/fid_stats_imagenet256.npz")
else:
    cd = find("taes-configs/domains.json", "taes-configs.zip", "domains.json")
    cls = json.load(open(cd))["domains"][DOMAIN]["class_ids"]
    r = np.load(find(f"results/fid_ref/{DOMAIN}_train10k.npz", "results.zip", f"fid_ref/{DOMAIN}_train10k.npz")); ref = (r["mu"], r["sigma"])
res = run_config(f"unpruned_{DOMAIN}_seed0", ref, cls=cls, n=10000, seed_base=0, meta={"mode": "none", "domain": DOMAIN})
print("STEP 5.2b DONE", flush=True)
