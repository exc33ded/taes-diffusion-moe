# ---- Phase 5 sweep/baseline lane: one parametrised kernel; edit the CONFIG line before each push ----
# SCORE: "std" = TAES scores from the Dataset; otherwise a key of BASELINE_I (extra taes/src/baseline_scores.py), e.g. "freq", "reap", "rand_s0"
# SEED: generation seed_base. Run id keeps the legacy name for std/seed0.
import zipfile
DOMAIN, B, K, N, SCORE, SEED = "animals", 4, 24, 10000, "freq", 0   # CONFIG (N=500 -> canary)
def find(rel, zipname, zsub):
    for base in [DS, "/kaggle/input/taes-artifacts"]:
        p = f"{base}/{rel}"
        if os.path.exists(p): return p
    zipfile.ZipFile(f"{DS}/{zipname}").extractall("/tmp/ds_" + zipname); return f"/tmp/ds_{zipname}/{zsub}"
cd_ = find("taes-configs/domains.json", "taes-configs.zip", "domains.json")
cls = json.load(open(cd_))["domains"][DOMAIN]["class_ids"]
r = np.load(find(f"results/fid_ref/{DOMAIN}_train10k.npz", "results.zip", f"fid_ref/{DOMAIN}_train10k.npz"))
ref = (r["mu"], r["sigma"])
if SCORE == "std":
    sc = torch.load(find(f"results/scores/{DOMAIN}_B{B}.pt", "results.zip", f"scores/{DOMAIN}_B{B}.pt"))
else:
    key = f"{DOMAIN}_B{B}_{SCORE}"
    sc = {"I": torch.tensor(BASELINE_I[key]), "edges": [round(i * 50 / B) for i in range(B + 1)]}
masks = build_masks(sc["I"], K)
union_frac = float(sum(m.any(0).sum().item() for m in masks) / (48 * len(masks)))
pr = Pruner(model, masks, sc["edges"])
canary = N < 10000
tag = "" if SCORE == "std" else f"_{SCORE}"
res = run_config(f"{'canary_' if canary else 'compute_'}{DOMAIN}_B{B}_k{K}{tag}" + ("" if canary else f"_seed{SEED}"), ref, cls=cls, n=N,
                 seed_base=999 if canary else SEED, pruner=pr,
                 meta={"mode": "compute", "domain": DOMAIN, "B": B, "k": K, "score": SCORE, "seed": SEED, "union_frac": union_frac, "canary": canary})
print(f"[union] routed fraction = {union_frac:.3f}  FID={res['fid']:.3f} n={res['n']} wall={res['wall_s']:.0f}s", flush=True)
print("SWEEP STEP DONE", flush=True)
