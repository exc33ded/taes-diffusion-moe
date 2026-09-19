# ---- Step 2.3: domains.json + calibration_images.json (CPU-only logic) ----
import re
import numpy as np

lines = [l.rstrip("\n").split(" ", 1) for l in open(f"{IMAGENET_ROOT}/LOC_synset_mapping.txt", encoding="utf-8")]
assert len(lines) == 1000 and lines[0][0] == "n01440764" and lines[0][1].startswith("tench"), lines[0]
WNID = [w for w, _ in lines]
LABEL = [l for _, l in lines]

def _hit(kw, label):
    return re.search(r"\b" + re.escape(kw) + r"\b", label, re.I) is not None   # word-boundary, NEVER substring

def select(kws, excl):
    return [i for i, l in enumerate(LABEL)
            if any(_hit(k, l) for k in kws) and not any(_hit(x, l) for x in excl)]

veh_all = select(VEHICLE_KW, VEHICLE_EXCLUDE)
ani_all = select(ANIMAL_KW, ANIMAL_EXCLUDE)
overlap = sorted(set(veh_all) & set(ani_all))
print(f"keyword matches: animals={len(ani_all)} vehicles={len(veh_all)} overlap={len(overlap)}", flush=True)
assert not overlap, overlap

ani_sel = sorted(int(i) for i in np.random.default_rng(0).choice(ani_all, 100, replace=False))
veh_sel = veh_all                                   # ImageNet-1k has no more
print(f"selected: animals={len(ani_sel)} vehicles={len(veh_sel)}", flush=True)

TRAIN = next(p for p in [f"{IMAGENET_ROOT}/ILSVRC/Data/CLS-LOC/train", f"{IMAGENET_ROOT}/train"]
             if os.path.isdir(f"{p}/n01440764"))
print("train dir:", TRAIN, flush=True)

def calibration(classes, seed, total=500):
    rng = np.random.default_rng(seed)
    base, rem = divmod(total, len(classes))
    extra = set(rng.permutation(len(classes))[:rem].tolist())      # seeded remainder assignment
    recs = []
    for j, c in enumerate(classes):
        n = base + (1 if j in extra else 0)
        files = sorted(os.listdir(f"{TRAIN}/{WNID[c]}"))
        for f in rng.choice(files, n, replace=False):
            recs.append({"image_id": f.rsplit(".", 1)[0], "path": f"{TRAIN}/{WNID[c]}/{f}",
                         "class_id": c, "wnid": WNID[c], "label": LABEL[c]})
    return recs

calib = {"animals": calibration(ani_sel, seed=0), "vehicles": calibration(veh_sel, seed=1)}
for d, recs in calib.items():
    assert len(recs) == 500 and len({r["image_id"] for r in recs}) == 500, (d, len(recs))
    for r in recs:                                                 # class <-> wnid <-> path pairing
        assert WNID[r["class_id"]] == r["wnid"] and r["image_id"].startswith(r["wnid"] + "_")
        assert os.path.isfile(r["path"])
    print(f"[{d}] 500/500 ok; classes used: {len({r['class_id'] for r in recs})}", flush=True)
print("spot-check:", [(r["image_id"], r["class_id"], r["label"][:20])
                      for r in (calib["animals"][0], calib["vehicles"][0])], flush=True)

domains = {
    "seed": {"class_draw_animals": 0, "calib_animals": 0, "calib_vehicles": 1},
    "matching": "word-boundary regex, case-insensitive, against full synset label; >=1 keyword and 0 exclusions",
    "keywords": {"animals": ANIMAL_KW, "vehicles": VEHICLE_KW},
    "exclusions": {"animals": ANIMAL_EXCLUDE, "vehicles": VEHICLE_EXCLUDE},
    "match_counts": {"animals": len(ani_all), "vehicles": len(veh_all), "overlap": 0},
    "selected_counts": {"animals": len(ani_sel), "vehicles": len(veh_sel)},
    "domains": {"animals": {"class_ids": ani_sel, "labels": [LABEL[i] for i in ani_sel]},
                "vehicles": {"class_ids": veh_sel, "labels": [LABEL[i] for i in veh_sel]}},
    "note": "100 animal vs 69 vehicle classes: ImageNet-1k taxonomy is animal-skewed (limitation, paper 6.4).",
}
json.dump(domains, open(f"{TAES_CONFIGS}/domains.json", "w"), indent=1)
json.dump(calib, open(f"{TAES_CONFIGS}/calibration_images.json", "w"), indent=1)
print("wrote", os.listdir(TAES_CONFIGS), "STEP 2.3 DONE", flush=True)
