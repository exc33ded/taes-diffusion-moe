# ---- Step 5.2a: per-domain FID reference stats from real ImageNet-train images (10k/domain, calibration images excluded) ----
from PIL import Image
from torch.utils.data import Dataset, DataLoader
TRAIN = f"{IMAGENET_ROOT}/ILSVRC/Data/CLS-LOC/train"
WNID = [l.split(" ", 1)[0] for l in open(f"{IMAGENET_ROOT}/LOC_synset_mapping.txt", encoding="utf-8")]
import zipfile
CD = f"{DS}/taes-configs"
if not os.path.isdir(CD):
    zipfile.ZipFile(f"{DS}/taes-configs.zip").extractall("/tmp/cfg"); CD = "/tmp/cfg"
dom = json.load(open(f"{CD}/domains.json"))["domains"]
cal = json.load(open(f"{CD}/calibration_images.json"))
N = 10000

def center_crop_arr(img, size=256):          # ADM's preprocessing (guided-diffusion image_datasets.py)
    while min(*img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    s = size / min(*img.size)
    img = img.resize(tuple(round(x * s) for x in img.size), resample=Image.BICUBIC)
    a = np.array(img); cy, cx = (a.shape[0] - size) // 2, (a.shape[1] - size) // 2
    return a[cy:cy + size, cx:cx + size]

class DS_(Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        return torch.from_numpy(center_crop_arr(Image.open(self.p[i]).convert("RGB"))).permute(2, 0, 1)

os.makedirs(f"{RES}/fid_ref", exist_ok=True)
for d in ["animals", "vehicles"]:
    classes = dom[d]["class_ids"]; excl = {c["image_id"] for c in cal[d]}
    rng = np.random.default_rng(5000 + len(d)); per = -(-N // len(classes)); paths = []
    for c in classes:                         # stratified: equal images per class, like the class-uniform generation
        fs = sorted(f for f in os.listdir(f"{TRAIN}/{WNID[c]}") if f.rsplit(".", 1)[0] not in excl)
        paths += [f"{TRAIN}/{WNID[c]}/{f}" for f in rng.choice(fs, per, replace=False)]
    paths = [paths[i] for i in sorted(rng.choice(len(paths), N, replace=False))]
    t0 = time.time(); A = []
    for i, x in enumerate(DataLoader(DS_(paths), batch_size=100, num_workers=4)):
        A.append(acts(x))
        if i % 20 == 0: print(f"[{d}] {i*100}/{N} {time.time()-t0:.0f}s", flush=True)
    A = np.concatenate(A); mu, sg = stats(A)
    np.savez(f"{RES}/fid_ref/{d}_train10k.npz", mu=mu, sigma=sg)
    print(f"[{d}] classes={len(classes)} n={len(A)} | interleaved-half FID (5k vs 5k, real vs real) = "
          f"{fid(A[::2], stats(A[1::2])):.2f}", flush=True)
# preprocessing check vs the locked ADM stats: 10 random train images per class (all 1000), same pipeline -> FID should be small
rng = np.random.default_rng(77); paths = []
for c in range(1000):
    fs = sorted(os.listdir(f"{TRAIN}/{WNID[c]}")); paths += [f"{TRAIN}/{WNID[c]}/{f}" for f in rng.choice(fs, 10, replace=False)]
A = np.concatenate([acts(x) for x in DataLoader(DS_(paths), batch_size=100, num_workers=4)])
print(f"[fullclass real 10k] FID vs ADM stats = {fid(A, f'{DS}/results/baseline/fid_stats_imagenet256.npz'):.2f}", flush=True)
print("STEP 5.2a DONE", flush=True)
