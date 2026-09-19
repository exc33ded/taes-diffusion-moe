# ---- Step 2.2: VAE latent cache for the 1000 calibration images ----
import zipfile
from PIL import Image
from torchvision import transforms as T

CJ = f"{DS}/taes-configs/calibration_images.json"
if not os.path.exists(CJ):                                   # Dataset mounted as zip
    zipfile.ZipFile(f"{DS}/taes-configs.zip").extractall("/tmp/cfg"); CJ = "/tmp/cfg/calibration_images.json"
calib = json.load(open(CJ))
print("calibration:", {d: len(v) for d, v in calib.items()}, flush=True)

tf = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor(), T.Lambda(lambda x: x * 2 - 1)])
LAT = f"{RES}/telemetry/latents"
os.makedirs(LAT, exist_ok=True)
index = {}

@torch.no_grad()
def encode(paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).cuda()
        out.append((vae.encode(x).latent_dist.mode() * 0.18215).cpu())   # deterministic, NEVER .sample()
    return torch.cat(out)

for d, recs in calib.items():
    t0 = time.time()
    lat = encode([r["path"] for r in recs])
    assert lat.shape == (500, 4, 32, 32) and not torch.isnan(lat).any(), lat.shape
    torch.save(lat, f"{LAT}/{d}.pt")
    for j, r in enumerate(recs):
        index[r["image_id"]] = {"latent_path": f"{d}.pt", "latent_index": j,
                                "class_id": r["class_id"], "domain": d}
    print(f"[{d}] {len(recs)}/{len(recs)}  {time.time()-t0:.1f}s  latents: {lat.shape}  "
          f"{lat.numel()*4/1e6:.1f} MB  mean={lat.mean():.3f} std={lat.std():.3f}", flush=True)
json.dump(index, open(f"{LAT}/index.json", "w"))
print("index entries:", len(index), "STEP 2.2 DONE", flush=True)
