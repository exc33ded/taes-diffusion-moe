# ---- Step 4.1: noise-floor null. Per domain: two disjoint 250-image halves (independent noise seeds), plus an unconditional (y=1000) pass on all 500 ----
import zipfile
LD = f"{DS}/results/telemetry/latents"
if not os.path.isdir(LD):
    zipfile.ZipFile(f"{DS}/results.zip").extractall("/tmp/res"); LD = "/tmp/res/telemetry/latents"
index = json.load(open(f"{LD}/index.json"))
calib_cls = {}
for v in index.values():
    calib_cls.setdefault(v["domain"], {})[v["latent_index"]] = v["class_id"]

DOMAINS, NB, BS = ["animals", "vehicles"], 50, 125
TEL = f"{RES}/telemetry_p4"
os.makedirs(TEL, exist_ok=True)

def run(lat, y, seed0, out_path, tag):
    tel = RouterTelemetry(model, n_domains=1, n_bins=NB); tel.register()
    t0 = time.time()
    for b in range(NB):
        g = torch.Generator("cuda").manual_seed(seed0 + b)
        for i in range(0, len(lat), BS):
            z1, yy = lat[i:i + BS], y[i:i + BS]
            t = (b + torch.rand(len(z1), device="cuda", generator=g)) / NB
            z0 = torch.randn(z1.shape, device="cuda", generator=g)
            zt = t.view(-1, 1, 1, 1) * z1 + (1 - t.view(-1, 1, 1, 1)) * z0
            tel.set_context(0, t)
            with torch.no_grad():
                model(zt, t, yy)
    tel.remove(); s = tel.state()
    assert (s["freq"].sum(-1) == len(lat) * 256 * 5).all() and (s["freq"] == s["l2_count"]).all()
    torch.save(s, out_path); print(f"[{tag}] n={len(lat)} done {time.time()-t0:.0f}s", flush=True)

for di, d in enumerate(DOMAINS):
    lat = torch.load(f"{LD}/{d}.pt").cuda()
    y = torch.tensor([calib_cls[d][j] for j in range(len(lat))], device="cuda")
    perm = torch.randperm(len(lat), generator=torch.Generator().manual_seed(4000 + di)).cuda()
    for h in range(2):
        idx = perm[h * 250:(h + 1) * 250]
        run(lat[idx], y[idx], 10000 + 1000 * di + 100 * h, f"{TEL}/{d}_h{h}.pt", f"{d} half{h}")
    run(lat, torch.full_like(y, 1000), 20000 + 1000 * di, f"{TEL}/{d}_uncond.pt", f"{d} uncond")
print("STEP 4.1 DONE", flush=True)
