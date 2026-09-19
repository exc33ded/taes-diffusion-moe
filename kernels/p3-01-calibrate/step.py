# ---- Step 3.1: calibration telemetry. Every latent x every one of 50 t-bins, one cond. forward each ----
import zipfile
LD = f"{DS}/results/telemetry/latents"
if not os.path.isdir(LD):                                    # Dataset mounted as zip
    zipfile.ZipFile(f"{DS}/results.zip").extractall("/tmp/res"); LD = "/tmp/res/telemetry/latents"
index = json.load(open(f"{LD}/index.json"))
calib_cls = {}                                               # domain -> class ids, in latent order
for v in index.values():
    calib_cls.setdefault(v["domain"], {})[v["latent_index"]] = v["class_id"]

DOMAINS, NB, BS = ["animals", "vehicles"], 50, 100
TEL = f"{RES}/telemetry"
os.makedirs(TEL, exist_ok=True)

for di, d in enumerate(DOMAINS):
    out_path = f"{TEL}/{d}.pt"
    lat = torch.load(f"{LD}/{d}.pt").cuda()                  # (500,4,32,32), z1
    y = torch.tensor([calib_cls[d][j] for j in range(len(lat))], device="cuda")
    tel = RouterTelemetry(model, n_domains=1, n_bins=NB)     # one domain per file -> domain_idx 0
    tel.register()
    t0 = time.time()
    for b in range(NB):
        g = torch.Generator("cuda").manual_seed(1000 * di + b)   # seed per (domain, bin)
        for i in range(0, len(lat), BS):
            z1, yy = lat[i:i + BS], y[i:i + BS]
            t = (b + torch.rand(len(z1), device="cuda", generator=g)) / NB   # jittered inside bin b
            z0 = torch.randn(z1.shape, device="cuda", generator=g)
            zt = t.view(-1, 1, 1, 1) * z1 + (1 - t.view(-1, 1, 1, 1)) * z0   # RF: t=0 noise, t=1 data
            tel.set_context(0, t)
            with torch.no_grad():
                model(zt, t, yy)
        if b % 5 == 4:
            print(f"[{d}] bin {b+1}/{NB}  {time.time()-t0:.0f}s", flush=True)
    tel.remove()
    s = tel.state()
    # gates: every (bin, layer) saw exactly 500 img x 256 tok x top-5 slots; router and expert counts agree
    assert (s["freq"].sum(-1) == 500 * 256 * 5).all(), s["freq"].sum(-1).unique()
    assert (s["freq"] == s["l2_count"]).all()
    print(f"[{d}] freq per (bin,layer) = {s['freq'].sum(-1).unique().tolist()}  t_seen={s['t_seen']}  "
          f"dead experts (freq==0 in all bins) per layer: {(s['freq'][0].sum(0) == 0).sum(-1).tolist()}  "
          f"mean gate {float(s['gate_sum'].sum()/s['freq'].sum()):.3f}", flush=True)
    torch.save(s, out_path)
print("STEP 3.1 DONE", flush=True)
