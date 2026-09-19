"""Sample -> FID for one config. Import-free module (after bootstrap, fid.py, prune.py); uses bootstrap globals model, rf, decode, RES, CFG, STEPS.
Locked (SESSION-BOOTSTRAP §4): euler, 25 steps, CFG 1.5, fp32, chunk 64, seed = seed_base*1000003 + chunk_start_index,
z then y drawn from one CUDA generator per chunk (same order as bootstrap.sample_latents). cls=None -> all 1000 classes (Phase-1 baseline)."""
import torchvision


@torch.no_grad()
def run_config(run_id, ref, cls=None, n=10000, seed_base=0, pruner=None, meta=None, bs=64):
    out = f"{RES}/{run_id}"; os.makedirs(out, exist_ok=True)
    cls_t = None if cls is None else torch.tensor(cls, device="cuda")
    if pruner: pruner.attach()
    A, t0, tf = [], time.time(), 0.0
    for i in range(0, n, bs):
        m = min(bs, n - i)
        g = torch.Generator("cuda").manual_seed(seed_base * 1000003 + i)
        z = torch.randn(m, 4, 32, 32, device="cuda", generator=g)
        y = torch.randint(0, 1000 if cls is None else len(cls), (m,), device="cuda", generator=g)
        if cls is not None: y = cls_t[y]
        t1 = time.time()
        lat = rf.sample(z, y, torch.full_like(y, 1000), sample_steps=STEPS, cfg=CFG)[-1]
        torch.cuda.synchronize(); tf += time.time() - t1
        img = to_u8(decode(lat))
        if i == 0:
            torchvision.utils.save_image(img[:16].float() / 255, f"{out}/samples16.png", nrow=8)
        A.append(acts(img))
        if (i // bs) % 10 == 0:
            print(f"[{run_id}] {i+m}/{n} {time.time()-t0:.0f}s", flush=True)
    if pruner: pruner.detach()
    A = np.concatenate(A); f = fid(A, ref)
    res = {"run_id": run_id, "fid": f, "n": n, "seed_base": seed_base, "meta": meta or {},
           "wall_s": time.time() - t0, "transformer_plus_sampler_s": tf, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    print(f"[{run_id}] FID-{n//1000}k = {f:.3f}", flush=True)
    return res
