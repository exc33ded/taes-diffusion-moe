# ---- Pivot gate, stage 1: can a T4 host DiT-XL/2 acceleration work? Timing + VRAM only, no FID. ----
import time
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
os.makedirs(f"{RES}/g1", exist_ok=True)
out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
for dt, bs in [(torch.float16, 16), (torch.float32, 8)]:
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=dt).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    tf = {"t": 0.0, "n": 0}
    orig = pipe.transformer.forward
    def timed(*a, _o=orig, **k):
        torch.cuda.synchronize(); t0 = time.time(); r = _o(*a, **k); torch.cuda.synchronize()
        tf["t"] += time.time() - t0; tf["n"] += 1; return r
    pipe.transformer.forward = timed
    labels = list(range(0, 1000, 1000 // bs))[:bs]
    for steps in (25, 50):
        tf.update(t=0.0, n=0); torch.cuda.reset_peak_memory_stats()
        g = torch.Generator("cuda").manual_seed(0)
        pipe(class_labels=labels, num_inference_steps=steps, guidance_scale=1.5, generator=g)   # warmup
        tf.update(t=0.0, n=0); torch.cuda.synchronize(); t0 = time.time()
        pipe(class_labels=labels, num_inference_steps=steps, guidance_scale=1.5, generator=g)
        torch.cuda.synchronize(); tot = time.time() - t0
        key = f"{str(dt).split('.')[-1]}_bs{bs}_steps{steps}"
        out[key] = {"s_per_img_total": tot / bs, "s_per_img_transformer": tf["t"] / bs, "transformer_calls": tf["n"],
                    "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "fid10k_gpu_hours_projected": tot / bs * 10000 / 3600}
        print(key, out[key], flush=True)
    del pipe; torch.cuda.empty_cache()
json.dump(out, open(f"{RES}/g1/timing.json", "w"), indent=1)
print("G1 DONE", flush=True)
