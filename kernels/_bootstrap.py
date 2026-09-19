# ============================================================
# TAES session bootstrap (SESSION-BOOTSTRAP.md §1 + §1a) — script form
# Prepended to every kernel's main.py by kernels/build.py
# ============================================================
import os, sys, glob, json, time, subprocess
import torch

WORK  = "/kaggle/working"
DSM   = "/tmp/EfficientMoE/DSMoE"      # /tmp: keep 1.1 GB ckpt + clone out of kernel output
CKDIR = "/tmp/ckpt/DSMoE-S-E48"
DS    = "/kaggle/input/datasets/mohammedsarim/taes-artifacts"
SNAP  = f"{WORK}/snapshot"
RES   = f"{WORK}/results"
STEPS, CFG, SEED = 25, 1.5, 0          # LOCKED — see results-log.md

print("mounts:", {p: os.listdir(p)[:5] for p in ["/kaggle/input", "/kaggle/input/datasets",
      "/kaggle/input/competitions"] if os.path.isdir(p)}, flush=True)

# ---- 0. GPU guard -------------------------------------------------------
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]} | torch {torch.__version__}", flush=True)
if cap < (7, 0):
    raise SystemExit("STOP: Pascal/P100 unsupported. Need T4.")

# ---- 1. deps ------------------------------------------------------------
subprocess.run("pip install -q omegaconf timm diffusers torchdiffeq "
               "huggingface_hub pytorch-fid", shell=True)
for d in ["smoke", "baseline", "telemetry", "scores", "figures"]:
    os.makedirs(f"{RES}/{d}", exist_ok=True)

# ---- 2. code ------------------------------------------------------------
if not os.path.isdir("/tmp/EfficientMoE"):
    subprocess.run(f"git clone --depth 1 https://github.com/yhlleo/EfficientMoE.git "
                   f"/tmp/EfficientMoE", shell=True)

# ---- 3. weights (1.11 GB, ~30 s) ---------------------------------------
if not os.path.exists(f"{CKDIR}/checkpoints/0700000.pt"):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="YHLLEO/DSMoE-S-E48", local_dir=CKDIR,
                      allow_patterns=["*.pt", "*.yaml"])

# ---- 4. patch: models/modules.py imports `fla` unconditionally ----------
P   = f"{DSM}/models/modules.py"
src = open(P).read()
OLD = "from fla.layers.kda import KimiDeltaAttention"
if OLD in src and "TAES: fla unused" not in src:
    open(P, "w").write(src.replace(OLD, '''try:
    from fla.layers.kda import KimiDeltaAttention
except ModuleNotFoundError:                      # TAES: fla unused (use_kda=False)
    class KimiDeltaAttention:
        def __init__(self, *a, **k):
            raise RuntimeError("fla not installed; use_kda must stay False")'''))
    print("patched modules.py", flush=True)

# ---- 5. build + load (EMA weights, ALWAYS) ------------------------------
sys.path.insert(0, DSM); os.chdir(DSM)
from omegaconf import OmegaConf
from models.models_DSMoE import DiT

cfg   = OmegaConf.load(glob.glob(f"{CKDIR}/config_*.yaml")[0])
model = DiT(**cfg.model.params)
ck    = torch.load(f"{CKDIR}/checkpoints/0700000.pt", map_location="cpu",
                   weights_only=False)
miss, unexp = model.load_state_dict(ck["ema"], strict=False)   # never ck["model"]
assert not miss and not unexp, (miss[:5], unexp[:5])
del ck
model = model.to("cuda").eval()

MOE_BLOCKS = [i for i, b in enumerate(model.blocks) if getattr(b, "use_moe", False)]
assert MOE_BLOCKS == [1, 3, 5, 7, 9, 11], MOE_BLOCKS

# ---- 6. sampler (VAE decode MUST be chunked; batch 64 OOMs on T4) -------
from diffusers.models import AutoencoderKL
from diffusion.rectified_flow import RectifiedFlow

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda").eval()
rf  = RectifiedFlow(model)

@torch.no_grad()
def sample_latents(n, seed=SEED, steps=STEPS, cfg_scale=CFG):
    g = torch.Generator("cuda").manual_seed(seed)
    z = torch.randn(n, 4, 32, 32, device="cuda", generator=g)
    y = torch.randint(0, 1000, (n,), device="cuda", generator=g)
    out = rf.sample(z, y, torch.full_like(y, 1000), sample_steps=steps, cfg=cfg_scale)
    return out[-1], y

@torch.no_grad()
def decode(lat, chunk=8):
    return torch.cat([vae.decode(lat[i:i+chunk] / 0.18215).sample
                      for i in range(0, len(lat), chunk)])

@torch.no_grad()
def sample(n, **kw):
    lat, y = sample_latents(n, **kw)
    return decode(lat), y

print(f"ready | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params "
      f"| MoE blocks {MOE_BLOCKS} | dataset mounted: {os.path.isdir(DS)}", flush=True)

# ---- §1a: ImageNet root + taes/ paths -----------------------------------
IMAGENET_ROOT = "/kaggle/input/competitions/imagenet-object-localization-challenge"
TAES_SRC      = f"{WORK}/taes/src"
TAES_CONFIGS  = f"{WORK}/taes/configs"
os.makedirs(TAES_SRC, exist_ok=True)
os.makedirs(TAES_CONFIGS, exist_ok=True)
print("imagenet mounted:", os.path.isdir(IMAGENET_ROOT), flush=True)
