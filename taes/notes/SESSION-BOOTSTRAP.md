# TAES — Kaggle session bootstrap

**Canonical setup for any new Kaggle notebook, any phase.**
Established at the end of Phase 1 (2026-07-31). Updated end of Phase 2 (2026-08-22).
Everything here is verified working.

> **Carry this file forward.** Every phase kickoff prompt must reference it, and any change
> made here must be committed and reflected in the next phase's prompt. Sessions die without
> warning; this file is what makes that cheap.

---

## 0. Notebook settings — do this in the UI before running anything

| Setting | Value | Why |
|---|---|---|
| Accelerator | **GPU T4 ×2** | P100 is sm_60; torch 2.10 ships no kernels for it. Every CUDA call fails with `no kernel image is available for execution on the device`. |
| Internet | **On** | HuggingFace checkpoint + pip |
| Add Input → Datasets | `mohammedsarim/taes-artifacts` | mounts at `/kaggle/input/datasets/mohammedsarim/taes-artifacts` (non-standard path — not `/kaggle/input/taes-artifacts`) |
| Add Input → Competitions | `imagenet-object-localization-challenge` | **needed from Phase 2 onward** (real ImageNet images for calibration/domain sampling). Search both the Competitions *and* Datasets tabs in Add Input — it only showed up under Competitions. First attach requires accepting the competition rules once (free, one click, on a separate Kaggle page) before it'll mount. Mounts at `/kaggle/input/competitions/imagenet-object-localization-challenge` (**not** `/kaggle/input/imagenet-object-localization-challenge`). |
| Add-ons → Secrets | `KAGGLE_USERNAME`, `KAGGLE_KEY` | needed to push Dataset versions. Never upload `kaggle.json` into the notebook. |

Changing the accelerator requires a **full session restart** — the dropdown alone does
nothing to a running kernel.

---

## 0b. CLI workflow (current, since 2026-09-19) — supersedes paste-a-cell

Code lives in the repo and runs on Kaggle via `kaggle kernels push`. No notebook UI needed.

```
kernels/_bootstrap.py        §1 + §1a as a script (ckpt + clone in /tmp — see F16)
kernels/build.py             main.py = _bootstrap + extra files + <kernel>/step.py
kernels/run.py               build → push → poll → print log → optional --pull REGEX
kernels/<name>/step.py       the step's code    kernels/<name>/kernel-metadata.json
```

```bash
set -a; . ./.env; set +a          # KAGGLE_API_TOKEN (mohammedsarim), gitignored
python kernels/run.py p2-01-hooks taes/src/hooks.py --pull 'smoke_2_1\.pt'
```

- `kernel-metadata.json`: `machine_shape: NvidiaTeslaT4` (= T4 ×2), `enable_gpu/internet: true`,
  `dataset_sources: [mohammedsarim/taes-artifacts]`, `competition_sources: [imagenet-object-localization-challenge]`.
- Extra files passed to `run.py` are **prepended** after the bootstrap (import-free modules like `hooks.py`).
- A kernel's `/kaggle/working` is its output; pull only what you need with `--pull` (a regex on file names).
- Set `PYTHONUTF8=1` on Windows or `kaggle kernels output` dies with a `charmap` error (`run.py` does).
- Each bootstrap costs ~2 min of a fresh container. Batch related work into one kernel when a step is cheap.
- Dataset push from CLI: see §3 CLI variant (stage first — F17).

---

## 1. Bootstrap cell — run first in every session

Idempotent. On a cold session it does everything (~2 min); on a warm one it skips the
downloads. Re-run it freely.

```python
# ============================================================
# TAES session bootstrap — safe to re-run, safe after session death
# ============================================================
import os, sys, glob, json, time, subprocess
import torch

WORK  = "/kaggle/working"
DSM   = f"{WORK}/EfficientMoE/DSMoE"
CKDIR = f"{WORK}/ckpt/DSMoE-S-E48"
DS    = "/kaggle/input/datasets/mohammedsarim/taes-artifacts"
SNAP  = f"{WORK}/snapshot"
RES   = f"{WORK}/results"
STEPS, CFG, SEED = 25, 1.5, 0          # LOCKED — see results-log.md

# ---- 0. GPU guard -------------------------------------------------------
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]} | torch {torch.__version__}")
if cap < (7, 0):
    raise SystemExit("STOP: Pascal/P100 unsupported. Session options -> "
                     "Accelerator -> GPU T4 x2 -> restart session.")

# ---- 1. deps ------------------------------------------------------------
subprocess.run("pip install -q omegaconf timm diffusers torchdiffeq "
               "huggingface_hub pytorch-fid", shell=True)
for d in ["smoke", "baseline", "telemetry", "scores", "figures"]:
    os.makedirs(f"{RES}/{d}", exist_ok=True)

# ---- 2. code ------------------------------------------------------------
if not os.path.isdir(f"{WORK}/EfficientMoE"):
    subprocess.run(f"git clone --depth 1 https://github.com/yhlleo/EfficientMoE.git "
                   f"{WORK}/EfficientMoE", shell=True)

# ---- 3. weights (1.11 GB, ~30 s) ---------------------------------------
if not os.path.exists(f"{CKDIR}/checkpoints/0700000.pt"):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="YHLLEO/DSMoE-S-E48", local_dir=CKDIR,
                      allow_patterns=["*.pt", "*.yaml"])

# ---- 4. patch: models/modules.py imports `fla` unconditionally ----------
#      use_kda=False everywhere, so the import is dead code. Stub it.
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
    print("patched modules.py")

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
      f"| MoE blocks {MOE_BLOCKS} | dataset mounted: {os.path.isdir(DS)}")
```

Expected final line:

```
ready | 69.2M params | MoE blocks [1, 3, 5, 7, 9, 11] | dataset mounted: True
```

---

## 1a. Phase-2-onward additions — ImageNet root + `taes/` repo paths

Not part of the core bootstrap cell (kept separate so the cell above still works standalone for
Phase 1-style smoke tests with no ImageNet dependency). Run once, right after the bootstrap cell,
in any session that touches calibration, domains, or latent caching:

```python
IMAGENET_ROOT = "/kaggle/input/competitions/imagenet-object-localization-challenge"
TAES_SRC      = f"{WORK}/taes/src"
TAES_CONFIGS  = f"{WORK}/taes/configs"
os.makedirs(TAES_SRC, exist_ok=True)
os.makedirs(TAES_CONFIGS, exist_ok=True)
print("imagenet mounted:", os.path.isdir(IMAGENET_ROOT))
```

**`os.makedirs` before every `%%writefile`.** `%%writefile path/to/file.py` does **not** create
parent directories — it just errors `FileNotFoundError` if they don't exist yet. Always run the
`os.makedirs(..., exist_ok=True)` cell before the first `%%writefile taes/src/whatever.py` in a
session.

Files this repo now expects, mirrored to GitHub at end of phase:

| Path | Written in | Contents |
|---|---|---|
| `taes/src/hooks.py` | Phase 2, Step 2.1 | `RouterTelemetry` — router hooks, F3/F9-safe |
| `taes/configs/domains.json` | Phase 2, Step 2.3 | animal/vehicle class-ID lists, seed=0 |
| `taes/configs/calibration_images.json` | Phase 2, Step 2.3 | 500+500 `{image_id, path, class_id, wnid, label}` |
| `results/telemetry/latents/{animals,vehicles}.pt` | Phase 2, Step 2.2 | cached VAE latents, `(500,4,32,32)` each |
| `results/telemetry/latents/index.json` | Phase 2, Step 2.2 | `image_id → {latent_path, latent_index, class_id, domain}` |

Full derivation, the keyword lists, and the 100-vs-69 class-count asymmetry: `results-log.md`,
Steps 2.1–2.3 entries (2026-08-22).

---

## 2. Restoring prior artifacts from the Dataset

The Dataset holds `results.zip` (telemetry, FID stats, grids) and `patches.zip`.
The 1.11 GB checkpoint is **not** in it — it re-downloads from HF in ~30 s, and including it
would make every version push slow.

```python
import zipfile, os
for z in ["results.zip", "patches.zip", "taes-src.zip", "taes-configs.zip"]:
    p = f"{DS}/{z}"
    if os.path.exists(p):
        zipfile.ZipFile(p).extractall(f"{WORK}/restored")
        print("restored", z)
print(os.listdir(f"{WORK}/restored"))
```

`taes-src.zip` / `taes-configs.zip` only exist in the Dataset from Phase 2 onward (§3 snapshots
them alongside `results/`, since `--dir-mode zip` produces one zip per top-level folder in
`SNAP`). Copy their extracted contents into `{WORK}/taes/src` and `{WORK}/taes/configs` to pick
up `hooks.py`, `domains.json`, and `calibration_images.json` from a prior session instead of
rewriting them.

`fid_stats_imagenet256.npz` lives in `results/baseline/`. **Never regenerate it** —
recomputing with different preprocessing silently breaks comparability with FID 23.30.

---

## 3. Pushing a new Dataset version — end of every session

```python
import os, shutil, subprocess
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = s.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = s.get_secret("KAGGLE_KEY")

os.makedirs(SNAP, exist_ok=True)
shutil.copytree(RES, f"{SNAP}/results", dirs_exist_ok=True)
# Phase 2+: taes/src and taes/configs live outside RES — snapshot them too, or hooks.py /
# domains.json / calibration_images.json silently never leave the dying container.
if os.path.isdir(f"{WORK}/taes/src"):
    shutil.copytree(f"{WORK}/taes/src", f"{SNAP}/taes-src", dirs_exist_ok=True)
if os.path.isdir(f"{WORK}/taes/configs"):
    shutil.copytree(f"{WORK}/taes/configs", f"{SNAP}/taes-configs", dirs_exist_ok=True)
subprocess.run(f"pip freeze > {SNAP}/requirements-frozen.txt", shell=True)
json.dump({"title": "taes-artifacts", "id": "mohammedsarim/taes-artifacts",
           "licenses": [{"name": "CC0-1.0"}]},
          open(f"{SNAP}/dataset-metadata.json", "w"), indent=2)

r = subprocess.run(f'kaggle datasets version -p {SNAP} -m "<MESSAGE>" --dir-mode zip',
                   shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr)
```

**CLI variant (from a local machine — the one actually used for Phase 2 redo):**

```bash
set -a; . ./.env; set +a; export PYTHONUTF8=1
kaggle datasets download mohammedsarim/taes-artifacts -p D:/tstage --unzip   # 1. START from current contents (F17)
# 2. add/replace folders in D:/tstage (taes-src/, taes-configs/, results/telemetry/...) + dataset-metadata.json
cd D:/tstage && kaggle datasets version -p . -m "<MESSAGE>" --dir-mode zip     # 3. SHORT path, or [Errno 2]
kaggle datasets files mohammedsarim/taes-artifacts                              # 4. confirm every folder listed
```

⚠️ **`--dir-mode zip` is mandatory.** Without it Kaggle prints
`Skipping folder: results; use '--dir-mode'` and silently uploads nothing but loose files.
`-d` is `--delete-old-versions`, **not** dir-mode. Always read the upload list and confirm
every folder you expect is named.

---

## 4. Locked settings — changing any invalidates all comparisons

| | |
|---|---|
| Backbone | `YHLLEO/DSMoE-S-E48` · N=48 · top-k=5 · depth 12 · rectified flow · 256×256 |
| Weights | `ckpt['ema']` — **never** `ckpt['model']` |
| Sampler | `RectifiedFlow.sample(mode='euler')` · 25 steps · CFG 1.5 |
| Precision | fp32 |
| FID | `pytorch-fid` 0.3.0 vs ADM `VIRTUAL_imagenet256_labeled.npz` (10k) |
| **Baseline** | **FID-10k = 23.30** (unpruned, seed_base 0) |
| Seeding | `seed = seed_base * 1000003 + image_index` |
| GPU | T4 (sm_75+) |
| VAE encode (calibration) | `posterior.mode() * 0.18215` — **deterministic**, never `.sample()`; preprocessing `Resize(256) → CenterCrop(256) → ToTensor → ×2−1` |
| Domain classes | `taes/configs/domains.json`, keyword-matched (word-boundary + exclusions, lists stored inside the json), seed=0 — 100 animal (of 397 matches) / 69 vehicle classes (asymmetric; see results-log Step 2.3) |
| RF timestep | `t ∈ [0,1]`, **t=0 noise → t=1 data** (F15); hooks bin it into 50 bins, bin 0 = noisiest |

---

## 5. Model topology cheat-sheet — for hooks and pruning

```
MoE-layer index l ∈ 0..5   ↔   block index 2l+1   ⇒   blocks 1, 3, 5, 7, 9, 11

blocks.{2l+1}.mlp                      DSMoE
blocks.{2l+1}.mlp.gate                 TopkRouter        <- router telemetry
blocks.{2l+1}.mlp.experts.blocks[e]    MLP(384,128)      <- routed expert, e ∈ 0..47
blocks.{2l+1}.mlp.shared_experts       MLP(384,128)      <- NEVER PRUNE
```

- Shared expert is one routed-expert wide ⇒ 49 units/layer, 1 unprunable ⇒
  **hard memory floor 1/49 = 2.04% of routed params**.
- With `use_shared_expert=True` there is **no identity path inside `DSMoE`** — the shared
  expert *is* the residual. Pruning it severs the connection.
- `e_score_correction_bias` is **all zeros** in both `model` and `ema` ⇒ no load-balancing
  confound; selection and weighting rank experts identically.
- `norm_topk_prob=True` is hardcoded ⇒ returned `topk_weights` always sum to 2.5.
  **Record the pre-normalisation sigmoid instead.**
- `DSNaiveMoE.forward` applies the gate weight *inside* the expert loop ⇒ hooking expert
  output captures `gate × f_e(x)`. Hook the expert `MLP` module directly.

---

## 6. Operational gotchas learned the hard way

1. **Kaggle's console stops streaming on long cells.** The kernel keeps running. Judge
   liveness by an artifact's mtime, not the console. Log every chunk with `flush=True`.
2. **You cannot run a second cell while one is executing** — one kernel. Build progress
   reporting into the long cell itself.
3. **Save & Run All (Commit) spawns a separate batch session** that re-executes from a clean
   container and burns quota in parallel with your interactive one. Know which you started.
4. **Always checkpoint long loops to disk** and resume from them. FID-10k writes
   `act_partial.npy` every 20 chunks.
5. **Timing must be reported split** (transformer vs VAE). The VAE is a fixed cost pruning
   never touches; lumping them understates every speedup.
6. Kaggle sessions cap at **12 hours**; FID-10k is ~90 min, so it fits comfortably.
7. **`%%writefile` does not create parent directories.** `os.makedirs(..., exist_ok=True)` first,
   every time, or the write fails with `FileNotFoundError`.
8. **Never use plain substring search on ImageNet synset labels.** `"car" in label` matches
   inside Latin binomial names (`carassius`, `carcharodon`, `carduelis`, ...) and `"bus"` matches
   inside `erythrocebus` — false-positived dozens of animal classes into a vehicle keyword filter
   in Phase 2. Always use word-boundary regex: `re.search(r"\bkw\b", label)`.
9. **Add Input → Competitions, not just Datasets**, for `imagenet-object-localization-challenge`
   — it only appeared under the Competitions tab, and required accepting the competition rules
   once on a separate page before Kaggle would attach it.

---

## 7. Measured compute budget

| Quantity | Value |
|---|---|
| Sampling | 0.4475 s/image (transformer 0.3052 + VAE 0.1423), 25 steps, batch 64, T4 |
| Peak VRAM | 10.76 GB |
| FID-10k, one config | ~90 min ≈ **1.5 GPU-h** |
| Kaggle quota | ~30 GPU-h/week ⇒ **~20 configs/week** |
| Phase 5 priority 1 | 4 runs ≈ 6 h ✅ |
| Phase 5 priority 2 (k-sweep) | 28 runs ≈ 42 h — **exceeds a week's quota**; cut to one domain or use the second T4 |
| Phase 6 FID-50k × 4 | ~30 h |

**Do not reorder WORKOUT-PLAN Step 5.2's priority list — it is quota-critical.**
