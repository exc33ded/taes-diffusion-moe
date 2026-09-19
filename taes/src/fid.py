"""FID helpers (pytorch-fid 0.3.0, pool3 2048-d). Import-free module (prepended after the bootstrap; needs torch, numpy as np, os).
Images enter as uint8 NCHW (0..255); pytorch-fid's InceptionV3 takes [0,1] and does the 299 resize itself."""
import numpy as np
import torch

_inc = None


def _net():
    global _inc
    if _inc is None:
        from pytorch_fid.inception import InceptionV3
        _inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    return _inc


@torch.no_grad()
def acts(u8):
    """(N,3,H,W) uint8 -> (N,2048) float64 numpy."""
    return _net()(u8.cuda().float() / 255.0)[0].squeeze(-1).squeeze(-1).double().cpu().numpy()


def to_u8(x):
    """VAE-decoded [-1,1] float -> uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def stats(a):
    return a.mean(0), np.cov(a, rowvar=False)


def fid(a, ref):
    """a: (N,2048) activations; ref: (mu, sigma) or npz path/dict with mu, sigma."""
    from pytorch_fid.fid_score import calculate_frechet_distance
    mu, sg = stats(a)
    if isinstance(ref, str):
        ref = np.load(ref)
    m2, s2 = (ref["mu"], ref["sigma"]) if not isinstance(ref, tuple) else ref
    return float(calculate_frechet_distance(mu, sg, m2, s2))
