"""python kernels/build.py <kernel-dir> [extra.py ...]
main.py = _bootstrap.py + extras (paths relative to repo root, e.g. taes/src/hooks.py) + <kernel-dir>/step.py
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
kdir = root / "kernels" / sys.argv[1]
parts = [root / "kernels" / "_bootstrap.py"] + [root / p for p in sys.argv[2:]]
step = kdir / "step.py"
if step.exists():
    parts.append(step)
(kdir / "main.py").write_text("\n\n".join(p.read_text() for p in parts))
print("built", kdir / "main.py")
