"""python kernels/run.py <kernel-dir> [extra.py ...] [--pull REGEX]
Build main.py, push, poll to completion, print the log (+ optionally pull files matching REGEX into <dir>/out/).
Needs KAGGLE_API_TOKEN in env (source .env first)."""
import sys, os, json, re, time, subprocess, pathlib
os.environ["PYTHONUTF8"] = "1"; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = pathlib.Path(__file__).resolve().parent.parent
args = sys.argv[1:]
pull = args[args.index("--pull") + 1] if "--pull" in args else None
if pull:
    i = args.index("--pull"); args = args[:i] + args[i + 2:]
name, extras = args[0], args[1:]
kdir = root / "kernels" / name
slug = json.loads((kdir / "kernel-metadata.json").read_text())["id"]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, encoding="utf-8", cwd=root)

print(sh(f"python kernels/build.py {name} {' '.join(extras)}").stdout.strip())
print(sh(f"kaggle kernels push -p kernels/{name}").stdout.strip(), flush=True)
while True:
    time.sleep(20)
    s = sh(f"kaggle kernels status {slug}").stdout
    print(s.strip().split('"')[-2], flush=True)
    if re.search(r"COMPLETE|ERROR|CANCEL", s):
        break
out = kdir / "out"; out.mkdir(exist_ok=True)
sh(f"kaggle kernels output {slug} -p kernels/{name}/out --file-pattern \"\\.log$\" -o")
try:
    for e in json.loads((out / f"{slug.split('/')[1]}.log").read_text(encoding="utf-8")):
        print(e.get("data", ""), end="")
except Exception as ex:
    print("log parse failed:", ex)
if pull:
    print(sh(f"kaggle kernels output {slug} -p kernels/{name}/out --file-pattern \"{pull}\" -o").stdout)
