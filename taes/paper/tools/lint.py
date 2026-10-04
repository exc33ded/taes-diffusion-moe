"""Static checks for the paper (no LaTeX needed).  python tools/lint.py
- every \\cite{...} key exists in references.bib; unused bib keys are listed
- every \\ref/\\Cref target has a \\label; duplicate labels are errors
- every \\includegraphics file exists
- brace balance per file; open \\todo count per file (must be 0 before submission)"""
import glob, os, re, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
files = ["main.tex"] + sorted(glob.glob("sections/*.tex"))
text = {f: open(f, encoding="utf-8").read() for f in files}
bib = open("references.bib", encoding="utf-8").read()
bibkeys = set(re.findall(r"^@\w+\{([^,\s]+),", bib, re.M))
errors, warnings = [], []

cited = set()
for f, t in text.items():
    t_nc = re.sub(r"(?<!\\)%.*", "", t)                      # drop comments
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t_nc):
        for k in m.group(1).split(","):
            k = k.strip()
            cited.add(k)
            if k not in bibkeys:
                errors.append(f"{f}: cite key not in references.bib: {k}")
    bal = t_nc.count("{") - t_nc.count("}") - t_nc.count("\\{") + t_nc.count("\\}")
    if bal != 0:
        errors.append(f"{f}: unbalanced braces ({bal:+d})")

labels = {}
for f, t in text.items():
    for l in re.findall(r"\\label\{([^}]*)\}", t):
        if l in labels:
            errors.append(f"duplicate label {l} in {f} and {labels[l]}")
        labels[l] = f
for f, t in text.items():
    for m in re.finditer(r"\\(?:ref|eqref|Cref|cref|autoref)\{([^}]*)\}", t):
        if m.group(1) not in labels:
            errors.append(f"{f}: \\ref to missing label {m.group(1)}")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
        if not os.path.exists(g):
            errors.append(f"{f}: missing figure file {g}")

todos = {f: len(re.findall(r"\\todo\{", t)) for f, t in text.items()}
unused = sorted(bibkeys - cited)
print(f"bib entries: {len(bibkeys)} | cited: {len(cited)} | unused: {len(unused)}")
if unused:
    print("  unused keys:", ", ".join(unused))
print("open \\todo per file:", {f: n for f, n in todos.items() if n}, "| total:", sum(todos.values()))
for w in warnings:
    print("WARN", w)
for e in errors:
    print("ERROR", e)
print("LINT FAILED" if errors else "lint ok (structure only; not a compile)")
sys.exit(1 if errors else 0)
