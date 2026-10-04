# TAES paper — TMLR submission folder

Target venue: **Transactions on Machine Learning Research (TMLR)**, empirical-study framing. Decision recorded 2026-10-04 in `../notes/results-log.md`.

## Layout
```
main.tex                 TMLR template (anonymous by default); \input's the sections
sections/*.tex           abstract, intro, related, method, experiments, results, analysis, limitations, appendix
references.bib           GENERATED from the arXiv API by tools/make_bib.py + tools/build_refs.py (do not hand-edit)
CITATIONS.md             verification log: what was verified, what was deliberately NOT cited
CLAIMS-AUDIT.md          every claim -> evidence -> status; update as results land
figures/                 diagnostic_1-3.png, noisefloor.png (regenerate final versions from ../results/figures)
tools/lint.py            cite/label/ref/figure/brace checks + open-\todo count (no LaTeX needed)
tmlr.sty, tmlr.bst, math_commands.tex, fancyhdr.sty, tmlr-style-LICENSE   official style files
legacy/                  the pre-TMLR skeleton (method text was ported into sections/method.tex)
```

## Build
No LaTeX is installed on the dev machine, so the paper has **not been compiled**. Either upload this folder to Overleaf, or install MiKTeX/TeX Live and run `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Run `python tools/lint.py` before every commit; it must report no errors, and the open-`\todo` count must be 0 before submission.

Regenerate the bibliography after editing `tools/build_refs.py` (or adding arXiv IDs to `tools/make_bib.py`):
```
python tools/make_bib.py && python tools/build_refs.py
```

## TMLR facts (verified from jmlr.org/tmlr on 2026-10-04)
- Official style: <https://github.com/JmlrOrg/tmlr-style-file> (linked from the author guide). Use `\usepackage{tmlr}` for submission, `[preprint]` for arXiv, `[accepted]` for camera-ready.
- Double-blind; **non-anonymous submissions are rejected**. Submit through OpenReview with a complete profile (affiliation, conflicts, history, action-editor suggestions, funding/IRB disclosures).
- Length: "any length", justified by content; submissions over 12 pages face longer review (reviews due in 2 weeks for ≤12 pages, 4 weeks otherwise; ~9 weeks to decision). Appendices after the references are allowed; supplementary material ≤100 MB (PDF/ZIP).
- Acceptance: technical soundness ("accurate and convincing evidence") and audience interest; **novelty of the method is not required**; modest contributions acceptable; reproducibility studies welcome. TMLR rejects papers that wrongly claim novelty.
- A preprint may exist online (not in another venue's official proceedings). All submissions are CC BY 4.0.

## Submission checklist
1. All numbers final; no `\todo`; `CLAIMS-AUDIT.md` has no "pending" or "drop-if-fails" rows left in the text.
2. Citation venues checked (see `CITATIONS.md`); no unverified papers cited.
3. Seeds / error bars on every load-bearing comparison; second backbone included.
4. Supplementary zip: code, calibration statistics, scores, run table, sample grids; checkpoint hashes and library versions in the appendix.
5. Anonymity: no names, repo URLs or Kaggle usernames in the PDF or supplement.
6. Compile, read the PDF end to end, check page count and figures.
