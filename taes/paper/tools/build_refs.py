"""references_arxiv.json -> references.bib + CITATIONS.md.
Venues are filled only where the venue page/listing was seen in-session (see VENUE); everything else is cited as an arXiv preprint.
Run after make_bib.py:  python tools/build_refs.py"""
import json

d = json.load(open("references_arxiv.json", encoding="utf-8"))
KEY = {"2407.11633": "fei2024ditmoe", "2409.15557": "ganjdanesh2024diffpruning", "2412.05628": "fang2024remixdit",
       "2505.21817": "yang2025alter", "2510.11962": "guo2025mosaicdiff", "2603.05105": "liu2026diffes",
       "2606.15615": "li2026moeca", "2403.09176": "park2024switchdit", "2503.14487": "shi2025diffmoe",
       "2503.16057": "yuan2025racedit", "2510.24711": "wei2025promoe", "2503.06923": "liu2025taylorseer",
       "2604.01622": "zhang2026ecdlm", "2602.08404": "wei2026team", "2512.01252": "liu2025dsmoe",
       "2510.13999": "lasby2025reap", "2504.06792": "dong2025easyep", "2509.10377": "zhou2025dern",
       "2606.15716": "liu2026scoreexperts", "2402.14800": "lu2024notallexperts", "2410.12013": "xie2024moepruner",
       "2407.00945": "liu2024eep", "2604.04356": "jha2026ream", "2312.17238": "eliseev2023offloading",
       "1409.0575": "russakovsky2015imagenet", "2505.16056": "liang2026localrouting", "2212.09748": "peebles2023dit",
       "2209.03003": "liu2023rectifiedflow", "2112.10752": "rombach2022ldm", "2207.12598": "ho2022cfg",
       "2105.05233": "dhariwal2021adm", "1706.08500": "heusel2017fid", "2401.06066": "dai2024deepseekmoe",
       "1701.06538": "shazeer2017moe"}
ICCV = "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)"
VENUE = {"2510.11962": ("inproceedings", {"booktitle": ICCV, "pages": "1655--1664", "year": "2025"}),
         "2503.06923": ("inproceedings", {"booktitle": ICCV, "year": "2025"}),
         "2409.15557": ("misc", {"note": "ECCV 2024"}),
         "2505.16056": ("misc", {"note": "ICLR 2026"})}
VERIFIED = {"2510.11962": "verified (ICCV 2025)", "2503.06923": "verified (ICCV 2025)",
            "2409.15557": "verified (ECCV 2024)", "2505.16056": "verified (ICLR 2026)"}
ROLE = {
    "fei2024ditmoe": "Prior: reports expert preference for denoising timestep in a trained DiT-MoE. Cite with Finding 1; our claim extends/quantifies it, it is NOT 'first'.",
    "ganjdanesh2024diffpruning": "Timestep-interval experts by fine-tuning pruned dense sub-networks. Contrast: ours is post-hoc, training-free, MoE experts.",
    "fang2024remixdit": "Timestep multi-expert denoising via learned mixing (trained). Related work.",
    "yang2025alter": "Temporal expert routing plus layer pruning with a trained hypernetwork. Related work.",
    "guo2025mosaicdiff": "Training-free timestep-staged structural pruning of dense DiT/SDXL. Closest training-free timestep-aware pruning; contrast: dense, not experts.",
    "liu2026diffes": "Stage-wise pruning by evolutionary search; notes the memory cost of stitching stage-wise models (same issue as our union). Related work.",
    "li2026moeca": "Training-free expert-branch caching across timesteps in DiT-MoE: closest post-hoc DiT-MoE/timestep-axis work with a real speedup. Must be discussed.",
    "park2024switchdit": "Trained timestep-aware MoE.", "shi2025diffmoe": "Trained timestep/noise-dependent capacity.",
    "yuan2025racedit": "Trained flexible routing for DiT-MoE.", "wei2025promoe": "Trained routing guidance for DiT-MoE.",
    "liu2025taylorseer": "Training-free feature forecasting; context for the saturation of caching-based acceleration.",
    "zhang2026ecdlm": "Timestep-dependent expert capacity in diffusion LMs (trained); context for adaptive per-step computation.",
    "wei2026team": "Temporal-consistency expert activation for MoE diffusion LMs.",
    "liu2025dsmoe": "Backbone family (DSMoE-S-E48, DSMoE-B-E48): architecture, checkpoints, reported FID.",
    "lasby2025reap": "REAP criterion (gate x activation norm): our 'reap' baseline.",
    "dong2025easyep": "EASY-EP: output-aware importance = gate x L2 norm; our full score is its frequency-weighted analogue.",
    "zhou2025dern": "Router-statistics expert pruning; LLM, no timestep axis.",
    "liu2026scoreexperts": "Unified freq x gate x activation formulation for one-shot MoE pruning (LLMs). Closest to our criteria ablation; position against it; consider adding its MAN criterion (mean activation norm, gate-free) as a variant.",
    "lu2024notallexperts": "Early post-training expert pruning/skipping for MoE LLMs.",
    "xie2024moepruner": "One-shot MoE-LLM pruning with router hints.", "liu2024eep": "Gradient-free expert pruning (EEP).",
    "jha2026ream": "Expert merging vs pruning (optional context).",
    "eliseev2023offloading": "MoE LM offloading to CPU RAM; prior art for the per-band offload framing.",
    "liang2026localrouting": "Local routing consistency and expert offloading; prior art for offload feasibility.",
    "russakovsky2015imagenet": "ImageNet / ILSVRC dataset citation.", "peebles2023dit": "DiT.",
    "liu2023rectifiedflow": "Rectified flow (training objective of the backbone).",
    "rombach2022ldm": "Latent diffusion; SD-VAE used for decoding.", "ho2022cfg": "Classifier-free guidance (CFG 1.5).",
    "dhariwal2021adm": "ADM; source of the ImageNet-256 FID reference statistics.", "heusel2017fid": "FID.",
    "dai2024deepseekmoe": "DeepSeek-style shared-expert MoE (architecture used by DSMoE).",
    "shazeer2017moe": "Sparsely-gated MoE (background)."}

bib = ["% GENERATED by tools/build_refs.py from arXiv API metadata (references_arxiv.json). Edit the script, not this file.\n"]
rows = []
for aid, key in KEY.items():
    r = d[aid]
    kind, extra = VENUE.get(aid, ("misc", {}))
    fields = [("author", " and ".join(r["authors"])), ("title", "{" + r["title"] + "}"), ("year", extra.get("year", r["year"]))]
    if kind == "inproceedings":
        fields.append(("booktitle", extra["booktitle"]))
        if "pages" in extra:
            fields.append(("pages", extra["pages"]))
    else:
        fields.append(("note", extra.get("note", "arXiv preprint")))
    fields += [("eprint", aid), ("archivePrefix", "arXiv"), ("url", "https://arxiv.org/abs/" + aid)]
    body = ",\n".join("  %s = {%s}" % (k, v) for k, v in fields)
    bib.append("@%s{%s,\n%s\n}\n" % (kind, key, body))
    au = r["authors"]
    short = au[0].split()[-1] + " et al." if len(au) > 2 else " & ".join(a.split()[-1] for a in au)
    rows.append((key, aid, short, r["year"], VERIFIED.get(aid, "arXiv only; venue NOT verified"), ROLE.get(key, "")))

bib.append("""@inproceedings{cheng2025diffmoe,
  author = {Kun Cheng and Xiao He and Lei Yu and Zhijun Tu and Mingrui Zhu and Nannan Wang and Xinbo Gao and Jie Hu},
  title = {{Diff-MoE}: Diffusion Transformer with Time-Aware and Space-Adaptive Experts},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {267},
  pages = {10010--10024},
  year = {2025},
  url = {https://proceedings.mlr.press/v267/cheng25d.html}
}
""")
rows.append(("cheng2025diffmoe", "(PMLR 267)", "Cheng et al.", "2025", "verified (ICML 2025, pp. 10010-10024)",
             "Trained time-aware, space-adaptive DiT-MoE (expert-specific timestep conditioning). Related work."))
bib.append("""@misc{seitzer2020pytorchfid,
  author = {Maximilian Seitzer},
  title = {{pytorch-fid}: {FID} Score for {PyTorch}},
  year = {2020},
  note = {Version 0.3.0},
  url = {https://github.com/mseitzer/pytorch-fid}
}
""")
rows.append(("seitzer2020pytorchfid", "(software)", "Seitzer", "2020", "software; v0.3.0 used (check repo URL/year before submission)",
             "FID implementation used for all FID values."))
open("references.bib", "w", encoding="utf-8").write("\n".join(bib))

md = ["# Citation verification log (generated by tools/build_refs.py)\n",
      "Title, authors, year and arXiv ID come from the arXiv API (queried 2026-10-04) and are authoritative. "
      "'Venue' is filled only where the venue page or listing was seen in-session; every other entry is cited as an arXiv preprint "
      "and **its published venue must be checked before submission**. Classics (DiT/ICCV 2023, LDM/CVPR 2022, ADM/NeurIPS 2021, "
      "FID/NeurIPS 2017, rectified flow/ICLR 2023, DeepSeekMoE/ACL 2024) are believed to be published but were not re-verified.\n",
      "**Deliberately NOT cited (unverified):** 'DIET' (not found by any search), 'OLED-MoE' (not confirmed by name; an arXiv ID "
      "quoted by an AI reviewer was never verified), 'TimeAware MoE (IEEE)'. Do not add them without locating the source.\n",
      "| Key | arXiv | Authors | Year | Venue status | Role in our paper |", "|---|---|---|---|---|---|"]
md += ["| `%s` | %s | %s | %s | %s | %s |" % row for row in rows]
open("CITATIONS.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print(len(rows), "entries written")
