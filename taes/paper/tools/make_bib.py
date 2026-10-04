"""Builds references.bib + CITATIONS table data from the arXiv API (authoritative titles/authors/dates).
python tools/make_bib.py   -> references_arxiv.json ; references.bib is assembled from it plus VENUES below (only venues verified in-session)."""
import json, re, sys, urllib.request, xml.etree.ElementTree as ET
IDS = """2407.11633 2409.15557 2412.05628 2505.21817 2510.11962 2603.05105 2606.15615 2403.09176 2503.14487 2503.16057 2510.24711
2503.06923 2604.01622 2602.08404 2512.01252 2510.13999 2504.06792 2509.10377 2606.15716 2402.14800 2410.12013 2407.00945 2604.04356
2312.17238 1409.0575 2505.16056 2212.09748 2209.03003 2112.10752 2207.12598 2105.05233 1706.08500 2401.06066 1701.06538""".split()
url = "http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(IDS)
raw = urllib.request.urlopen(url, timeout=60).read()
ns = {"a": "http://www.w3.org/2005/Atom"}
out = {}
for e in ET.fromstring(raw).findall("a:entry", ns):
    aid = re.sub(r"v\d+$", "", e.find("a:id", ns).text.rsplit("/", 1)[-1])
    out[aid] = {"title": " ".join(e.find("a:title", ns).text.split()),
                "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                "year": e.find("a:published", ns).text[:4], "published": e.find("a:published", ns).text[:10]}
missing = [i for i in IDS if i not in out]
json.dump(out, open("references_arxiv.json", "w"), indent=1)
print(len(out), "fetched; missing:", missing)
for k, v in out.items(): print(k, v["year"], v["authors"][0], "|", v["title"][:70])
