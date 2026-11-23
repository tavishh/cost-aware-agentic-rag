"""Collect the passage texts referenced by a set of tree files into one passages file
(same format as passages_final.jsonl: {"docid", "title", "text"}).

Sources: outputs/retrieval/passages.jsonl + every outputs/tree_work/*/passages_L*.jsonl
(these hold every passage the tree pipeline ever showed the model). Any docid still missing
is reported; --fill_from_index fetches it from the Lucene index given by BM25_INDEX
(DPR prebuilt index if unset).

Run on the cluster from the project root:
  python cache/build_passages_final.py --out outputs/tree/passages_final_e5.jsonl \
      outputs/tree/nq_*_e5_tree15_*accumulate.jsonl
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import argparse, glob, json

ap = argparse.ArgumentParser()
ap.add_argument("trees", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--fill_from_index", action="store_true")
a = ap.parse_args()

need = set()
for fn in a.trees:
    n_nodes = 0
    for line in open(fn):
        n = json.loads(line)
        n_nodes += 1
        for rt in n["retrievals"]:
            need.update(rt["docids"])
        need.update(n.get("context_docids", []))
    print(f"{os.path.basename(fn)}: {n_nodes} nodes")
print(f"unique docids needed: {len(need)}")

found = {}
sources = ["outputs/retrieval/passages.jsonl"] + sorted(glob.glob("outputs/tree_work/*/passages_L*.jsonl"))
for src in sources:
    if not os.path.exists(src):
        continue
    for line in open(src):
        x = json.loads(line)
        d = x["docid"]
        if d in need and d not in found:
            found[d] = {"docid": d, "title": x["title"], "text": x["text"]}

missing = sorted(need - found.keys())
print(f"found in passage stores: {len(found)}   missing: {len(missing)}")
if missing and a.fill_from_index:
    from retrieval.bm25 import get_text
    for d in missing:
        title, _, text = get_text(d).replace('""', '"').partition("\n")
        found[d] = {"docid": d, "title": title.strip('"'), "text": text}
    print(f"filled {len(missing)} from index")
    missing = []
if missing:
    print("first missing:", missing[:5])
    sys.exit("missing passages: rerun with --fill_from_index (and the right BM25_INDEX)")

with open(a.out, "w", encoding="utf-8") as f:
    for d in sorted(found):
        f.write(json.dumps(found[d], ensure_ascii=False) + "\n")
print(f"wrote {a.out} ({len(found)} passages)")
