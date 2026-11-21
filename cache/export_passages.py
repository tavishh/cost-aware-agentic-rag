import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from retrieval.bm25 import get_text

ids = set()
for f in glob.glob("outputs/retrieval/*_top*.jsonl"):
    for l in open(f):
        ids.update(json.loads(l)["docids"])

with open("outputs/retrieval/passages.jsonl", "w") as f:
    for d in sorted(ids, key=int):
        raw = get_text(d)
        raw = raw.replace('""', '"')
        title, _, text = raw.partition("\n")
        f.write(json.dumps({"docid": d, "title": title.strip('"'),
                            "text": text}, ensure_ascii=False) + "\n")
print("passages:", len(ids))