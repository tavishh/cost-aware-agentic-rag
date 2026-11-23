"""Add has_answer using the SAME get_text / has_answer as the BM25 cache.
Run from project root:  python -m retrieval.e5_add_has_answer --inp ... --out ..."""
import argparse, json
from retrieval.bm25 import get_text, has_answer   # <- adjust if your bm25.py signatures differ

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.inp)]
with open(a.out, "w") as f:
    for r in rows:
        r["has_answer"] = [bool(has_answer(get_text(d), r["gold"])) for d in r["docids"]]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"retriever": "intfloat/e5-base-v2", "query_prefix": "query: ",
           "passage_format": "passage: {title} {text}", "pooling": "mean+l2",
           "search": "exact inner product, fp16", "k": len(rows[0]["docids"]),
           "corpus": "wikipedia-dpr-100w (docids from local Lucene index)", "n": len(rows)},
          open(a.out.replace(".jsonl", ".meta.json"), "w"), indent=2)
print("wrote", a.out, len(rows))
