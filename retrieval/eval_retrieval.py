import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, string
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")

def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def has_answer(text, answers):
    t = normalize(text)
    return any(normalize(a) in t for a in answers)

rows = [json.loads(l) for l in open("outputs/retrieval/nq_bm25_top50.jsonl")]
ks = [1, 5, 10, 20, 50]
hits = {k: 0 for k in ks}
for r in rows:
    flags = [has_answer(json.loads(searcher.doc(d).raw())["contents"], r["gold"]) for d in r["docids"]]
    for k in ks:
        hits[k] += any(flags[:k])
for k in ks:
    print(f"Hit@{k:<3} {hits[k] / len(rows):.0%}")