"""BM25 top-K retrieval cache for any corpus (set BM25_INDEX), same format as the NQ caches,
plus passages_L0 for the tree work dir (root-question passages, ranks 1..depth).
Also reports supporting-paragraph recall when questions carry support_titles.
  BM25_INDEX=/path/lucene-musique python cache/build_bm25_cache.py \
      --questions outputs/retrieval/musique_test_questions.jsonl \
      --cache outputs/retrieval/musique_test_bm25_top50.jsonl --n 500
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cache"))

import argparse, glob, json
from tree_common import load_jsonl, write_jsonl, work_dir
from retrieval.bm25 import searcher, get_text, has_answer

p = argparse.ArgumentParser()
p.add_argument("--questions", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--n", type=int, required=True)
p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
p.add_argument("--k", type=int, default=50)
p.add_argument("--depth", type=int, default=15)
a = p.parse_args()
assert "_bm25" in os.path.basename(a.cache), "cache name must contain _bm25 (work_dir naming)"

rows = load_jsonl(a.questions)
ids = [str(i) for i in range(len(rows))]
res = searcher.batch_search([r["question"] for r in rows], ids, k=a.k, threads=8)

wd = work_dir(a.cache, a.n, a.model)
known = set()
for fn in ["outputs/retrieval/passages.jsonl"] + glob.glob(f"{wd}/passages_L*.jsonl"):
    if not fn.endswith("passages_L0.jsonl") and os.path.exists(fn):
        known.update(json.loads(l)["docid"] for l in open(fn))

out, new_p, sup = [], {}, {k: [] for k in (5, 15, 50)}
for i, r in enumerate(rows):
    hits = res[ids[i]]
    docids = [h.docid for h in hits]
    texts = {d: get_text(d) for d in docids}
    out.append({"qid": r["qid"], "question": r["question"], "gold": r["gold"],
                "docids": docids, "scores": [round(h.score, 4) for h in hits],
                "has_answer": [has_answer(texts[d], r["gold"]) for d in docids]})
    if r.get("support_titles"):
        titles = [texts[d].replace('""', '"').partition("\n")[0].strip('"') for d in docids]
        st = set(r["support_titles"])
        for k in sup:
            sup[k].append(len(st & set(titles[:k])) / len(st))
    if i < a.n:
        for d in docids[:a.depth]:
            if d not in known and d not in new_p:
                title, _, text = texts[d].replace('""', '"').partition("\n")
                new_p[d] = {"docid": d, "title": title.strip('"'), "text": text}

write_jsonl(a.cache, out)
write_jsonl(f"{wd}/passages_L0.jsonl", list(new_p.values()))
json.dump({"retriever": "bm25 (pyserini default)", "index": os.environ.get("BM25_INDEX", "wikipedia-dpr-100w"),
           "questions": a.questions, "k": a.k, "n": len(out)},
          open(a.cache.replace(".jsonl", ".meta.json"), "w"), indent=2)
hit = "  ".join(f"@{k}={100*sum(any(x['has_answer'][:k]) for x in out)/len(out):.1f}" for k in (1, 5, 10, 20, 50))
print(f"wrote {a.cache} ({len(out)})  answer-Hit {hit}")
if sup[5]:
    print("support recall  " + "  ".join(f"@{k}={100*sum(v)/len(v):.1f}" for k, v in sup.items()))
print(f"passages_L0: {len(new_p)} -> {wd}/passages_L0.jsonl")
