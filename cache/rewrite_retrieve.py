import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cache"))

import argparse, glob, json
from tree_common import load_jsonl, write_jsonl, work_dir
from retrieval.bm25 import searcher, get_text, has_answer

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
p.add_argument("--cache", default="outputs/retrieval/nq_test_bm25_top50.jsonl")
p.add_argument("--n", type=int, default=500)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--depth", type=int, default=15)   # 3 步 × 5 条
args = p.parse_args()

wd = work_dir(args.cache, args.n, args.model)
rws = [x for x in load_jsonl(f"{wd}/rewrites_L{args.layer}.jsonl") if x["new_query"]]

uniq = {}
for x in rws:
    uniq.setdefault((x["qid"], x["new_query"]), x["gold"])
pairs = list(uniq)
ids = [f"r{i}" for i in range(len(pairs))]
res = searcher.batch_search([q for _, q in pairs], ids, k=args.depth, threads=8)

known = set()
for fn in ["outputs/retrieval/passages.jsonl"] + glob.glob(f"{wd}/passages_L*.jsonl"):
    if not fn.endswith(f"passages_L{args.layer}.jsonl"):
        known.update(json.loads(l)["docid"] for l in open(fn))

rankings, new_p = [], {}
for i, (qid, q) in enumerate(pairs):
    hits = res[ids[i]]
    docids = [h.docid for h in hits]
    texts = {d: get_text(d) for d in docids}
    rankings.append({"qid": qid, "query": q, "docids": docids,
                     "scores": [round(h.score, 4) for h in hits],
                     "has_answer": [has_answer(texts[d], uniq[(qid, q)]) for d in docids]})
    for d in docids:
        if d not in known and d not in new_p:
            title, _, text = texts[d].replace('""', '"').partition("\n")
            new_p[d] = {"docid": d, "title": title.strip('"'), "text": text}

write_jsonl(f"{wd}/rankings_L{args.layer}.jsonl", rankings)
write_jsonl(f"{wd}/passages_L{args.layer}.jsonl", list(new_p.values()))
hit5 = sum(any(r["has_answer"][:5]) for r in rankings) / max(len(rankings), 1)
print(f"queries: {len(pairs)}  new passages: {len(new_p)}  Hit@5 of rewrites: {hit5:.1%}")