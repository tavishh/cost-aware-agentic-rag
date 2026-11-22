"""Contamination check, step 1 (Mac, ragcache env, from project root):
DPR NQ test (= original NQ dev, 3,610 questions; no e5 version could have trained on it)
-> questions file + BM25 top-50 cache with has_answer (same format/functions as our BM25 caches).

  python -m retrieval.make_dpr_nq_test
"""
import argparse, ast, json
from retrieval.bm25 import searcher, get_text, has_answer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="0 = all questions")
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--stem", default="outputs/retrieval/nq_dprtest")
ap.add_argument("--topics_file", default="",
                help="local topics.dpr.nq.test.txt (question<TAB>answers per line); skips download")
ap.add_argument("--dpr_json", default="", help="DPR biencoder json(.gz), e.g. biencoder-nq-dev.json.gz")
ap.add_argument("--exclude", default="", help="jsonl whose questions are dropped (e.g. our 500 test set)")
ap.add_argument("--qid_prefix", default="dprtest")
a = ap.parse_args()

if a.dpr_json:
    import gzip
    op = gzip.open if a.dpr_json.endswith(".gz") else open
    with op(a.dpr_json, "rt", encoding="utf-8") as f:
        data = json.load(f)
    topics = {str(i): {"title": d["question"], "answers": d["answers"]} for i, d in enumerate(data)}
    del data
elif a.topics_file:
    topics = {}
    for i, line in enumerate(open(a.topics_file, encoding="utf-8")):
        line = line.rstrip("\n")
        if not line:
            continue
        q, _, ans = line.partition("\t")
        topics[str(i)] = {"title": q, "answers": ans}
else:
    from pyserini.search import get_topics
    topics = get_topics("dpr-nq-test")
print("topics:", len(topics), "| sample:", next(iter(topics.items())))

def parse_answers(x):
    if isinstance(x, (list, tuple)):
        return [str(s) for s in x]
    x = str(x).strip()
    if x.startswith("["):
        return [str(s) for s in ast.literal_eval(x)]
    return [x]

rows = [{"qid": f"{a.qid_prefix}_{qid}", "question": t["title"], "gold": parse_answers(t["answers"])}
        for qid, t in topics.items()]
if a.exclude:
    drop = {json.loads(l)["question"].strip().lower() for l in open(a.exclude)}
    before = len(rows)
    rows = [r for r in rows if r["question"].strip().lower() not in drop]
    print(f"excluded {before - len(rows)} questions found in {a.exclude}")
if a.n:
    rows = rows[:a.n]

with open(f"{a.stem}_questions.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

ids = [r["qid"] for r in rows]
res = searcher.batch_search([r["question"] for r in rows], ids, k=a.k, threads=8)
with open(f"{a.stem}_bm25_top{a.k}.jsonl", "w") as f:
    for r in rows:
        hits = res[r["qid"]]
        docids = [h.docid for h in hits]
        f.write(json.dumps({**r, "docids": docids, "scores": [round(h.score, 4) for h in hits],
                            "has_answer": [has_answer(get_text(d), r["gold"]) for d in docids]},
                           ensure_ascii=False) + "\n")
print(f"wrote {a.stem}_questions.jsonl and {a.stem}_bm25_top{a.k}.jsonl ({len(rows)} questions)")
