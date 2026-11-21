import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import argparse, json, datetime
from importlib.metadata import version
from retrieval.bm25 import searcher, get_text, has_answer

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)   # Adaptive-RAG 的 test_subsampled.jsonl
p.add_argument("--k", type=int, default=50)
p.add_argument("--name", default="nq_arag_test")
args = p.parse_args()

rows = []
for l in open(args.src):
    x = json.loads(l)
    gold = list(dict.fromkeys(s for a in x["answers_objects"] for s in a["spans"] if s))
    rows.append({"qid": x["question_id"], "question": x["question_text"], "gold": gold})
print("questions:", len(rows))

results = searcher.batch_search([r["question"] for r in rows],
                                [r["qid"] for r in rows], k=args.k, threads=8)

out = f"outputs/retrieval/{args.name}_bm25_top{args.k}.jsonl"
with open(out, "w") as f:
    for r in rows:
        hits = results[r["qid"]]
        docids = [h.docid for h in hits]
        f.write(json.dumps({**r, "docids": docids,
                            "scores": [round(h.score, 4) for h in hits],
                            "has_answer": [has_answer(get_text(d), r["gold"]) for d in docids]},
                           ensure_ascii=False) + "\n")

json.dump({"index": "wikipedia-dpr-100w", "pyserini": version("pyserini"),
           "source": os.path.abspath(args.src),
           "source_desc": "Adaptive-RAG processed_data/nq/test_subsampled.jsonl",
           "n": len(rows), "k": args.k, "has_answer": "word-boundary-v2",
           "created": datetime.datetime.now().isoformat()},
          open(out.replace(".jsonl", ".meta.json"), "w"), indent=2)

# 检查和我们的 train/dev 是否有重复题目
seen = set()
for fn in ["outputs/retrieval/nq_train_bm25_top50.jsonl", "outputs/retrieval/nq_dev_bm25_top50.jsonl"]:
    seen |= {json.loads(l)["question"].strip().lower() for l in open(fn)}
dup = sum(r["question"].strip().lower() in seen for r in rows)
print("overlap with our train/dev:", dup)

for k in [1, 5, 10, 20, 50]:
    hit = sum(any(json.loads(l)["has_answer"][:k]) for l in open(out)) / len(rows)
    print(f"Hit@{k:<3} {hit:.1%}")
print("saved:", out)

