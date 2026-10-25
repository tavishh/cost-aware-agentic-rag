import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import argparse, json, random
from datasets import load_dataset
from retrieval.bm25 import searcher, get_text, has_answer

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="nq")
p.add_argument("--split", default="test")
p.add_argument("--n", type=int, default=500)
p.add_argument("--k", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", args.dataset, split=args.split)
random.seed(args.seed)
ds = ds.select(sorted(random.sample(range(len(ds)), min(args.n, len(ds)))))

results = searcher.batch_search(ds["question"], ds["id"], k=args.k, threads=8)

os.makedirs("outputs/retrieval", exist_ok=True)
out = f"outputs/retrieval/{args.dataset}_{args.split}_bm25_top{args.k}.jsonl"
with open(out, "w") as f:
    for ex in ds:
        hits = results[ex["id"]]
        docids = [h.docid for h in hits]
        f.write(json.dumps({
            "qid": ex["id"],
            "question": ex["question"],
            "gold": ex["golden_answers"],
            "docids": docids,
            "scores": [round(h.score, 4) for h in hits],
            "has_answer": [has_answer(get_text(d), ex["golden_answers"]) for d in docids],
        }, ensure_ascii=False) + "\n")
print("saved:", out, "questions:", len(ds))