import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import argparse, json, random, datetime, re
from importlib.metadata import version
from datasets import load_dataset
from retrieval.bm25 import searcher, get_text, has_answer

p = argparse.ArgumentParser()
p.add_argument("--n_train", type=int, default=4000)
p.add_argument("--n_dev", type=int, default=500)
p.add_argument("--k", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--exclude", default="outputs/retrieval/nq_arag_test_bm25_top50.jsonl")
args = p.parse_args()

ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="dev")
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
excl = {norm(json.loads(l)["question"]) for l in open(args.exclude)}
pool = [i for i in range(len(ds)) if norm(ds[i]["question"]) not in excl]
print(f"NQ dev: {len(ds)}  excluded (in test set): {len(ds) - len(pool)}  pool: {len(pool)}")

random.Random(args.seed).shuffle(pool)
subsets = {"dtrain": sorted(pool[:args.n_train]),
           "ddev": sorted(pool[args.n_train:args.n_train + args.n_dev])}

for name, idx in subsets.items():
    sub = ds.select(idx)
    results = searcher.batch_search(sub["question"], sub["id"], k=args.k, threads=8)
    out = f"outputs/retrieval/nq_{name}_bm25_top{args.k}.jsonl"
    with open(out, "w") as f:
        for ex in sub:
            docids = [h.docid for h in results[ex["id"]]]
            f.write(json.dumps({
                "qid": ex["id"], "question": ex["question"], "gold": ex["golden_answers"],
                "docids": docids,
                "scores": [round(h.score, 4) for h in results[ex["id"]]],
                "has_answer": [has_answer(get_text(d), ex["golden_answers"]) for d in docids],
            }, ensure_ascii=False) + "\n")
    json.dump({"index": "wikipedia-dpr-100w", "pyserini": version("pyserini"),
               "dataset": "nq", "source_split": "dev (FlashRAG), excluding Adaptive-RAG test",
               "split": name, "n": len(idx), "k": args.k, "seed": args.seed,
               "has_answer": "word-boundary-v2", "created": datetime.datetime.now().isoformat()},
              open(out.replace(".jsonl", ".meta.json"), "w"), indent=2)
    rows = [json.loads(l) for l in open(out)]
    hit5 = sum(any(r["has_answer"][:5]) for r in rows) / len(rows)
    gold = sum(len(r["gold"]) for r in rows) / len(rows)
    print(f"{name}: {len(rows)} questions  Hit@5 {hit5:.1%}  avg gold {gold:.2f}  -> {out}")