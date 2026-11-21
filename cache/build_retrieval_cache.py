import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import argparse, json, random, datetime
from importlib.metadata import version
from datasets import load_dataset
from retrieval.bm25 import searcher, get_text, has_answer

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="nq")
p.add_argument("--split", default="test")
p.add_argument("--n", type=int, default=500)
p.add_argument("--k", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dev_n", type=int, default=0)
args = p.parse_args()

ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", args.dataset, split=args.split)
rng = random.Random(args.seed)
ds = ds.select(sorted(rng.sample(range(len(ds)), min(args.n, len(ds)))))

order = list(range(len(ds)))
if args.dev_n > 0:
    random.Random(args.seed + 1).shuffle(order)
    subsets = {"dev": sorted(order[:args.dev_n]),
               args.split: sorted(order[args.dev_n:])}
else:
    subsets = {args.split: order}

results = searcher.batch_search(ds["question"], ds["id"], k=args.k, threads=8)

os.makedirs("outputs/retrieval", exist_ok=True)
for name, rows in subsets.items():
    out = f"outputs/retrieval/{args.dataset}_{name}_bm25_top{args.k}.jsonl"
    with open(out, "w") as f:
        for i in rows:
            ex = ds[i]
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

    meta = {
        "index": "wikipedia-dpr-100w",
        "pyserini": version("pyserini"),
        "dataset": args.dataset,
        "split": name,
        "source_split": args.split,
        "n": len(rows),
        "k": args.k,
        "seed": args.seed,
        "has_answer": "word-boundary-v2",
        "created": datetime.datetime.now().isoformat(),
    }
    with open(out.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved:", out, "questions:", len(rows))