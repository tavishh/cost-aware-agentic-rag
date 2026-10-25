import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, os, random
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher

DATASET, N, K = "nq", 100, 50

ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", DATASET, split="test")
#seed for reproducibility
random.seed(42)
ds = ds.select(sorted(random.sample(range(len(ds)), N)))

searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")
results = searcher.batch_search(ds["question"], ds["id"], k=K, threads=8) #threads speed up retrieval

os.makedirs("outputs/retrieval", exist_ok=True)
out = f"outputs/retrieval/{DATASET}_bm25_top{K}.jsonl"
with open(out, "w") as f:
    for ex in ds:
        hits = results[ex["id"]]
        f.write(json.dumps({
            "qid": ex["id"],
            "question": ex["question"],
            "gold": ex["golden_answers"],
            "docids": [h.docid for h in hits],
            "scores": [round(h.score, 4) for h in hits],
        }, ensure_ascii=False) + "\n")
print("saved:", out)
