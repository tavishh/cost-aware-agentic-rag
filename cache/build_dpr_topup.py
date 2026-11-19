import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import json, random, re, datetime
from importlib.metadata import version
from datasets import load_dataset
from retrieval.bm25 import searcher, get_text, has_answer

norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
dpr = set(json.load(open("outputs/retrieval/dpr_nq_dev_questions.json")))

def questions(name):
    return [norm(json.loads(l)["question"]) for l in open(f"outputs/retrieval/{name}_bm25_top50.jsonl")]

used = set(questions("nq_arag_test")) | set(questions("nq_dtrain")) | set(questions("nq_ddev"))
need = {"dtraintop": 4000 - sum(q in dpr for q in questions("nq_dtrain")),
        "ddevtop": 500 - sum(q in dpr for q in questions("nq_ddev"))}

ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="dev")
seen, pool = set(), []
for i in range(len(ds)):
    q = norm(ds[i]["question"])
    if q in dpr and q not in used and q not in seen:
        pool.append(i); seen.add(q)
print("need:", need, " available pool:", len(pool))

random.Random(7).shuffle(pool)
start = 0
for name, k in need.items():
    idx = sorted(pool[start:start + k]); start += k
    sub = ds.select(idx)
    results = searcher.batch_search(sub["question"], sub["id"], k=50, threads=8)
    out = f"outputs/retrieval/nq_{name}_bm25_top50.jsonl"
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
               "source_split": "FlashRAG NQ dev ∩ DPR NQ dev, excluding test and existing dtrain/ddev",
               "split": name, "n": len(idx), "seed": 7, "k": 50,
               "created": datetime.datetime.now().isoformat()},
              open(out.replace(".jsonl", ".meta.json"), "w"), indent=2)
    rows = [json.loads(l) for l in open(out)]
    print(f"{name}: {len(rows)} questions  Hit@5 {sum(any(r['has_answer'][:5]) for r in rows)/len(rows):.1%}  -> {out}")
