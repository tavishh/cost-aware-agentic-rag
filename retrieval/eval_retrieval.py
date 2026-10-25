import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/retrieval/nq_test_bm25_top50.jsonl"
rows = [json.loads(l) for l in open(path)]
ks = [1, 5, 10, 15, 20, 50]
print(path, "questions:", len(rows))
for k in ks:
    hit = sum(any(r["has_answer"][:k]) for r in rows) / len(rows)
    print(f"Hit@{k:<3} {hit:.1%}")