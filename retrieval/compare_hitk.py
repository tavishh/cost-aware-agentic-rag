"""Hit@k for BM25, e5, and RRF fusion (k_rrf=60).
Run from project root:  python -m retrieval.compare_hitk --bm25 ... --e5 ..."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--bm25", required=True)
ap.add_argument("--e5", required=True)
ap.add_argument("--rrf_k", type=int, default=60)
ap.add_argument("--rrf_out", default=None)
a = ap.parse_args()

load = lambda p: {r["qid"]: r for r in map(json.loads, open(p))}
B, E = load(a.bm25), load(a.e5)
assert B.keys() == E.keys(), "qid mismatch"
KS = [1, 5, 10, 20, 50]


def rrf(b, e):
    sc, ha = {}, {}
    for r in (b, e):
        for rank, (d, h) in enumerate(zip(r["docids"], r["has_answer"])):
            sc[d] = sc.get(d, 0) + 1 / (a.rrf_k + rank + 1)
            ha[d] = h
    top = sorted(sc, key=sc.get, reverse=True)[:50]
    return {"qid": b["qid"], "question": b["question"], "gold": b["gold"], "docids": top,
            "scores": [sc[d] for d in top], "has_answer": [ha[d] for d in top]}


R = {q: rrf(B[q], E[q]) for q in B}
for name, D in [("BM25", B), ("e5", E), ("RRF", R)]:
    print(f"{name:5s}", "  ".join(
        f"@{k}={100 * sum(any(r['has_answer'][:k]) for r in D.values()) / len(D):.1f}" for k in KS))

only_b = sum(any(B[q]["has_answer"][:5]) and not any(E[q]["has_answer"][:5]) for q in B)
only_e = sum(any(E[q]["has_answer"][:5]) and not any(B[q]["has_answer"][:5]) for q in B)
print(f"@5 only-BM25={only_b}  only-e5={only_e}")

if a.rrf_out:
    with open(a.rrf_out, "w") as f:
        for r in R.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", a.rrf_out)
