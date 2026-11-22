"""Contamination check, final step (Mac): compare e5's margin over BM25 on two question sets.
  python -m retrieval.contamination_check \
    --set "ours (DPR NQ dev)" outputs/retrieval/nq_arag_test_bm25_top50.jsonl outputs/retrieval/nq_arag_test_e5_top50.jsonl \
    --set "DPR NQ test" outputs/retrieval/nq_dprtest_bm25_top50.jsonl outputs/retrieval/nq_dprtest_e5_top50.jsonl
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs=3, action="append", metavar=("NAME", "BM25", "E5"), required=True)
a = ap.parse_args()
KS = [1, 5, 10, 20, 50]

def hits(path):
    rows = [json.loads(l) for l in open(path)]
    return {k: 100 * sum(any(r["has_answer"][:k]) for r in rows) / len(rows) for k in KS}, len(rows)

print(f"{'set':22s} {'n':>5} {'k':>3} {'BM25':>6} {'e5':>6} {'margin':>7} {'miss-rate cut':>13}")
for name, pb, pe in a.set:
    hb, nb = hits(pb)
    he, ne = hits(pe)
    assert nb == ne, f"{name}: size mismatch {nb} vs {ne}"
    for k in KS:
        cut = 100 * (he[k] - hb[k]) / (100 - hb[k]) if hb[k] < 100 else float("nan")
        print(f"{name:22s} {nb:>5} {k:>3} {hb[k]:>6.1f} {he[k]:>6.1f} {he[k]-hb[k]:>+7.1f} {cut:>12.0f}%")
    print()
print("margin = e5 - BM25 (points); miss-rate cut = share of BM25 misses that e5 recovers")
