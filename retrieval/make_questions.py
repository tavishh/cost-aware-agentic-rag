"""Extract {qid, question, gold} in original order from a tree file (root nodes)
or from a retrieval cache. Run on Mac from project root:
  python -m retrieval.make_questions --inp <tree or cache jsonl> --out outputs/retrieval/nq_<split>_questions.jsonl
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

seen, out = set(), []
with open(a.inp) as f:
    for line in f:
        r = json.loads(line)
        if r.get("step", 0) != 0 or r["qid"] in seen:
            continue
        seen.add(r["qid"])
        out.append({"qid": r["qid"], "question": r["question"], "gold": r["gold"]})
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {a.out}  ({len(out)} questions)")
