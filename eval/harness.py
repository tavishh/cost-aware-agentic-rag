import json
from collections import defaultdict

D = "outputs/tree/"
M = "Llama-3.1-8B-Instruct_accumulate.jsonl"
FILES = {
    "bm25": {"dev": D + "nq_ddev_dpr_tree15_" + M, "test": D + "nq_arag_test_tree15_" + M},
    "e5":   {"dev": D + "nq_ddev_dpr_e5_tree15_" + M, "test": D + "nq_arag_test_e5_tree15_" + M},
}

def load(fn):
    by_path = defaultdict(dict)          # path -> {qid: node}
    for line in open(fn):
        n = json.loads(line)
        path = n["node_id"][len(n["qid"]):].lstrip("/") or "root"
        by_path[path][n["qid"]] = n
    return by_path

def tokens(n):
    c = n["cum_cost"]
    return c["tokens_in"] + c["tokens_out"]

def score(bp, path):
    nodes = list(bp[path].values())
    acc = sum(n["metrics"]["inacc"] for n in nodes) / len(nodes)
    tok = sum(tokens(n) for n in nodes) / len(nodes)
    return acc, tok, len(nodes)

def oracle(bp, paths):
    qids = list(bp["root"].keys())
    total = sum(max(bp[p][q]["metrics"]["inacc"] for p in paths if q in bp[p]) for q in qids)
    return total / len(qids)

def show(label, acc, tok, n):
    print(f"  {label:34} InAcc={acc:.3f}  tokens={tok:6.0f}  n={n}")

for name, f in FILES.items():
    dev, test = load(f["dev"]), load(f["test"])
    print(f"\n=== retriever: {name} ===")
    for split, bp in (("dev", dev), ("test", test)):
        print(f"[{split}]")
        show("never retrieve (root)", *score(bp, "root"))
        show("retrieve once (R)", *score(bp, "R"))
        show("retrieve x3 (R/R/R)", *score(bp, "R/R/R"))

    depth3 = sorted(p for p in dev if p.count("/") == 2)
    best = max(depth3, key=lambda p: score(dev, p)[0])     # chosen on DEV only
    print(f"[test] best 3-step path chosen on dev: {best}")
    show(f"  {best} on test", *score(test, best))

    all_paths = list(test.keys())
    ret_only = ["root", "R", "R/R", "R/R/R"]
    print(f"[test] oracle, all paths:    {oracle(test, all_paths):.3f}")
    print(f"[test] oracle, retrieve only: {oracle(test, ret_only):.3f}")