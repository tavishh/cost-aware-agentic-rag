import json
from collections import defaultdict

base = "outputs/tree/nq_arag_test_tree15_Llama-3.1-8B-Instruct_accumulate"
runs = {"current (stop, 32 tok)": base + ".jsonl",
        "simple, no stop": base + "_nostop_only-root-R.jsonl",
        "fewshot, no stop": base + "_fewshot_nostop_only-root-R.jsonl"}

print(f"{'setting':24s} {'path':5s} {'EM':>6} {'F1':>6} {'InAcc':>6} {'ans_tok':>7}")
for name, fn in runs.items():
    g = defaultdict(list)
    for l in open(fn):
        n = json.loads(l)
        p = n["node_id"][len(n["qid"]):].lstrip("/") or "root"
        if p in ("root", "R"):
            g[p].append(n)
    for p in ("root", "R"):
        xs = g[p]
        m = len(xs)
        avg = lambda k: sum(x["metrics"][k] for x in xs) / m
        print(f"{name:24s} {p:5s} {avg('em'):6.3f} {avg('f1'):6.3f} {avg('inacc'):6.3f} "
              f"{sum(x['answer_call']['tokens_out'] for x in xs)/m:7.1f}")