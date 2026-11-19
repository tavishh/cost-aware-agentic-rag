import os, json, re, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

M = "Llama-3.1-8B-Instruct_accumulate"
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
dpr = set(json.load(open("outputs/retrieval/dpr_nq_dev_questions.json")))
test_q = {norm(json.loads(l)["question"]) for l in open("outputs/retrieval/nq_arag_test_bm25_top50.jsonl")}

def read(fn):
    return [json.loads(l) for l in open(fn)]

final = {}
for base, top in [("dtrain", "dtraintop"), ("ddev", "ddevtop")]:
    old = [n for n in read(f"outputs/tree/nq_{base}_tree15_{M}.jsonl") if norm(n["question"]) in dpr]
    new = read(f"outputs/tree/nq_{top}_tree15_{M}.jsonl")
    nodes = old + new
    qs = {n["qid"] for n in nodes}
    overlap = sum(norm(n["question"]) in test_q for n in nodes if n["parent_id"] is None)
    out = f"outputs/tree/nq_{base}_dpr_tree15_{M}.jsonl"
    with open(out, "w") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    roots = [n for n in nodes if n["parent_id"] is None]
    rets = [n for n in nodes if n["node_id"].endswith("/R") and n["step"] == 1]
    print(f"{base}_dpr: {len(qs)} questions, {len(nodes)} nodes, overlap with test {overlap}, "
          f"root InAcc {sum(n['metrics']['inacc'] for n in roots)/len(roots):.3f}, "
          f"R InAcc {sum(n['metrics']['inacc'] for n in rets)/len(rets):.3f} -> {out}")
    final[base] = nodes

final["arag_test"] = read(f"outputs/tree/nq_arag_test_tree15_{M}.jsonl")

# 汇总所有最终树用到的段落
need = {d for nodes in final.values() for n in nodes for rt in n["retrievals"] for d in rt["docids"]}
files = ["outputs/retrieval/passages.jsonl"] + glob.glob("outputs/tree_work/*Llama*/passages_L*.jsonl")
passages = {}
for fn in files:
    for l in open(fn):
        x = json.loads(l)
        if x["docid"] in need:
            passages[x["docid"]] = x
out = "outputs/tree/passages_final.jsonl"
with open(out, "w") as f:
    for d in sorted(passages, key=int):
        f.write(json.dumps(passages[d], ensure_ascii=False) + "\n")
print(f"passages needed: {len(need)}, found: {len(passages)} -> {out}")
