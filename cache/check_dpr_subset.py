import os, sys, json, gzip, re
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())

# 读 DPR NQ dev，只保留问题文本（文件很大，读一次后存一份小的）
cache = "outputs/retrieval/dpr_nq_dev_questions.json"
if os.path.exists(cache):
    dpr = set(json.load(open(cache)))
else:
    src = os.path.expanduser("~/Downloads/dpr/biencoder-nq-dev.json.gz")
    data = json.load(gzip.open(src, "rt"))
    dpr = {norm(x["question"]) for x in data}
    json.dump(sorted(dpr), open(cache, "w"))
    del data
print("DPR NQ dev questions:", len(dpr))

# 测试集应该全部在 DPR 里
test = [json.loads(l) for l in open("outputs/retrieval/nq_arag_test_bm25_top50.jsonl")]
print("Adaptive-RAG test in DPR:", sum(norm(r["question"]) in dpr for r in test), "/", len(test))

# 按「在 / 不在 DPR」分组，比较 root 和 R 的准确率
for name in ["dtrain", "ddev"]:
    fn = f"outputs/tree/nq_{name}_tree15_Llama-3.1-8B-Instruct_accumulate.jsonl"
    groups = defaultdict(lambda: {"root": [], "R": []})
    for l in open(fn):
        n = json.loads(l)
        path = n["node_id"][len(n["qid"]):].lstrip("/") or "root"
        if path in ("root", "R"):
            g = "in DPR" if norm(n["question"]) in dpr else "not in DPR"
            groups[g][path].append(n["metrics"]["inacc"])
    print(f"\n{name}:")
    for g in ["in DPR", "not in DPR"]:
        r, R = groups[g]["root"], groups[g]["R"]
        if r:
            print(f"  {g:11s} n={len(r):5d}  root {sum(r)/len(r):.3f}  R {sum(R)/len(R):.3f}")