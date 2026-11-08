import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import json, time, re, string, argparse
from collections import Counter, defaultdict
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--cache", default="outputs/retrieval/nq_test_bm25_top50.jsonl")
p.add_argument("--n", type=int, default=500)
p.add_argument("--k", type=int, default=5)
p.add_argument("--max_steps", type=int, default=3)
p.add_argument("--context", choices=["accumulate", "latest"], default="accumulate")
p.add_argument("--out", default=None)
args = p.parse_args()

if args.out is None:
    short = args.model.split("/")[-1]
    stem = os.path.basename(args.cache).split("_bm25")[0]
    args.out = f"outputs/tree/{stem}_tree_{short}_{args.context}.jsonl"

# ---------- 读检索缓存和段落 ----------
rows = [json.loads(l) for l in open(args.cache)][:args.n]
max_rank = args.k * args.max_steps
need = {d for r in rows for d in r["docids"][:max_rank]}
passages = {}
for l in open("outputs/retrieval/passages.jsonl"):
    x = json.loads(l)
    if x["docid"] in need:
        passages[x["docid"]] = x

# rankings[qid][查询] = (docids, scores, has_answer)
# 目前只有原始问题；以后 rewrite 产生的新查询也放在这里
rankings = {r["qid"]: {r["question"]: (r["docids"], r["scores"], r["has_answer"])}
            for r in rows}

# ---------- prompt ----------
SYSTEM = "Answer the question with a short phrase of a few words. Do not explain."

def context_docids(retrievals):
    if not retrievals:
        return []
    if args.context == "latest":
        return retrievals[-1]["docids"]
    return [d for rt in retrievals for d in rt["docids"]]

def make_msgs(question, docids):
    if docids:
        ctx = "\n\n".join(
            f"Wikipedia Title: {passages[d]['title']}\n{passages[d]['text']}" for d in docids)
        user = f"{ctx}\n\nQuestion: {question}\nAnswer:"
    else:
        user = f"Question: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]

# ---------- 评测指标 ----------
def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(pred, golds):
    return int(any(norm(pred) == norm(g) for g in golds))

def f1(pred, golds):
    best, pt = 0.0, norm(pred).split()
    for g in golds:
        gt = norm(g).split()
        same = sum((Counter(pt) & Counter(gt)).values())
        if same:
            pr, rc = same / len(pt), same / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best

def inacc(pred, golds):
    return int(any(norm(g) and norm(g) in norm(pred) for g in golds))

# ---------- 展开树 ----------
KEEP = ("qid", "question", "gold")

def expand_retrieve(par):
    """对当前查询再检索一次，取下一批 k 条。"""
    q = par["current_query"]
    docids, scores, ha = rankings[par["qid"]][q]
    start = sum(1 for rt in par["retrievals"] if rt["query"] == q) * args.k
    if start + args.k > len(docids):
        return None
    rt = {"query": q,
          "source": "original" if q == par["question"] else "rewrite",
          "ranks": [start + 1, start + args.k],
          "docids": docids[start:start + args.k],
          "scores": scores[start:start + args.k],
          "has_answer": ha[start:start + args.k]}
    return {**{k: par[k] for k in KEEP},
            "node_id": par["node_id"] + "/R", "parent_id": par["node_id"],
            "action": "retrieve", "step": par["step"] + 1,
            "current_query": q, "retrievals": par["retrievals"] + [rt]}

EXPANDERS = {"R": expand_retrieve}   # 以后在这里加 "W": expand_rewrite

nodes = []
for r in rows:
    root = {
            "qid": r["qid"], "question": r["question"], "gold": r["gold"],
            "node_id": r["qid"], "parent_id": None, "action": None, "step": 0,
            "current_query": r["question"], "retrievals": []}
    nodes.append(root)
    frontier = [root]
    for _ in range(args.max_steps):
        nxt = []
        for par in frontier:
            for act, fn in EXPANDERS.items():
                child = fn(par)
                if child:
                    nodes.append(child)
                    nxt.append(child)
        frontier = nxt

print(f"questions: {len(rows)}  nodes: {len(nodes)}  context: {args.context}")

# ---------- 生成 ----------
llm = LLM(model=args.model, dtype="bfloat16", seed=0)
params = SamplingParams(temperature=0, max_tokens=32, logprobs=20)

msgs = [make_msgs(n["question"], context_docids(n["retrievals"])) for n in nodes]
t0 = time.time()
outs = llm.chat(msgs, params, use_tqdm=True)
per_call = (time.time() - t0) / len(nodes)

for n, o in zip(nodes, outs):
    g = o.outputs[0]
    text = g.text.strip()
    n["context_docids"] = context_docids(n["retrievals"])
    n["answer_call"] = {
        "text": text,
        "tokens_in": len(o.prompt_token_ids),
        "tokens_out": len(g.token_ids),
        "latency_s": round(per_call, 4),
        "token_logprobs": [
            [[lp.decoded_token, round(lp.logprob, 4)]
             for lp in sorted(pos.values(), key=lambda x: -x.logprob)]
            for pos in g.logprobs],
    }
    n["metrics"] = {"em": em(text, n["gold"]),
                    "f1": round(f1(text, n["gold"]), 4),
                    "inacc": inacc(text, n["gold"])}
    # 在这个节点作答的总成本 = 路径上的检索次数 + 这一次回答调用
    # （以后加 rewrite 时，再加上路径上的改写调用）
    n["cum_cost"] = {"retrieval_calls": len(n["retrievals"]),
                     "llm_calls": 1,
                     "tokens_in": n["answer_call"]["tokens_in"],
                     "latency_s": n["answer_call"]["latency_s"]}

# ---------- 保存 ----------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    for n in nodes:
        f.write(json.dumps(n, ensure_ascii=False) + "\n")
json.dump({"model": args.model, "cache": args.cache, "n": len(rows), "k": args.k,
           "max_steps": args.max_steps, "context": args.context,
           "actions": list(EXPANDERS), "temperature": 0, "logprobs_topk": 20,
           "system_prompt": SYSTEM},
          open(args.out.replace(".jsonl", ".meta.json"), "w"), indent=2)

# ---------- 汇总 ----------
by_step = defaultdict(list)
for n in nodes:
    by_step[n["step"]].append(n)
print(f"\n{'step':>4} {'nodes':>6} {'EM':>6} {'InAcc':>6} {'tok_in':>7} {'evid':>6}")
for s in sorted(by_step):
    sub = by_step[s]
    m = len(sub)
    evid = sum(any(h for rt in x["retrievals"] for h in rt["has_answer"]) for x in sub)
    print(f"{s:>4} {m:>6} "
          f"{sum(x['metrics']['em'] for x in sub)/m:>6.3f} "
          f"{sum(x['metrics']['inacc'] for x in sub)/m:>6.3f} "
          f"{sum(x['answer_call']['tokens_in'] for x in sub)/m:>7.0f} "
          f"{evid/m:>6.3f}")

best = defaultdict(int)
for n in nodes:
    best[n["qid"]] = max(best[n["qid"]], n["metrics"]["inacc"])
print(f"\noracle InAcc (best node per question): {sum(best.values())/len(best):.3f}")
print("saved:", args.out)
