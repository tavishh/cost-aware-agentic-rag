import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "cache"))

import re, time, argparse
from tree_common import load_jsonl, write_jsonl, work_dir, load_passages, load_rankings, call_record

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--cache", default="outputs/retrieval/nq_test_bm25_top50.jsonl")
p.add_argument("--n", type=int, default=500)
p.add_argument("--k", type=int, default=5)
p.add_argument("--max_steps", type=int, default=3)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--rewrite_ctx", choices=["titles", "none"], default="titles")
args = p.parse_args()

rows = load_jsonl(args.cache)[:args.n]
wd = work_dir(args.cache, args.n, args.model)
rankings = load_rankings(rows, wd)
K = args.k

def new_retrieval(par, q):
    """对查询 q 取下一批 K 条；该查询没有检索结果或已用完则返回 None"""
    if q not in rankings[par["qid"]]:
        return None
    docids, scores, ha = rankings[par["qid"]][q]
    start = sum(1 for rt in par["retrievals"] if rt["query"] == q) * K
    if start + K > len(docids):
        return None
    return {"query": q, "source": "original" if q == par["question"] else "rewrite",
            "ranks": [start + 1, start + K],
            "docids": docids[start:start + K], "scores": scores[start:start + K],
            "has_answer": ha[start:start + K]}

def child(par, tag, action, q, rt, extra_calls):
    return {"qid": par["qid"], "question": par["question"], "gold": par["gold"],
            "node_id": f"{par['node_id']}/{tag}", "parent_id": par["node_id"],
            "action": action, "step": par["step"] + 1, "current_query": q,
            "retrievals": par["retrievals"] + [rt],
            "path_rewrite_calls": par["path_rewrite_calls"] + extra_calls}

# ---------- 建本层节点 ----------
if args.layer == 0:
    nodes = [{"qid": r["qid"], "question": r["question"], "gold": r["gold"],
              "node_id": r["qid"], "parent_id": None, "action": None, "step": 0,
              "current_query": r["question"], "retrievals": [], "path_rewrite_calls": []}
             for r in rows]
else:
    parents = load_jsonl(f"{wd}/nodes_L{args.layer - 1}.jsonl")
    rws = {x["parent_id"]: x for x in load_jsonl(f"{wd}/rewrites_L{args.layer}.jsonl")}
    nodes, no_w = [], 0
    for par in parents:
        rt = new_retrieval(par, par["current_query"])
        if rt:
            nodes.append(child(par, "R", "retrieve", par["current_query"], rt, []))
        rw = rws.get(par["node_id"])
        rt = new_retrieval(par, rw["new_query"]) if rw and rw["new_query"] else None
        if rt:
            call = {k: rw["rewrite_call"][k] for k in ("tokens_in", "tokens_out", "latency_s")}
            c = child(par, "W", "rewrite", rw["new_query"], rt, [call])
            c["rewrite_call"] = rw["rewrite_call"]
            nodes.append(c)
        else:
            no_w += 1
    print(f"parents: {len(parents)}  missing W children: {no_w}")

write_jsonl(f"{wd}/nodes_L{args.layer}.jsonl", nodes)
print(f"layer {args.layer}: {len(nodes)} nodes -> {wd}/nodes_L{args.layer}.jsonl")

# ---------- 为本层节点生成改写查询（最后一层不需要） ----------
if args.layer < args.max_steps:
    from vllm import LLM, SamplingParams
    need = {d for n in nodes for rt in n["retrievals"] for d in rt["docids"]}
    passages = load_passages(wd, need)

    SYSTEM = "You write search queries for a Wikipedia search engine."

    def rewrite_msgs(n):
        prev = list(dict.fromkeys([n["question"]] + [rt["query"] for rt in n["retrievals"]]))
        parts = [f"Question: {n['question']}",
                 "Previous search queries:\n" + "\n".join(f"- {q}" for q in prev)]
        if args.rewrite_ctx == "titles" and n["retrievals"]:
            titles = list(dict.fromkeys(passages[d]["title"]
                                        for rt in n["retrievals"] for d in rt["docids"]))
            parts.append("Titles of passages already retrieved:\n" +
                         "\n".join(f"- {t}" for t in titles))
        parts.append("Write one new search query that is different from the previous queries "
                     "and more likely to retrieve a passage that answers the question. "
                     "Output only the query.")
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)}]

    def clean_query(t):
        t = t.strip().split("\n")[0].strip().strip('"').strip("'").strip()
        t = re.sub(r"^(new\s+)?(search\s+)?query\s*:\s*", "", t, flags=re.I)
        return t[:200] or None

    llm = LLM(model=args.model, dtype="bfloat16", seed=0)
    params = SamplingParams(temperature=0, max_tokens=48, logprobs=20)
    msgs = [rewrite_msgs(n) for n in nodes]
    t0 = time.time()
    outs = llm.chat(msgs, params, use_tqdm=True)
    per = (time.time() - t0) / len(nodes)

    out = []
    for n, o in zip(nodes, outs):
        rec = call_record(o, per)
        out.append({"qid": n["qid"], "parent_id": n["node_id"], "gold": n["gold"],
                    "new_query": clean_query(rec["text"]), "rewrite_call": rec})
    write_jsonl(f"{wd}/rewrites_L{args.layer + 1}.jsonl", out)

    print(f"rewrites -> {wd}/rewrites_L{args.layer + 1}.jsonl  "
          f"(empty: {sum(x['new_query'] is None for x in out)})")
    for x in out[:5]:
        print(f"  {x['parent_id']:25s} -> {x['new_query']}")