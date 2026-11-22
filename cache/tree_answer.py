import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "cache"))

import glob, json, time, argparse
from collections import defaultdict
from tree_common import (load_jsonl, write_jsonl, work_dir, load_passages,
                         call_record, em, f1, inacc, cache_stem)
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
p.add_argument("--cache", default="outputs/retrieval/nq_test_bm25_top50.jsonl")
p.add_argument("--n", type=int, default=500)
p.add_argument("--context", choices=["accumulate", "latest"], default="accumulate")
p.add_argument("--prompt", choices=["simple", "fewshot"], default="simple")
p.add_argument("--shots", type=int, default=5)
p.add_argument("--stop", choices=["newline", "none"], default="newline")
p.add_argument("--max_tokens", type=int, default=32)
p.add_argument("--only", default="")   # 例如 root,R；为空表示全部节点
args = p.parse_args()

def path_of(n):
    return n["node_id"][len(n["qid"]):].lstrip("/") or "root"

wd = work_dir(args.cache, args.n, args.model)
nodes = []
for fn in sorted(glob.glob(f"{wd}/nodes_L*.jsonl")):
    nodes += load_jsonl(fn)
if args.only:
    keep = set(args.only.split(","))
    nodes = [n for n in nodes if path_of(n) in keep]
need = {d for n in nodes for rt in n["retrievals"] for d in rt["docids"]}
passages = load_passages(wd, need)

# ---------- prompt ----------
DEMO_FILES = {"noctx": "prompts/no_context_direct_qa_flan_t5.txt",
              "ctx": "prompts/gold_with_2_distractors_context_direct_qa_flan_t5.txt"}

def load_demos(path, k):
    """按 '# METADATA:' 切分示例，去掉每块第一行的元数据，取前 k 个"""
    demos = []
    for block in open(path).read().split("# METADATA:"):
        block = block.strip()
        if block:
            demos.append(block.split("\n", 1)[1].strip())
    return "\n\n\n".join(demos[:k])

if args.prompt == "fewshot":
    SYSTEM = "Answer the last question in the same format as the examples. Give only the short answer."
    DEMOS = {key: load_demos(path, args.shots) for key, path in DEMO_FILES.items()}
else:
    SYSTEM = "Answer the question with a short phrase of a few words. Do not explain."

def context_docids(rts):
    if not rts:
        return []
    return rts[-1]["docids"] if args.context == "latest" else [d for rt in rts for d in rt["docids"]]

def make_msgs(question, docids):
    ctx = "\n\n".join(f"Wikipedia Title: {passages[d]['title']}\n{passages[d]['text']}"
                      for d in docids)
    if args.prompt == "fewshot":
        block = (ctx + "\n\n" if docids else "") + f"Q: Answer the following question.\n{question}\nA:"
        user = DEMOS["ctx" if docids else "noctx"] + "\n\n\n" + block
    elif docids:
        user = f"{ctx}\n\nQuestion: {question}\nAnswer:"
    else:
        user = f"Question: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def clean_answer(t):
    """不截断时模型可能接着写下一个示例，在空行或下一个 Q: 处切掉"""
    t = t.strip()
    for sep in ["\n\n", "\nQ:"]:
        t = t.split(sep)[0]
    t = t.strip()
    if args.stop == "newline":
        t = t.split("\n")[0].strip()
    return t[2:].strip() if t.startswith("A:") else t

# ---------- 生成 ----------
llm = LLM(model=args.model, dtype="bfloat16", seed=0)
params = SamplingParams(temperature=0, max_tokens=args.max_tokens, logprobs=20,
                        stop=["\n"] if args.stop == "newline" else None)
msgs = [make_msgs(n["question"], context_docids(n["retrievals"])) for n in nodes]
t0 = time.time()
outs = llm.chat(msgs, params, use_tqdm=True)
per = (time.time() - t0) / len(nodes)

for n, o in zip(nodes, outs):
    n["context_docids"] = context_docids(n["retrievals"])
    a = n["answer_call"] = call_record(o, per)
    a["text"] = clean_answer(a["text"])
    n["metrics"] = {"em": em(a["text"], n["gold"]),
                    "f1": round(f1(a["text"], n["gold"]), 4),
                    "inacc": inacc(a["text"], n["gold"])}
    rc = n["path_rewrite_calls"]
    n["cum_cost"] = {"retrieval_calls": len(n["retrievals"]),
                     "llm_calls": 1 + len(rc),
                     "tokens_in": a["tokens_in"] + sum(c["tokens_in"] for c in rc),
                     "tokens_out": a["tokens_out"] + sum(c["tokens_out"] for c in rc),
                     "latency_s": round(a["latency_s"] + sum(c["latency_s"] for c in rc), 4)}

# ---------- 保存 ----------
stem = cache_stem(args.cache)
suffix = ("_fewshot" if args.prompt == "fewshot" else "") \
         + ("_nostop" if args.stop == "none" else "") \
         + (f"_only-{args.only.replace(',', '-')}" if args.only else "")
out = f"outputs/tree/{stem}_tree15_{args.model.split('/')[-1]}_{args.context}{suffix}.jsonl"
write_jsonl(out, nodes)

meta = {"model": args.model, "cache": args.cache, "n": args.n, "context": args.context,
        "work_dir": wd, "prompt": args.prompt, "system_prompt": SYSTEM,
        "temperature": 0, "logprobs_topk": 20, "stop": args.stop,
        "max_tokens": args.max_tokens, "only": args.only}
if args.prompt == "fewshot":
    tok = llm.get_tokenizer()
    meta.update({"shots": args.shots, "demo_files": DEMO_FILES,
                 "demo_tokens": {k: len(tok.encode(v)) for k, v in DEMOS.items()}})
json.dump(meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=2)

# ---------- 汇总 ----------
by_path = defaultdict(list)
for n in nodes:
    by_path[path_of(n)].append(n)
print(f"\n{'path':10s} {'nodes':>6} {'EM':>6} {'InAcc':>6} {'ans_tok':>7} {'cum_tok_in':>10}")
for path in sorted(by_path, key=lambda s: (s.count("/"), s)):
    sub = by_path[path]
    m = len(sub)
    print(f"{path:10s} {m:>6} "
          f"{sum(x['metrics']['em'] for x in sub)/m:>6.3f} "
          f"{sum(x['metrics']['inacc'] for x in sub)/m:>6.3f} "
          f"{sum(x['answer_call']['tokens_out'] for x in sub)/m:>7.1f} "
          f"{sum(x['cum_cost']['tokens_in'] for x in sub)/m:>10.0f}")

best_all, best_r = defaultdict(int), defaultdict(int)
for n in nodes:
    v = n["metrics"]["inacc"]
    best_all[n["qid"]] = max(best_all[n["qid"]], v)
    if "W" not in path_of(n):
        best_r[n["qid"]] = max(best_r[n["qid"]], v)
print(f"\noracle InAcc  R-only: {sum(best_r.values())/len(best_r):.3f}   "
      f"full tree: {sum(best_all.values())/len(best_all):.3f}")
print("saved:", out)