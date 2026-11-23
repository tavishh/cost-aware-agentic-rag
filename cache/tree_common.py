import os, re, json, glob, string
from collections import Counter

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def cache_stem(cache):
    """nq_X_bm25_top50.jsonl -> nq_X (unchanged for BM25); nq_X_e5_top50.jsonl -> nq_X_e5"""
    b = os.path.basename(cache)
    if "_bm25" in b:
        return b.split("_bm25")[0]
    if "_e5" in b:
        return b.split("_e5")[0] + "_e5"
    return b.split(".jsonl")[0]

def work_dir(cache, n, model):
    return f"outputs/tree_work/{cache_stem(cache)}_n{n}_{model.split('/')[-1]}"

def load_passages(wd, need=None):
    files = ["outputs/retrieval/passages.jsonl"] + sorted(glob.glob(f"{wd}/passages_L*.jsonl"))
    P = {}
    for fn in files:
        for l in open(fn):
            x = json.loads(l)
            if need is None or x["docid"] in need:
                P[x["docid"]] = x
    return P

def load_rankings(rows, wd):
    """rankings[qid][查询] = (docids, scores, has_answer)"""
    R = {r["qid"]: {r["question"]: (r["docids"], r["scores"], r["has_answer"])} for r in rows}
    for fn in sorted(glob.glob(f"{wd}/rankings_L*.jsonl")):
        for x in load_jsonl(fn):
            if x["qid"] in R:
                R[x["qid"]].setdefault(x["query"], (x["docids"], x["scores"], x["has_answer"]))
    return R

def call_record(o, latency):
    g = o.outputs[0]
    return {"text": g.text.strip(),
            "tokens_in": len(o.prompt_token_ids),
            "tokens_out": len(g.token_ids),
            "latency_s": round(latency, 4),
            "token_logprobs": [[[lp.decoded_token, round(lp.logprob, 4)]
                                for lp in sorted(pos.values(), key=lambda x: -x.logprob)]
                               for pos in (g.logprobs or [])]}

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