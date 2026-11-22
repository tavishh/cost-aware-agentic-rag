"""e5 retrieval for the tree pipeline. Drop-in replacement for rewrite_retrieve.py,
split into GPU search steps (vllm env) and annotation steps (pyserini env):

  search-cache    questions -> raw top-50 cache                      (vllm env, GPU)
  annotate-cache  raw cache -> cache with has_answer + passages_L0   (pyserini env)
  search-layer    rewrites_L{l} -> rankings_L{l}.raw                 (vllm env, GPU)
  annotate-layer  rankings_L{l}.raw -> rankings_L{l} + passages_L{l} (pyserini env)

Output files/formats are identical to the BM25 pipeline, so tree_expand.py / tree_answer.py
need no retriever-specific logic. has_answer uses the same get_text/has_answer as BM25.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cache"))

import argparse, glob, json
from tree_common import load_jsonl, write_jsonl, work_dir

EMB_DIR = os.environ.get("E5_EMB_DIR", "/work/neu/p2026_0112_neu/e5/emb")
MODEL_DIR = os.environ.get("E5_MODEL_DIR", os.path.expanduser("~/models/e5-base-v2"))


# ---------------- GPU search (vllm env) ----------------
def e5_search(queries, k, chunk=500_000, qbs=1024):
    """Exact inner-product top-k over all shards. Returns [(docids[str], scores[float])]."""
    if not queries:
        return []
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).cuda().eval()
    qs = []
    with torch.inference_mode():
        for b in range(0, len(queries), 256):
            enc = tok([f"query: {q}" for q in queries[b:b + 256]], max_length=64,
                      padding=True, truncation=True, return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = ((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).float()
            qs.append(torch.nn.functional.normalize(v, dim=-1).half())
    del model
    torch.cuda.empty_cache()

    Q = torch.cat(qs)
    nq = Q.shape[0]
    best_s = torch.full((nq, k), -1e4, device="cuda")
    best_i = torch.full((nq, k), -1, dtype=torch.long, device="cuda")
    shards = sorted(glob.glob(f"{EMB_DIR}/emb_??.npy"))
    assert len(shards) == 8, f"expected 8 shards in {EMB_DIR}, found {len(shards)}"
    with torch.inference_mode():
        for sp in shards:
            E = np.load(sp, mmap_mode="r")
            ids = np.load(sp.replace("emb_", "ids_"))
            assert len(E) == len(ids)
            for c in range(0, len(E), chunk):
                e = torch.from_numpy(np.array(E[c:c + chunk])).cuda()
                gid = torch.from_numpy(ids[c:c + chunk]).cuda()
                for qb in range(0, nq, qbs):
                    s = (Q[qb:qb + qbs] @ e.T).float()
                    ts, ti = s.topk(k, dim=1)
                    cs = torch.cat([best_s[qb:qb + qbs], ts], 1)
                    ci = torch.cat([best_i[qb:qb + qbs], gid[ti]], 1)
                    ns, nidx = cs.topk(k, dim=1)
                    best_s[qb:qb + qbs] = ns
                    best_i[qb:qb + qbs] = ci.gather(1, nidx)
            print("searched", os.path.basename(sp), flush=True)
    S, I = best_s.cpu().tolist(), best_i.cpu().tolist()
    return [([str(d) for d in i], [round(x, 4) for x in s]) for s, i in zip(S, I)]


# ---------------- annotation helpers (pyserini env) ----------------
def split_passage(raw):
    """Same title/text parsing as rewrite_retrieve.py."""
    title, _, text = raw.replace('""', '"').partition("\n")
    return title.strip('"'), text


def known_docids(wd, layer):
    known = set()
    for fn in ["outputs/retrieval/passages.jsonl"] + glob.glob(f"{wd}/passages_L*.jsonl"):
        if not fn.endswith(f"passages_L{layer}.jsonl"):
            known.update(json.loads(l)["docid"] for l in open(fn))
    return known


def raw_cache_path(cache):
    return cache.replace(".jsonl", ".raw.jsonl")


# ---------------- subcommands ----------------
def search_cache(a):
    rows = load_jsonl(a.questions)
    res = e5_search([r["question"] for r in rows], a.k)
    out = raw_cache_path(a.cache)
    write_jsonl(out, [{"qid": r["qid"], "question": r["question"], "gold": r["gold"],
                       "docids": d, "scores": s} for r, (d, s) in zip(rows, res)])
    print(f"wrote {out}  ({len(rows)} questions, k={a.k})")


def annotate_cache(a):
    from retrieval.bm25 import get_text, has_answer
    rows = load_jsonl(raw_cache_path(a.cache))
    wd = work_dir(a.cache, a.n, a.model)
    known = known_docids(wd, 0)
    new_p = {}
    for i, r in enumerate(rows):
        texts = {d: get_text(d) for d in r["docids"]}
        r["has_answer"] = [has_answer(texts[d], r["gold"]) for d in r["docids"]]
        if i < a.n:   # passages the tree can use for the original question (first `depth` ranks)
            for d in r["docids"][:a.depth]:
                if d not in known and d not in new_p:
                    title, text = split_passage(texts[d])
                    new_p[d] = {"docid": d, "title": title, "text": text}
    write_jsonl(a.cache, rows)
    write_jsonl(f"{wd}/passages_L0.jsonl", list(new_p.values()))
    json.dump({"retriever": "intfloat/e5-base-v2", "query_prefix": "query: ",
               "passage_format": "passage: {title} {text}", "pooling": "mean+l2",
               "search": "exact inner product over 21M DPR passages, fp16",
               "k": len(rows[0]["docids"]), "n": len(rows),
               "has_answer": "retrieval.bm25.has_answer on get_text(docid)"},
              open(a.cache.replace(".jsonl", ".meta.json"), "w"), indent=2)
    ks = [1, 5, 10, 20, 50]
    hit = "  ".join(f"@{k}={100 * sum(any(r['has_answer'][:k]) for r in rows) / len(rows):.1f}"
                    for k in ks)
    print(f"wrote {a.cache}  Hit {hit}")
    print(f"passages_L0: {len(new_p)} new -> {wd}/passages_L0.jsonl")


def search_layer(a):
    wd = work_dir(a.cache, a.n, a.model)
    rws = [x for x in load_jsonl(f"{wd}/rewrites_L{a.layer}.jsonl") if x["new_query"]]
    uniq = {}
    for x in rws:
        uniq.setdefault((x["qid"], x["new_query"]), x["gold"])
    pairs = list(uniq)
    res = e5_search([q for _, q in pairs], a.depth)
    out = f"{wd}/raw_rankings_L{a.layer}.jsonl"
    write_jsonl(out, [{"qid": qid, "query": q, "gold": uniq[(qid, q)], "docids": d, "scores": s}
                      for (qid, q), (d, s) in zip(pairs, res)])
    print(f"wrote {out}  ({len(pairs)} unique rewrite queries, k={a.depth})")


def annotate_layer(a):
    from retrieval.bm25 import get_text, has_answer
    wd = work_dir(a.cache, a.n, a.model)
    raw = load_jsonl(f"{wd}/raw_rankings_L{a.layer}.jsonl")
    known = known_docids(wd, a.layer)
    rankings, new_p = [], {}
    for x in raw:
        texts = {d: get_text(d) for d in x["docids"]}
        rankings.append({"qid": x["qid"], "query": x["query"], "docids": x["docids"],
                         "scores": x["scores"],
                         "has_answer": [has_answer(texts[d], x["gold"]) for d in x["docids"]]})
        for d in x["docids"]:
            if d not in known and d not in new_p:
                title, text = split_passage(texts[d])
                new_p[d] = {"docid": d, "title": title, "text": text}
    write_jsonl(f"{wd}/rankings_L{a.layer}.jsonl", rankings)
    write_jsonl(f"{wd}/passages_L{a.layer}.jsonl", list(new_p.values()))
    hit5 = sum(any(r["has_answer"][:5]) for r in rankings) / max(len(rankings), 1)
    print(f"queries: {len(rankings)}  new passages: {len(new_p)}  Hit@5 of rewrites: {hit5:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["search-cache", "annotate-cache", "search-layer", "annotate-layer"])
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cache", required=True, help="e.g. outputs/retrieval/nq_arag_test_e5_top50.jsonl")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--questions", help="search-cache: jsonl with qid, question, gold")
    p.add_argument("--k", type=int, default=50, help="search-cache depth")
    p.add_argument("--layer", type=int)
    p.add_argument("--depth", type=int, default=15, help="3 steps x 5, same as rewrite_retrieve.py")
    a = p.parse_args()
    if a.cmd in ("search-layer", "annotate-layer"):
        assert a.layer is not None, "--layer required"
    if a.cmd == "search-cache":
        assert a.questions, "--questions required"
    {"search-cache": search_cache, "annotate-cache": annotate_cache,
     "search-layer": search_layer, "annotate-layer": annotate_layer}[a.cmd](a)
