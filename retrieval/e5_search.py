"""Exact (brute-force) top-K over all shards. Input jsonl needs qid, question, gold.
Output: {qid, question, gold, docids[K](str), scores[K]} ; has_answer added later on Mac."""
import argparse, glob, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from e5_encode import MODEL, mean_pool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--qbs", type=int, default=1024)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.queries)]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    qs = []
    with torch.inference_mode():
        for b in range(0, len(rows), 256):
            enc = tok([f"query: {r['question']}" for r in rows[b:b + 256]], max_length=64,
                      padding=True, truncation=True, return_tensors="pt").to("cuda")
            v = mean_pool(model(**enc).last_hidden_state, enc["attention_mask"]).float()
            qs.append(torch.nn.functional.normalize(v, dim=-1).half())
    Q = torch.cat(qs)
    nq = Q.shape[0]
    best_s = torch.full((nq, a.k), -1e4, device="cuda")
    best_i = torch.full((nq, a.k), -1, dtype=torch.long, device="cuda")

    shards = sorted(glob.glob(f"{a.emb_dir}/emb_??.npy"))
    assert len(shards) > 0, "no shards found"
    print("shards:", len(shards), flush=True)
    with torch.inference_mode():
        for sp in shards:
            E = np.load(sp, mmap_mode="r")
            ids = np.load(sp.replace("emb_", "ids_"))
            assert len(E) == len(ids)
            for c in range(0, len(E), a.chunk):
                e = torch.from_numpy(np.ascontiguousarray(E[c:c + a.chunk])).cuda()
                gid = torch.from_numpy(ids[c:c + a.chunk]).cuda()
                for qb in range(0, nq, a.qbs):
                    s = (Q[qb:qb + a.qbs] @ e.T).float()
                    ts, ti = s.topk(a.k, dim=1)
                    cs = torch.cat([best_s[qb:qb + a.qbs], ts], 1)
                    ci = torch.cat([best_i[qb:qb + a.qbs], gid[ti]], 1)
                    ns, nidx = cs.topk(a.k, dim=1)
                    best_s[qb:qb + a.qbs] = ns
                    best_i[qb:qb + a.qbs] = ci.gather(1, nidx)
            print("searched", sp, flush=True)

    S, I = best_s.cpu().tolist(), best_i.cpu().tolist()
    with open(a.out, "w") as f:
        for r, s, i in zip(rows, S, I):
            f.write(json.dumps({"qid": r["qid"], "question": r["question"], "gold": r["gold"],
                                "docids": [str(x) for x in i], "scores": s},
                               ensure_ascii=False) + "\n")
    print("wrote", a.out, nq)
