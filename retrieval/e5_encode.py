"""Encode one corpus shard with e5-base-v2 -> emb_XX.npy (fp16, L2-normalized) + ids_XX.npy (int64)."""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Local copy or HF id; set E5_MODEL_DIR on clusters without HF access.
MODEL = os.environ.get("E5_MODEL_DIR", "intfloat/e5-base-v2")


def mean_pool(h, m):
    m = m.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="benchmark: only first N docs")
    a = ap.parse_args()

    ids, texts = [], []
    with open(f"{a.corpus_dir}/corpus_{a.shard:02d}.jsonl") as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            d = json.loads(line)
            ids.append(int(d["id"]))
            texts.append(f"passage: {d['title']} {d['text']}")
    n = len(texts)
    print("docs", n, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

    order = np.argsort([len(t) for t in texts])[::-1]      # long first: OOM shows up early
    tag = f"{a.shard:02d}" + (f"_lim{a.limit}" if a.limit else "")
    emb = np.lib.format.open_memmap(f"{a.out_dir}/emb_{tag}.npy", mode="w+",
                                    dtype=np.float16, shape=(n, 768))
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, n, a.bs):
            idx = order[b:b + a.bs]
            enc = tok([texts[i] for i in idx], max_length=a.max_len, padding=True,
                      truncation=True, return_tensors="pt").to("cuda")
            out = model(**enc).last_hidden_state
            v = torch.nn.functional.normalize(mean_pool(out, enc["attention_mask"]).float(), dim=-1)
            emb[idx] = v.half().cpu().numpy()
            if (b // a.bs) % 200 == 0:
                r = (b + len(idx)) / (time.time() - t0)
                print(f"{b + len(idx)}/{n}  {r:.0f} psg/s  eta {(n - b) / r / 60:.1f} min", flush=True)
    emb.flush()
    np.save(f"{a.out_dir}/ids_{tag}.npy", np.array(ids, dtype=np.int64))
    print(f"done {n} in {(time.time() - t0) / 60:.1f} min", flush=True)
