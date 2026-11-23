"""Dump DPR wiki passages from the local Lucene index -> sharded jsonl {id,title,text}.
docid comes straight from Lucene, so alignment with BM25 is guaranteed."""
import argparse, json, os, multiprocessing as mp


def parse(raw):
    c = json.loads(raw)["contents"]           # pyserini DPR: '"Title"\ntext'
    title, _, text = c.partition("\n")
    return title.strip().strip('"'), text.strip()


def work(args):
    index, shard, start, end, outdir = args
    from pyserini.search.lucene import LuceneSearcher   # import inside spawn worker (JVM)
    s = LuceneSearcher(index)
    path = f"{outdir}/corpus_{shard:02d}.jsonl"
    with open(path, "w") as f:
        for i in range(start, end):
            d = s.doc(i)                      # int = internal Lucene id
            title, text = parse(d.raw())
            f.write(json.dumps({"id": d.docid(), "title": title, "text": text},
                               ensure_ascii=False) + "\n")
    return shard, end - start


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num_shards", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    from pyserini.search.lucene import LuceneSearcher
    s0 = LuceneSearcher(a.index)
    n = s0.num_docs
    print("num_docs", n, flush=True)
    d0 = s0.doc(0)
    print("sample docid:", d0.docid(), flush=True)
    print("sample raw:", d0.raw()[:300], flush=True)
    print("sample parsed:", parse(d0.raw()), flush=True)
    del s0, d0

    step = (n + a.num_shards - 1) // a.num_shards
    jobs = [(a.index, k, k * step, min(n, (k + 1) * step), a.outdir) for k in range(a.num_shards)]
    with mp.get_context("spawn").Pool(a.num_shards) as p:
        for shard, cnt in p.imap_unordered(work, jobs):
            print(f"shard {shard} done: {cnt}", flush=True)
