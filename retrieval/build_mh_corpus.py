"""Build Pyserini JsonCollections for the multi-hop corpora, following IRCoT/Adaptive-RAG's
retriever_server/build_index.py:

  hotpotqa         one doc per 2017 Wikipedia abstract (sentences joined with " ").
                   Source: mteb/hotpotqa corpus.jsonl (5,233,329 abstracts; same set as the
                   original enwiki-20171001 abstracts dump, which is no longer downloadable).
  2wikimultihopqa  union of all context paragraphs in train/dev/test, deduplicated on title+text.
  musique          union of all paragraphs in ans+full x dev/test/train, deduplicated on title+text.

Docids are prefixed per corpus (hotpotqa_*, 2wiki_*, musique_*) so they never collide with
DPR docids in the shared passage store. contents = "title\ntext" (same split as DPR's
'"title"\ntext', so get_text/has_answer/passage parsing work unchanged).

Run from ~/Downloads/Adaptive-RAG (where raw_data/ lives):
  python build_mh_corpus.py musique --out ~/mh_corpus
"""
import argparse, json, os, sys


def write_docs(docs, out_dir, name, shard_size=1_000_000):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    n, f, shard = 0, None, -1
    for docid, title, text in docs:
        if n % shard_size == 0:
            if f:
                f.close()
            shard += 1
            f = open(os.path.join(d, f"docs{shard:02d}.jsonl"), "w", encoding="utf-8")
        f.write(json.dumps({"id": docid, "contents": f"{title}\n{text}"}, ensure_ascii=False) + "\n")
        n += 1
    if f:
        f.close()
    print(f"{name}: {n} docs -> {d}")
    return n


def hotpotqa_docs(src):
    with open(src, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            yield f"hotpotqa_{x['_id']}", x["title"].strip(), " ".join(x["text"].split())


def dedup(pairs, prefix):
    seen, i = set(), 0
    for title, text in pairs:
        key = (title, text)
        if key in seen:
            continue
        seen.add(key)
        yield f"{prefix}_{i}", title, text
        i += 1


def twowiki_pairs(raw):
    names = ["train.json", "dev.json", "test.json"]   # same order as build_index.py
    for name in names:
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
        for inst in json.load(open(path, encoding="utf-8")):
            for title, sents in inst["context"]:
                yield title, " ".join(sents)


def musique_pairs(raw):
    names = [f"musique_{v}_v1.0_{s}.jsonl" for v in ("ans", "full") for s in ("dev", "test", "train")]
    for name in names:                                  # same order as build_index.py
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
        for line in open(path, encoding="utf-8"):
            if line.strip():
                for p in json.loads(line)["paragraphs"]:
                    yield p["title"], p["paragraph_text"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--raw", default="raw_data")
    ap.add_argument("--out", default=os.path.expanduser("~/mh_corpus"))
    a = ap.parse_args()
    if a.dataset == "hotpotqa":
        docs = hotpotqa_docs(os.path.join(a.raw, "hotpotqa", "mteb", "corpus.jsonl"))
    elif a.dataset == "2wikimultihopqa":
        docs = dedup(twowiki_pairs(os.path.join(a.raw, "2wikimultihopqa")), "2wiki")
    else:
        docs = dedup(musique_pairs(os.path.join(a.raw, "musique")), "musique")
    write_docs(docs, a.out, a.dataset)
