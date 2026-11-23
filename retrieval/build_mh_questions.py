"""Build {qid, question, gold, support_titles} question files for the multi-hop datasets.
  test   = Adaptive-RAG processed_data/<ds>/test_subsampled.jsonl (500; from original dev)
  dtrain = 4000, ddev = 500, sampled (seed 0) from the original DEV set (--source dev, default for
           hotpotqa / 2wikimultihopqa, same distribution as test) or TRAIN set (--source train, needed
           for musique whose dev is too small), excluding any question in test_subsampled or
           dev_500_subsampled (zero leakage).
  gold = the single reference answer, no aliases (matches Adaptive-RAG's test files); --aliases adds them.
Run from ~/Downloads/Adaptive-RAG:
  python build_mh_questions.py musique --out_dir "$MAC/outputs/retrieval"
"""
import argparse, glob, json, os, random


def norm_q(q):
    return " ".join(q.lower().split())


def ircot_gold(answers_objects):
    gold = []
    for ao in answers_objects:
        gold += [s for s in ao.get("spans", []) if s]
        if not ao.get("spans") and ao.get("number"):
            gold.append(str(ao["number"]))
    return list(dict.fromkeys(gold))


def load_processed(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        rows.append({"qid": x["question_id"], "question": x["question_text"],
                     "gold": ircot_gold(x["answers_objects"]),
                     "support_titles": sorted({c["title"] for c in x.get("contexts", [])
                                               if c.get("is_supporting")})})
    return rows


# ---------- original train sets ----------
def hotpotqa_pool(raw, split, aliases):
    import pandas as pd
    sub = "train" if split == "train" else "validation"
    files = sorted(glob.glob(os.path.join(raw, "hotpotqa", "hf", "distractor", sub, "*.parquet")))
    assert files, "hotpotqa parquet not found"
    for fn in files:
        for r in pd.read_parquet(fn).itertuples():
            yield {"qid": f"hotpotqa_{split}_{r.id}", "question": r.question, "gold": [r.answer],
                   "support_titles": sorted(set(r.supporting_facts["title"]))}


def load_2wiki_aliases(raw):
    path = os.path.join(raw, "2wikimultihopqa", "id_aliases.json")
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8").read().strip()
    out = {}
    try:
        data = json.loads(text)
        items = data.items() if isinstance(data, dict) else []
        for k, v in items:
            out[k] = v if isinstance(v, list) else v.get("aliases", [])
    except json.JSONDecodeError:            # JSON lines
        for line in text.splitlines():
            d = json.loads(line)
            k = d.get("Q_id") or d.get("id")
            out[k] = d.get("aliases", [])
    return out


def twowiki_pool(raw, split, aliases):
    al = load_2wiki_aliases(raw) if aliases else {}
    name = "train.json" if split == "train" else "dev.json"
    for x in json.load(open(os.path.join(raw, "2wikimultihopqa", name), encoding="utf-8")):
        gold = [x["answer"]] + al.get(x.get("answer_id", ""), [])
        yield {"qid": f"2wiki_{split}_{x['_id']}", "question": x["question"],
               "gold": list(dict.fromkeys(g for g in gold if g)),
               "support_titles": sorted({t for t, _ in x["supporting_facts"]})}


def musique_pool(raw, split, aliases):
    name = "musique_ans_v1.0_train.jsonl" if split == "train" else "musique_ans_v1.0_dev.jsonl"
    for line in open(os.path.join(raw, "musique", name), encoding="utf-8"):
        if not line.strip():
            continue
        x = json.loads(line)
        if not x.get("answerable", True):
            continue
        gold = [x["answer"]] + (x.get("answer_aliases", []) if aliases else [])
        yield {"qid": f"musique_{split}_{x['id']}", "question": x["question"],
               "gold": list(dict.fromkeys(g for g in gold if g)),
               "support_titles": sorted({p["title"] for p in x["paragraphs"] if p.get("is_supporting")})}


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path}  ({len(rows)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["hotpotqa", "2wikimultihopqa", "musique"])
    ap.add_argument("--raw", default="raw_data")
    ap.add_argument("--processed", default="processed_data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["train", "dev"], default=None,
                    help="where dtrain/ddev come from (default: dev for hotpotqa/2wiki, train for musique)")
    ap.add_argument("--aliases", action="store_true", help="add answer aliases to gold (default: off)")
    a = ap.parse_args()
    a.source = a.source or ("train" if a.dataset == "musique" else "dev")
    print(f"dtrain/ddev source: original {a.source} set   aliases: {a.aliases}")
    short = {"hotpotqa": "hotpotqa", "2wikimultihopqa": "2wiki", "musique": "musique"}[a.dataset]

    test = load_processed(os.path.join(a.processed, a.dataset, "test_subsampled.jsonl"))
    dev500 = load_processed(os.path.join(a.processed, a.dataset, "dev_500_subsampled.jsonl"))
    print(f"test: {len(test)}  sample: {test[0]}")
    banned = {norm_q(r["question"]) for r in test + dev500}

    pool = {"hotpotqa": hotpotqa_pool, "2wikimultihopqa": twowiki_pool,
            "musique": musique_pool}[a.dataset](a.raw, a.source, a.aliases)
    pool = [r for r in pool if r["gold"] and norm_q(r["question"]) not in banned]
    print(f"{a.source} pool after exclusion: {len(pool)}")
    assert len(pool) >= a.n_train + a.n_dev, f"pool too small ({len(pool)}) for {a.n_train}+{a.n_dev}"
    random.Random(a.seed).shuffle(pool)
    dtrain, ddev = pool[:a.n_train], pool[a.n_train:a.n_train + a.n_dev]
    assert not ({norm_q(r["question"]) for r in dtrain} & {norm_q(r["question"]) for r in ddev})

    os.makedirs(a.out_dir, exist_ok=True)
    write(test, f"{a.out_dir}/{short}_test_questions.jsonl")
    write(ddev, f"{a.out_dir}/{short}_ddev_questions.jsonl")
    write(dtrain, f"{a.out_dir}/{short}_dtrain_questions.jsonl")
