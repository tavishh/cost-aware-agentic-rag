"""Prepare a score-blind development audit and compare frozen support annotations.

No gold answers or cached correctness metrics participate in selection or review.
This is an exploratory audit, not a substitute for independent human annotation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from verifier.score_nodes import fingerprint, read_jsonl


def unique(rows, key):
    result = {}
    for row in rows:
        name = row[key]
        if name in result:
            raise ValueError(f"duplicate {key}: {name}")
        result[name] = row
    return result


def prepare(tree_path, passages_path, out_dir, seed=20261122, per_depth=6, exclude_qids=()):
    if per_depth < 1:
        raise ValueError("per_depth must be positive")
    directory = Path(out_dir)
    if directory.exists():
        raise FileExistsError("use a new audit directory to preserve frozen inputs")
    nodes = list(unique(read_jsonl(tree_path), "node_id").values())
    passages = unique(read_jsonl(passages_path), "docid")
    rng, used, selected = random.Random(seed), set(exclude_qids), []
    for depth in (1, 2, 3):
        pool = {}
        for node in nodes:
            if node["step"] == depth and node["qid"] not in used:
                pool.setdefault(node["qid"], []).append(node)
        for qid in rng.sample(sorted(pool), per_depth):
            node = rng.choice(sorted(pool[qid], key=lambda n: n["node_id"]))
            selected.append(node)
            used.add(qid)
    directory.mkdir(parents=True)
    with (directory/"review_inputs.jsonl").open("w") as review, (directory/"nodes.jsonl").open("w") as raw:
        for i, node in enumerate(selected, 1):
            # Whitelist generator inputs and structural fields. Drop gold-derived data recursively.
            clean = {k: node[k] for k in ("node_id", "qid", "parent_id", "step", "action", "question",
                                         "context_docids", "answer_call")}
            clean["retrievals"] = [{k: rt[k] for k in ("query", "docids", "scores")}
                                   for rt in node["retrievals"]]
            raw.write(json.dumps(clean)+"\n")
            row = {"case": i, "node_id": node["node_id"], "qid": node["qid"], "depth": node["step"],
                   "question": node["question"], "answer": node["answer_call"]["text"],
                   "passages": [passages[d] for d in node["context_docids"]]}
            review.write(json.dumps(row)+"\n")
    manifest = {"seed": seed, "source_sha256": hashlib.sha256(Path(tree_path).read_bytes()).hexdigest(),
                "selection": "distinct questions per depth; no score/gold filtering",
                "per_depth": per_depth, "cases": len(selected), "excluded_qids": sorted(exclude_qids)}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def evaluate(inputs_path, annotations_path, scores_path, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    inputs = unique(read_jsonl(inputs_path), "node_id")
    annotations = unique(read_jsonl(annotations_path), "node_id")
    scores = unique(read_jsonl(scores_path), "node_id")
    if set(inputs) != set(annotations) or set(inputs) != set(scores):
        raise ValueError("inputs, annotations and scores must have identical node IDs")
    paired, unclear, unavailable, rows = [], [], [], []
    config = None
    for nid, item in inputs.items():
        annotation, score = annotations[nid], scores[nid]
        label = annotation["label"]
        if label not in ("supported", "unsupported", "unclear"):
            raise ValueError("invalid annotation label")
        ids = {p["docid"] for p in item["passages"]}
        if not set(annotation["evidence_ids"]).issubset(ids):
            raise ValueError("annotation cites evidence outside the node")
        if label == "supported" and not annotation["evidence_ids"]:
            raise ValueError("supported annotations need evidence citations")
        actual_input = {k: item[k] for k in ("node_id", "qid", "question", "answer")}
        actual_input["passages"] = [{"docid": p["docid"],
                                    "text": (f"Wikipedia Title: {p['title']}\n{p['text']}"
                                             if p.get("title") else p["text"])} for p in item["passages"]]
        if fingerprint(actual_input) != score["input_hash"]:
            raise ValueError("stale score input")
        value = score["grounding_score"]
        if score["status"] == "invalid_claim":
            if value is not None or not score.get("claim_issues"):
                raise ValueError("invalid claim requires a null score and explicit issues")
        elif score["status"] != "ok" or isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("audit requires finite scores or explicit invalid claims")
        if config is not None and config != score["verifier"]:
            raise ValueError("mixed verifier methods")
        config = score["verifier"]
        rows.append({"node_id": nid, "depth": item["depth"], "label": label,
                     "grounding_score": value, "predicted_supported": value >= threshold if value is not None else None})
        if label == "unclear":
            unclear.append(nid)
        elif value is None:
            unavailable.append(nid)
        else:
            paired.append((value, int(label == "supported")))
    positives = [v for v, y in paired if y]
    negatives = [v for v, y in paired if not y]
    tp = sum(v >= threshold and y for v, y in paired)
    fp = sum(v >= threshold and not y for v, y in paired)
    fn = len(positives)-tp
    tn = len(negatives)-fp
    auc = (sum((p > n) + .5*(p == n) for p in positives for n in negatives)
           / (len(positives)*len(negatives))) if positives and negatives else None
    return {"cases": len(inputs), "labels": dict(Counter(a["label"] for a in annotations.values())),
            "excluded_unclear": unclear, "threshold": threshold, "roc_auc": auc,
            "unavailable": unavailable,
            "score_coverage": len(paired)/(len(inputs)-len(unclear)) if len(inputs)>len(unclear) else None,
            "metric_scope": "available scores on non-unclear labels; inspect coverage separately",
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "accuracy": (tp+tn)/len(paired) if paired else None,
            "verifier": config, "rows": rows,
            "annotation_provenance": sorted({a.get("annotator", "unspecified") for a in annotations.values()}),
            "human_validated": all(a.get("human_validated") is True for a in annotations.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--tree", required=True)
    prep.add_argument("--passages", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--seed", type=int, default=20261122)
    prep.add_argument("--per-depth", type=int, default=6)
    prep.add_argument("--exclude-inputs", help="prior review JSONL; exclude those questions")
    compare = sub.add_parser("evaluate")
    compare.add_argument("--inputs", required=True)
    compare.add_argument("--annotations", required=True)
    compare.add_argument("--scores", required=True)
    compare.add_argument("--threshold", type=float, default=.5)
    a = p.parse_args()
    if a.command == "prepare":
        excluded = {r["qid"] for r in read_jsonl(a.exclude_inputs)} if a.exclude_inputs else set()
        result = prepare(a.tree, a.passages, a.out, a.seed, a.per_depth, excluded)
    else:
        result = evaluate(a.inputs, a.annotations, a.scores, a.threshold)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
