"""Write one versioned grounding score per native cache node, without gold labels.

python -m verifier.score_nodes --tree TREE --passages PASSAGES --out SCORES
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from common.node_adapter import context_passages
from verifier.nli import DEFAULT_MODEL, NLIBackend


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    with open(path) as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}, line {number}") from exc


def inference_input(node, passage_lookup, context_mode=None):
    passages = context_passages(node, passage_lookup, context_mode=context_mode)
    return {"node_id": node["node_id"], "qid": node["qid"],
            "question": node["question"], "answer": node["answer_call"]["text"],
            "passages": [{"docid": p.docid, "text": p.text} for p in passages]}, passages


def score_node(node, passage_lookup, backend, context_mode=None):
    inputs, passages = inference_input(node, passage_lookup, context_mode)
    row = {"schema_version": 1, "node_id": inputs["node_id"], "qid": inputs["qid"],
           "input_hash": fingerprint(inputs), "answer_hash": fingerprint(inputs["answer"]),
           "verifier": backend.config, "context_docids": [p.docid for p in passages],
           "has_evidence": bool(passages), "grounding_score": None,
           "status": "error", "cost": {"verifier_calls": 0, "input_tokens": 0,
                                         "output_tokens": 0, "latency_ms": 0.0}}
    if not inputs["answer"].strip():
        row["status"] = "empty_answer"
        return row
    if not passages:
        row.update(grounding_score=0.0, status="no_evidence")
        return row
    # Model/runtime errors fail the command. They must not masquerade as score 0.
    result = backend.score(inputs["question"], inputs["answer"], passages)
    value = result["grounding_score"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid grounding score")
    row.update(result, status="ok")
    return row


def run(tree, passages_path, output, backend, *, context_mode=None, limit=None, resume=False):
    output = Path(output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.resolve() in (Path(tree).resolve(), Path(passages_path).resolve()):
        raise ValueError("output must not overwrite an input file")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if output.exists() and not resume:
        raise FileExistsError("output exists; use --resume to validate and reuse it")
    passage_lookup = {}
    for passage in read_jsonl(passages_path):
        if passage["docid"] in passage_lookup:
            raise ValueError("duplicate passage ID")
        passage_lookup[passage["docid"]] = passage
    existing = {}
    if resume:
        for path in (output, partial):
            if path.exists():
                for row in read_jsonl(path):
                    nid = row["node_id"]
                    if nid in existing and existing[nid] != row:
                        raise ValueError("conflicting checkpoint rows")
                    existing[nid] = row
    elif partial.exists():
        raise FileExistsError("partial output exists; use --resume or a new output path")
    # Validate resume inputs before touching a good checkpoint.
    selected = set()
    for index, node in enumerate(read_jsonl(tree)):
        if limit is not None and index >= limit:
            break
        nid = node["node_id"]
        if nid in selected:
            raise ValueError(f"duplicate node_id: {nid}")
        selected.add(nid)
        if nid in existing:
            inputs, _ = inference_input(node, passage_lookup, context_mode)
            old = existing[nid]
            if (old.get("schema_version") != 1 or old["verifier"] != backend.config
                    or old["input_hash"] != fingerprint(inputs)):
                raise ValueError(f"stale verifier result: {nid}")
    if not selected or set(existing) - selected:
        raise ValueError("empty input or checkpoint nodes outside the selected input")
    # Read checkpoints before replacing the partial file; flush after every node.
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, count = set(), 0
    with partial.open("w") as f:
        for node in read_jsonl(tree):
            if limit is not None and count >= limit:
                break
            nid = node["node_id"]
            if nid in seen:
                raise ValueError(f"duplicate node_id: {nid}")
            seen.add(nid)
            if nid in existing:
                row = existing[nid]
                inputs, _ = inference_input(node, passage_lookup, context_mode)
                if (row.get("schema_version") != 1 or row["verifier"] != backend.config
                        or row["input_hash"] != fingerprint(inputs)):
                    raise ValueError(f"stale verifier result: {nid}")
            else:
                row = score_node(node, passage_lookup, backend, context_mode)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            count += 1
            if count % 100 == 0:
                print(f"scored {count} nodes", flush=True)
    if not count:
        raise ValueError("tree contains no nodes")
    if set(existing) - seen:
        raise ValueError("checkpoint contains nodes outside the selected input; use a new output")
    partial.replace(output)
    print(f"saved {count} node scores to {output}")
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", required=True)
    p.add_argument("--passages", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--cache-dir")
    p.add_argument("--context-mode", choices=("accumulate", "latest"))
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    backend = NLIBackend(a.model, a.revision, a.device, a.batch_size,
                         a.max_length, a.stride, a.cache_dir)
    run(a.tree, a.passages, a.out, backend, context_mode=a.context_mode,
        limit=a.limit, resume=a.resume)


if __name__ == "__main__":
    main()
