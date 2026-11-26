"""Strict score joins and opt-in RL features; existing 17-feature runs are unchanged."""
import math
from verifier.score_nodes import fingerprint, read_jsonl

GROUNDING_FEATURE_NAMES = ["grounding_score", "grounding_available", "grounding_has_evidence"]


def load_scores(path, nodes, *, require_all=True):
    scores = {}
    config = None
    for row in read_jsonl(path):
        nid = row["node_id"]
        if nid in scores:
            raise ValueError(f"duplicate verifier node: {nid}")
        if nid not in nodes:
            raise ValueError(f"unknown verifier node: {nid}")
        if row.get("schema_version") != 1:
            raise ValueError("unsupported verifier schema")
        if row["answer_hash"] != fingerprint(nodes[nid]["answer_call"]["text"]):
            raise ValueError(f"stale answer score: {nid}")
        if "context_docids" in nodes[nid] and row["context_docids"] != nodes[nid]["context_docids"]:
            raise ValueError(f"wrong retriever/context score: {nid}")
        if config is not None and config != row["verifier"]:
            raise ValueError("mixed verifier configurations")
        config = row["verifier"]
        value = row["grounding_score"]
        status = row["status"]
        if status not in ("ok", "no_evidence", "empty_answer"):
            raise ValueError(f"unusable verifier result: {nid}")
        if status == "empty_answer":
            if value is not None:
                raise ValueError("empty answer must have null score")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("grounding score must be finite and in [0, 1]")
        if not isinstance(row["has_evidence"], bool):
            raise ValueError("has_evidence must be boolean")
        if status == "no_evidence" and (value != 0 or row["has_evidence"]):
            raise ValueError("inconsistent no-evidence result")
        scores[nid] = row
    if require_all and set(nodes) != set(scores):
        raise ValueError(f"missing scores for {len(set(nodes)-set(scores))} nodes")
    return scores


def grounding_features(scores, node_id):
    row = scores.get(node_id)
    if row is None:
        return [0.0, 0.0, 0.0]
    value = row["grounding_score"]
    return [float(value) if value is not None else 0.0,
            float(value is not None), float(row["has_evidence"])]


def grounded_node_features(nodes, node_id, scores):
    from rl.features import node_features
    return node_features(nodes, node_id) + grounding_features(scores, node_id)
