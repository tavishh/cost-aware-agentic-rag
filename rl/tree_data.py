"""Step 2.1: load Chester's tree15 node files into a node_id -> node lookup."""
import json
from collections import Counter

# Fields the environment and feature code need; everything else is dropped to save memory.
KEEP = ("node_id", "qid", "parent_id", "step", "action", "question", "gold",
        "retrievals", "answer_call", "rewrite_call")

CHILD_TAG = {"RETRIEVE": "R", "REWRITE": "W"}


def load_trees(path):
    """Return (nodes, roots): nodes maps node_id -> node, roots maps qid -> root node_id."""
    nodes, roots = {}, {}
    with open(path) as f:
        for line in f:
            n = json.loads(line)
            for rt in n["retrievals"]:        # passage texts live in passages_final.jsonl
                rt.pop("has_answer", None)    # gold-derived: must never reach the policy
            nodes[n["node_id"]] = {k: n[k] for k in KEEP if k in n}
            if n["parent_id"] is None:
                roots[n["qid"]] = n["node_id"]
    return nodes, roots


def child_id(node_id, action):
    """node_id of the child reached by RETRIEVE or REWRITE (may not exist in the cache)."""
    return f"{node_id}/{CHILD_TAG[action]}"


def summarize(nodes, roots):
    """Quick sanity check: question count, nodes per question, and nodes per depth."""
    depth = Counter(n["step"] for n in nodes.values())
    missing_w = sum(1 for nid, n in nodes.items()
                    if n["step"] < 3 and child_id(nid, "REWRITE") not in nodes)
    print(f"questions: {len(roots)}  nodes: {len(nodes)}  "
          f"nodes/question: {len(nodes) / len(roots):.2f}")
    print("nodes per step:", dict(sorted(depth.items())))
    print("nodes at step < 3 with no W child:", missing_w)


if __name__ == "__main__":
    import sys
    summarize(*load_trees(sys.argv[1]))