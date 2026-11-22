"""State features: turn a node into the numbers the DQN sees.

Only information available at inference time is used: the path so far, the draft
answer and its token log-probabilities, and the BM25 scores of retrieved passages.
Gold answers, has_answer and metrics are never touched here.
"""
import math

from rl.env import ACTIONS, action_mask

FEATURE_NAMES = [
    # progress
    "step", "n_rewrites", "can_retrieve", "can_rewrite",
    # confidence of the current draft answer (greedy token = top-1 of the stored top-20)
    "mean_nll", "max_nll", "first_token_prob", "mean_entropy", "max_entropy", "answer_len",
    # retrieval signals
    "last_max_score", "last_mean_score", "first_max_score", "score_change",
    # answer stability along the path
    "same_as_parent", "same_as_root", "distinct_answers",
]


def _norm(text):
    return " ".join(text.lower().split())


def _token_stats(token_logprobs):
    """Per-token NLL of the chosen token and entropy over the top-20 (+ leftover mass)."""
    nll, ent = [], []
    for pos in token_logprobs:
        if not pos:
            continue
        probs = [math.exp(lp) for _, lp in pos]
        rest = max(0.0, 1.0 - sum(probs))          # mass outside the stored top-20
        dist = probs + ([rest] if rest > 0 else [])
        nll.append(-pos[0][1])
        ent.append(-sum(p * math.log(p) for p in dist if p > 0))
    return nll, ent


def _path(node_id):
    """All node_ids from the root to node_id, e.g. q, q/R, q/R/W."""
    parts = node_id.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def node_features(nodes, node_id):
    """Feature vector (list of floats, in FEATURE_NAMES order) for one node."""
    node = nodes[node_id]
    path = [nodes[p] for p in _path(node_id)]
    mask = dict(zip(ACTIONS, action_mask(nodes, node_id)))

    nll, ent = _token_stats(node["answer_call"]["token_logprobs"])
    first_prob = math.exp(-nll[0]) if nll else 0.0

    rts = node["retrievals"]
    last = rts[-1]["scores"] if rts else []
    first = rts[0]["scores"] if rts else []

    answers = [_norm(n["answer_call"]["text"]) for n in path]

    feats = {
        "step": node["step"],
        "n_rewrites": sum(1 for n in path if n["action"] == "rewrite"),
        "can_retrieve": float(mask[ACTIONS[1]]),
        "can_rewrite": float(mask[ACTIONS[2]]),
        "mean_nll": sum(nll) / len(nll) if nll else 0.0,
        "max_nll": max(nll) if nll else 0.0,
        "first_token_prob": first_prob,
        "mean_entropy": sum(ent) / len(ent) if ent else 0.0,
        "max_entropy": max(ent) if ent else 0.0,
        "answer_len": node["answer_call"]["tokens_out"],
        "last_max_score": max(last) if last else 0.0,
        "last_mean_score": sum(last) / len(last) if last else 0.0,
        "first_max_score": max(first) if first else 0.0,
        "score_change": (max(last) - max(first)) if rts else 0.0,
        "same_as_parent": float(len(answers) > 1 and answers[-1] == answers[-2]),
        "same_as_root": float(len(answers) > 1 and answers[-1] == answers[0]),
        "distinct_answers": len(set(answers)),
    }
    return [float(feats[name]) for name in FEATURE_NAMES]


class Scaler:
    """Standardize features with mean/std fitted on the training split only."""

    def fit(self, rows):
        n = len(rows)
        self.mean = [sum(col) / n for col in zip(*rows)]
        self.std = [max(1e-6, math.sqrt(sum((x - m) ** 2 for x in col) / n))
                    for col, m in zip(zip(*rows), self.mean)]
        return self

    def transform(self, row):
        return [(x - m) / s for x, m, s in zip(row, self.mean, self.std)]


def auc(scores, labels):
    """ROC-AUC of scores for predicting labels (1 = correct). 0.5 = no signal."""
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):                      # average ranks for ties
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


if __name__ == "__main__":
    # Sanity check: how well does each feature alone separate correct from wrong drafts?
    import sys
    from rl.env import inacc
    from rl.tree_data import load_trees

    nodes, roots = load_trees(sys.argv[1])
    rows, labels = [], []
    for node_id, n in nodes.items():
        gold = nodes[roots[n["qid"]]]["gold"]
        rows.append(node_features(nodes, node_id))
        labels.append(inacc(n["answer_call"]["text"], gold))
    print(f"nodes: {len(rows)}  correct drafts: {sum(labels) / len(labels):.3f}")
    print(f"{'feature':18s} {'mean':>8s} {'AUC':>6s}   (AUC far from 0.5 = useful signal)")
    for k, name in enumerate(FEATURE_NAMES):
        col = [r[k] for r in rows]
        print(f"{name:18s} {sum(col) / len(col):8.3f} {auc(col, labels):6.3f}")