"""Adapt Chester's native nodes without leaking labels or inventing a new cache."""
import math
from common.schemas import (
    Action, Cost, GenerationResult, Observation, Passage, UncertaintyResult,
)


def call_cost(call, *, verifier=False):
    return Cost(
        generation_calls=0 if verifier else 1,
        verifier_calls=1 if verifier else 0,
        input_tokens=call["tokens_in"], output_tokens=call["tokens_out"],
        latency_ms=call["latency_s"] * 1000,
    )


def ancestor_path(nodes, node_id):
    path, seen = [], set()
    qid = nodes[node_id]["qid"]
    while node_id is not None:
        if node_id in seen:
            raise ValueError("cycle in parent links")
        seen.add(node_id)
        node = nodes[node_id]  # fail on missing ancestors, never silently undercharge
        if node["qid"] != qid:
            raise ValueError("parent crosses question boundary")
        path.append(node)
        node_id = node["parent_id"]
    return list(reversed(path))


def cumulative_cost(nodes, node_id):
    path = ancestor_path(nodes, node_id)
    calls = [n["answer_call"] for n in path]
    calls += [n["rewrite_call"] for n in path if n.get("action") == "rewrite"]
    return Cost(
        retrieval_calls=len(path) - 1, generation_calls=len(calls),
        input_tokens=sum(c["tokens_in"] for c in calls),
        output_tokens=sum(c["tokens_out"] for c in calls),
        # Generation time only: current cache has no retrieval timing.
        latency_ms=sum(c["latency_s"] for c in calls) * 1000,
    )


def context_passages(node, passages, *, context_mode=None):
    """Use the exact generator context, preserving order and repeated passages."""
    if "context_docids" in node:
        ids = node["context_docids"]
    elif not node["retrievals"]:
        ids = []
    elif context_mode == "accumulate":
        ids = [d for rt in node["retrievals"] for d in rt["docids"]]
    elif context_mode == "latest":
        ids = node["retrievals"][-1]["docids"]
    else:
        raise ValueError("context_docids missing: specify accumulate/latest from cache metadata")
    scores = {d: s for rt in node["retrievals"]
              for d, s in zip(rt["docids"], rt["scores"])}
    result = []
    for docid in ids:
        passage = passages[docid]
        title = passage.get("title", "")
        text = f"Wikipedia Title: {title}\n{passage['text']}" if title else passage["text"]
        result.append(Passage(docid, text, scores.get(docid)))
    return tuple(result)


def greedy_uncertainty(call, *, greedy):
    if not greedy:
        return UncertaintyResult("greedy_top1_mean_nll_v1", None, "greedy decoding not confirmed")
    positions = call.get("token_logprobs")
    if not positions or any(not p for p in positions):
        return UncertaintyResult("greedy_top1_mean_nll_v1", None, "missing candidate log probabilities")
    values = [max(float(lp) for _, lp in pos) for pos in positions]
    if any(not math.isfinite(x) or x > 0 for x in values):
        raise ValueError("invalid candidate log probabilities")
    # This proxy includes all cached positions, not only cleaned answer tokens.
    return UncertaintyResult("greedy_top1_mean_nll_v1", -sum(values) / len(values))


def node_to_observation(nodes, node_id, passages, *, model_version, prompt_version,
                        greedy=False, context_mode=None, max_steps=3):
    node = nodes[node_id]
    path = ancestor_path(nodes, node_id)
    history = tuple({"retrieve": Action.RETRIEVE, "rewrite": Action.REWRITE}[n["action"]]
                    for n in path[1:])
    if len(history) > max_steps:
        raise ValueError("node exceeds configured budget")
    return Observation(
        qid=node["qid"], question=node["question"],
        query=node.get("current_query") or (node["retrievals"][-1]["query"]
                                          if node["retrievals"] else node["question"]),
        passages=context_passages(node, passages, context_mode=context_mode),
        generation=GenerationResult(node["answer_call"]["text"], model_version,
                                    prompt_version, cost=call_cost(node["answer_call"])),
        uncertainty=greedy_uncertainty(node["answer_call"], greedy=greedy),
        action_history=history, remaining_steps=max_steps-len(history),
        cumulative_cost=cumulative_cost(nodes, node_id),
    )
