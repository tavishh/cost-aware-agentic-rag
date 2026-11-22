"""Oracle: the best achievable policy, computed exactly from the cached trees.

At every node, the oracle compares
    ANSWER   -> +1 if the cached answer is correct, else -lam
    ABSTAIN  -> 0
    RETRIEVE -> step cost + value of the /R child
    REWRITE  -> step cost + value of the /W child
and keeps the best. Working backwards from the deepest nodes gives the optimal
value of every node (backward induction). It uses the gold answer, so it is an
upper bound for any real policy, not something a policy can achieve.
"""
from common.schemas import Action
from rl.env import TreeEnv, inacc, valid_actions
from rl.tree_data import child_id


def action_value(env, node_id, action, value):
    """Immediate reward of `action` at node_id plus the value of where it leads."""
    if action is Action.ANSWER:
        gold = env.nodes[env.roots[env.nodes[node_id]["qid"]]]["gold"]
        correct = inacc(env.nodes[node_id]["answer_call"]["text"], gold, env.strict)
        return 1.0 if correct else -env.lam
    if action is Action.ABSTAIN:
        return 0.0
    cost = -env.c_ret - env.c_llm - (env.c_llm if action is Action.REWRITE else 0.0)
    return cost + value[child_id(node_id, action.value)]


def solve(env):
    """Return (value, best) for every node: value[node_id] and best[node_id] = best action."""
    value, best = {}, {}
    # Deepest nodes first, so children are always solved before their parents.
    for node_id in sorted(env.nodes, key=lambda n: -env.nodes[n]["step"]):
        options = {a: action_value(env, node_id, a, value)
                   for a in valid_actions(env.nodes, node_id, env.allow_abstain)}
        best[node_id] = max(options, key=options.get)
        value[node_id] = options[best[node_id]]
    return value, best


def run_policy(env, choose):
    """Play every question with choose(node_id) -> Action and average the metrics."""
    totals = {"reward": 0.0, "correct": 0, "abstained": 0,
              "retrieval_calls": 0, "llm_calls": 0}
    for qid in env.roots:
        node_id, done = env.reset(qid), False
        while not done:
            node_id, r, done, info = env.step(choose(node_id))
            totals["reward"] += r
        totals["correct"] += info["correct"] or 0
        totals["abstained"] += info["abstained"]
        totals["retrieval_calls"] += info["retrieval_calls"]
        totals["llm_calls"] += info["llm_calls"]
    n = len(env.roots)
    out = {k: v / n for k, v in totals.items()}
    answered = n - totals["abstained"]
    # Selective accuracy: correct among the questions the policy chose to answer.
    out["selective_acc"] = totals["correct"] / answered if answered else 0.0
    return out


def accuracy_oracle(env):
    """Fraction of questions with at least one correct node anywhere in the tree (ignores cost)."""
    hit = {q: 0 for q in env.roots}
    for node_id, n in env.nodes.items():
        gold = env.nodes[env.roots[n["qid"]]]["gold"]
        hit[n["qid"]] |= inacc(n["answer_call"]["text"], gold, env.strict)
    return sum(hit.values()) / len(hit)


if __name__ == "__main__":
    import sys
    from rl.tree_data import load_trees

    nodes, roots = load_trees(sys.argv[1])
    for lam in (1.0, 0.5):
        env = TreeEnv(nodes, roots, lam=lam)
        _, best = solve(env)
        print(f"\n=== lambda = {lam} (c_ret={env.c_ret}, c_llm={env.c_llm}) ===")
        print(f"accuracy oracle (best node anywhere, no cost): {accuracy_oracle(env):.3f}")
        for name, choose in [("always abstain", lambda n: Action.ABSTAIN),
                             ("answer at root", lambda n: Action.ANSWER),
                             ("cost-aware oracle", lambda n: best[n])]:
            m = run_policy(env, choose)
            print(f"{name:18s} reward {m['reward']:+.3f}  InAcc {m['correct']:.3f}  "
                  f"abstain {m['abstained']:.3f}  sel.acc {m['selective_acc']:.3f}  "
                  f"RC {m['retrieval_calls']:.2f}  LMC {m['llm_calls']:.2f}")
