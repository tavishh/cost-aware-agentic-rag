"""Baselines and a fair comparison of all policies on dev.

- fixed paths: every R/W sequence of length 0-3 followed by ANSWER (best chosen on val)
- confidence threshold: answer if first-token prob >= t, else retrieve again (up to k
  steps); at the last step answer if prob >= t_last, else abstain (t, t_last, k tuned on val)
- saved DQN / FQI checkpoints
Each policy is compared with retrieve-once using a paired bootstrap on dev questions.

Usage (from the repo root):
    python3 -m rl.baselines --train <train file> --dev <dev file> --lam 0.5 \
        --dqn outputs/rl/ddqn_val_lam0.5_s0.pt ... --fqi outputs/rl/fqi_lam0.5.pt
"""
import argparse
import itertools
import random

import torch

from common.schemas import Action
from rl.dqn import QNet, featurize, greedy_chooser
from rl.env import ACTIONS, TreeEnv, valid_actions
from rl.features import FEATURE_NAMES, Scaler, node_features
from rl.fqi import chooser as fqi_chooser
from rl.oracle import solve
from rl.tree_data import load_trees, split_roots

PROB = FEATURE_NAMES.index("first_token_prob")


def play(env, choose):
    """Per-question results: list of (reward, correct, abstained, rc, lmc)."""
    out = []
    for qid in env.roots:
        node_id, done, total = env.reset(qid), False, 0.0
        while not done:
            node_id, r, done, info = env.step(choose(node_id))
            total += r
        out.append((total, info["correct"] or 0, info["abstained"],
                    info["retrieval_calls"], info["llm_calls"]))
    return out


def summary(rows):
    n = len(rows)
    answered = n - sum(r[2] for r in rows)
    return {"reward": sum(r[0] for r in rows) / n,
            "InAcc": sum(r[1] for r in rows) / n,
            "abstain": sum(r[2] for r in rows) / n,
            "sel_acc": sum(r[1] for r in rows) / answered if answered else 0.0,
            "RC": sum(r[3] for r in rows) / n,
            "LMC": sum(r[4] for r in rows) / n}


def paired_ci(a, b, n_boot=2000, seed=0):
    """95% CI of mean(a - b) over the same questions, by resampling questions."""
    rng = random.Random(seed)
    diffs = [x[0] - y[0] for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def fixed_path(plan):
    """Follow the plan of R/W steps, then ANSWER."""
    def choose(node_id):
        depth = node_id.count("/")
        if depth < len(plan):
            a = Action.RETRIEVE if plan[depth] == "R" else Action.REWRITE
            return a if a in valid_actions_cache[node_id] else Action.ANSWER
        return Action.ANSWER
    return choose


def threshold(nodes, t, t_last, k):
    """Answer when confident, otherwise retrieve again; abstain if still unsure at step k."""
    def choose(node_id):
        p = feats_cache[node_id][PROB]
        step = nodes[node_id]["step"]
        can_go = Action.RETRIEVE in valid_actions_cache[node_id]
        if p >= t:
            return Action.ANSWER
        if step < k and can_go:
            return Action.RETRIEVE
        return Action.ANSWER if p >= t_last else Action.ABSTAIN
    return choose


def build_caches(*node_dicts):
    """Raw features and valid actions for every node the rule-based policies may visit."""
    global feats_cache, valid_actions_cache
    feats_cache, valid_actions_cache = {}, {}
    for nodes in node_dicts:
        for i in nodes:
            feats_cache[i] = node_features(nodes, i)
            valid_actions_cache[i] = valid_actions(nodes, i)


def tune_best_fixed(val_env):
    """Best R/W plan (length 0-3, then ANSWER) on validation."""
    plans = [""] + ["".join(p) for k in (1, 2, 3) for p in itertools.product("RW", repeat=k)]
    return max(plans, key=lambda p: summary(play(val_env, fixed_path(p)))["reward"])


def tune_threshold(val_env, val_nodes):
    """(t, t_last, k) for the confidence-threshold rule, chosen on validation."""
    grid = [i / 20 for i in range(21)]
    return max(((t, tl, k) for t in grid for tl in grid if tl <= t for k in (0, 1, 2, 3)),
               key=lambda g: summary(play(val_env, threshold(val_nodes, *g)))["reward"])


def load_scaler(ckpt):
    s = Scaler()
    s.mean, s.std = ckpt["mean"], ckpt["std"]
    return s


def main(args):
    tr_nodes, all_roots = load_trees(args.train)
    _, val_roots = split_roots(all_roots, n_val=args.n_val)
    dv_nodes, dv_roots = load_trees(args.dev)
    cost = dict(lam=args.lam, c_ret=args.c_ret, c_llm=args.c_llm)
    val_env = TreeEnv(tr_nodes, val_roots, **cost)
    dev_env = TreeEnv(dv_nodes, dv_roots, **cost)

    build_caches(tr_nodes, dv_nodes)

    policies = {}
    once = play(dev_env, fixed_path("R"))
    policies["retrieve once"] = once
    policies["answer at root"] = play(dev_env, fixed_path(""))
    policies["R/R/R"] = play(dev_env, fixed_path("RRR"))

    # Best fixed path and confidence threshold, both chosen on validation.
    best_plan = tune_best_fixed(val_env)
    policies[f"best fixed ({best_plan or 'root'})"] = play(dev_env, fixed_path(best_plan))
    best = tune_threshold(val_env, tr_nodes)
    policies[f"threshold (t={best[0]}, t_last={best[1]}, k={best[2]})"] = \
        play(dev_env, threshold(dv_nodes, *best))

    # Saved learned policies.
    for path in args.dqn or []:
        ck = torch.load(path, weights_only=False)
        idx, X, M = featurize(dv_nodes, load_scaler(ck))
        net = QNet(len(FEATURE_NAMES), len(ACTIONS))
        net.load_state_dict(ck["state"])
        policies[f"DQN {path.split('/')[-1]}"] = play(dev_env, greedy_chooser(net, idx, X, M))
    if args.fqi:
        ck = torch.load(args.fqi, weights_only=False)
        idx, X, M = featurize(dv_nodes, load_scaler(ck))
        nets = {}
        for d, state in ck["nets"].items():
            nets[int(d)] = QNet(len(FEATURE_NAMES), len(ACTIONS), ck["args"]["hidden"])
            nets[int(d)].load_state_dict(state)
        policies["FQI"] = play(dev_env, fqi_chooser(nets, dv_nodes, idx, X, M))

    _, best_act = solve(dev_env)
    policies["oracle"] = play(dev_env, lambda n: best_act[n])

    print(f"dev, lambda={args.lam}, c_ret={args.c_ret}, c_llm={args.c_llm}")
    thr = next(rows for name, rows in policies.items() if name.startswith("threshold"))
    print(f"{'policy':44s} {'reward':>7s} {'InAcc':>6s} {'abst':>5s} {'selAcc':>6s} "
          f"{'RC':>5s} {'LMC':>5s}   vs retrieve-once [95% CI]    vs threshold [95% CI]")
    for name, rows in policies.items():
        s = summary(rows)
        d, lo, hi = paired_ci(rows, once)
        d2, lo2, hi2 = paired_ci(rows, thr)
        print(f"{name:44s} {s['reward']:+7.3f} {s['InAcc']:6.3f} {s['abstain']:5.3f} "
              f"{s['sel_acc']:6.3f} {s['RC']:5.2f} {s['LMC']:5.2f}   "
              f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]   {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--c_ret", type=float, default=0.05)
    p.add_argument("--c_llm", type=float, default=0.02)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--dqn", nargs="*")
    p.add_argument("--fqi", default=None)
    main(p.parse_args())
