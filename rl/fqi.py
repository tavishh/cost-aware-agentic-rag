"""Fitted Q-iteration (batch RL) over the cached trees.

Like the oracle, it works backwards from the deepest nodes, but it never looks at
the gold answer when acting: at each depth it *learns* Q(features, action) by
regression, then uses those predictions as the value of children one level up.

    depth 3:  learn Q from ANSWER outcomes (+1 / -lam); ABSTAIN = 0
    depth 2:  RETRIEVE/REWRITE target = step cost + max_a Q3(child)
    ...       down to the root

Every transition is known in advance, so there is no exploration and no moving
target: four supervised fits instead of a long, noisy DQN run.

Usage (from the repo root):
    python3 -m rl.fqi --train <train file> --dev <dev file> --lam 0.5
"""
import argparse

import torch

from rl.dqn import QNet, featurize
from rl.env import ACTIONS, TreeEnv, inacc
from rl.features import FEATURE_NAMES, Scaler, node_features
from rl.oracle import run_policy
from rl.tree_data import child_id, load_trees, split_roots

ANSWER, RETRIEVE, REWRITE, ABSTAIN = range(4)     # column indices, in ACTIONS order


def q_values(net, X, M):
    """Predicted Q for a batch; ABSTAIN is exactly 0 by definition; invalid actions -inf."""
    q = net(X).clone()
    q[:, ABSTAIN] = 0.0
    return q.masked_fill(~M, float("-inf"))


def fit_depth(X, M, Y, args):
    """Regress Y (targets per action) on X, using only valid, learnable actions."""
    net = QNet(X.shape[1], len(ACTIONS), args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    W = M.float().clone()
    W[:, ABSTAIN] = 0.0                            # ABSTAIN is fixed at 0, not learned
    for _ in range(args.epochs):
        loss = (((net(X) - Y) ** 2) * W).sum() / W.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def fqi(nodes, roots, idx, X, M, args):
    """Train one Q-network per depth (3, 2, 1, 0) on the given questions."""
    by_depth = {d: [] for d in range(4)}
    for node_id, n in nodes.items():
        if n["qid"] in roots:
            by_depth[n["step"]].append(node_id)

    step_cost = {RETRIEVE: -args.c_ret - args.c_llm,
                 REWRITE: -args.c_ret - 2 * args.c_llm}
    value, nets = {}, {}
    for d in (3, 2, 1, 0):
        ids = by_depth[d]
        rows = torch.tensor([idx[i] for i in ids])
        Xd, Md = X[rows], M[rows]
        Y = torch.zeros(len(ids), len(ACTIONS))
        for r, node_id in enumerate(ids):
            gold = nodes[roots[nodes[node_id]["qid"]]]["gold"]
            correct = inacc(nodes[node_id]["answer_call"]["text"], gold)
            Y[r, ANSWER] = 1.0 if correct else -args.lam
            for col, tag in ((RETRIEVE, "RETRIEVE"), (REWRITE, "REWRITE")):
                if Md[r, col]:
                    Y[r, col] = step_cost[col] + value[child_id(node_id, tag)]
        nets[d] = fit_depth(Xd, Md, Y, args)
        with torch.no_grad():
            v = q_values(nets[d], Xd, Md).max(1).values
        value.update({node_id: float(v[r]) for r, node_id in enumerate(ids)})
    return nets


def chooser(nets, nodes, idx, X, M):
    """Greedy policy: use the Q-network for the node's depth."""
    def choose(node_id):
        k = idx[node_id]
        with torch.no_grad():
            q = q_values(nets[nodes[node_id]["step"]], X[k:k + 1], M[k:k + 1])
        return ACTIONS[int(q.argmax())]
    return choose


def main(args):
    torch.manual_seed(args.seed)
    tr_nodes, all_roots = load_trees(args.train)
    tr_roots, val_roots = split_roots(all_roots, n_val=args.n_val)
    dv_nodes, dv_roots = load_trees(args.dev)
    scaler = Scaler().fit([node_features(tr_nodes, i) for i, n in tr_nodes.items()
                           if n["qid"] in tr_roots])
    tr_idx, tr_X, tr_M = featurize(tr_nodes, scaler)
    dv_idx, dv_X, dv_M = featurize(dv_nodes, scaler)

    cost = dict(lam=args.lam, c_ret=args.c_ret, c_llm=args.c_llm)
    val_env = TreeEnv(tr_nodes, val_roots, **cost)
    dev_env = TreeEnv(dv_nodes, dv_roots, **cost)

    # Choose the number of training epochs on validation, then report on dev.
    best = None
    for epochs in [int(e) for e in args.epochs_grid.split(",")]:
        args.epochs = epochs
        nets = fqi(tr_nodes, tr_roots, tr_idx, tr_X, tr_M, args)
        m = run_policy(val_env, chooser(nets, tr_nodes, tr_idx, tr_X, tr_M))
        print(f"epochs {epochs:5d}  val reward {m['reward']:+.3f}  InAcc {m['correct']:.3f}  "
              f"abstain {m['abstained']:.3f}  sel.acc {m['selective_acc']:.3f}  "
              f"RC {m['retrieval_calls']:.2f}  LMC {m['llm_calls']:.2f}")
        if best is None or m["reward"] > best[0]:
            best = (m["reward"], epochs, nets)

    _, epochs, nets = best
    d = run_policy(dev_env, chooser(nets, dv_nodes, dv_idx, dv_X, dv_M))
    print(f"\nFQI (epochs={epochs}, chosen on val) on dev:",
          {k: round(v, 3) for k, v in d.items()})

    if args.save:
        torch.save({"nets": {k: n.state_dict() for k, n in nets.items()},
                    "mean": scaler.mean, "std": scaler.std,
                    "features": FEATURE_NAMES, "args": vars(args)}, args.save)
        print("saved:", args.save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--c_ret", type=float, default=0.05)
    p.add_argument("--c_llm", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--epochs_grid", default="100,300,1000")
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    main(p.parse_args())
