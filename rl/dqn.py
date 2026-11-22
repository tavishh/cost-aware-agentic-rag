"""DQN over the cached tree environment.

Usage (from the repo root):
    python3 -m rl.dqn --train <train file> --dev <dev file> --lam 0.5

Every transition can be replayed from the cache, so training only needs a CPU.
"""
import argparse
import random
import time

import torch
import torch.nn as nn

from rl.env import ACTIONS, TreeEnv, action_mask
from rl.features import FEATURE_NAMES, Scaler, node_features
from rl.oracle import run_policy, solve
from rl.tree_data import load_trees


class QNet(nn.Module):
    """Small MLP: features -> one Q-value per action."""

    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def featurize(nodes, scaler):
    """Precompute scaled features and action masks for every node, once."""
    ids = list(nodes)
    X = torch.tensor([scaler.transform(node_features(nodes, i)) for i in ids],
                     dtype=torch.float32)
    M = torch.tensor([action_mask(nodes, i) for i in ids], dtype=torch.bool)
    return {i: k for k, i in enumerate(ids)}, X, M


def masked_q(q, mask):
    """Invalid actions get -inf so they are never chosen or used as targets."""
    return q.masked_fill(~mask, float("-inf"))


def greedy_chooser(net, index, X, M):
    """Policy function for run_policy: pick the best valid action at a node."""
    def choose(node_id):
        k = index[node_id]
        with torch.no_grad():
            q = masked_q(net(X[k:k + 1]), M[k:k + 1])
        return ACTIONS[int(q.argmax())]
    return choose


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tr_nodes, tr_roots = load_trees(args.train)
    dv_nodes, dv_roots = load_trees(args.dev)
    scaler = Scaler().fit([node_features(tr_nodes, i) for i in tr_nodes])
    tr_idx, tr_X, tr_M = featurize(tr_nodes, scaler)
    dv_idx, dv_X, dv_M = featurize(dv_nodes, scaler)

    cost = dict(lam=args.lam, c_ret=args.c_ret, c_llm=args.c_llm)
    env = TreeEnv(tr_nodes, tr_roots, **cost)
    dev_env = TreeEnv(dv_nodes, dv_roots, **cost)

    net = QNet(len(FEATURE_NAMES), len(ACTIONS))
    target = QNet(len(FEATURE_NAMES), len(ACTIONS))
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    buffer, pos, step = [], 0, 0          # ring buffer of transitions
    qids = list(tr_roots)
    best_dev, best_state = float("-inf"), None
    t0 = time.time()

    for ep in range(args.episodes):
        # Epsilon decays linearly from 1.0 to 0.05 over the first half of training.
        eps = max(0.05, 1.0 - ep / (0.5 * args.episodes))
        node_id, done = env.reset(random.choice(qids)), False
        while not done:
            k = tr_idx[node_id]
            valid = [a for a, ok in zip(ACTIONS, tr_M[k].tolist()) if ok]
            if random.random() < eps:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    action = ACTIONS[int(masked_q(net(tr_X[k:k + 1]), tr_M[k:k + 1]).argmax())]
            next_id, r, done, _ = env.step(action)
            transition = (k, ACTIONS.index(action), r, tr_idx[next_id], done)
            if len(buffer) < args.buffer:
                buffer.append(transition)
            else:                                   # overwrite the oldest transition
                buffer[pos] = transition
                pos = (pos + 1) % args.buffer
            node_id = next_id
            step += 1

            if len(buffer) >= args.batch:
                s, a, r_, s2, d = zip(*random.sample(buffer, args.batch))
                s, a, s2 = torch.tensor(s), torch.tensor(a), torch.tensor(s2)
                r_ = torch.tensor(r_, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)
                q = net(tr_X[s]).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    q_next = masked_q(target(tr_X[s2]), tr_M[s2]).max(1).values
                    q_next = torch.where(d.bool(), torch.zeros_like(q_next), q_next)
                    y = r_ + args.gamma * q_next
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step % args.target_every == 0:
                target.load_state_dict(net.state_dict())

        if (ep + 1) % args.eval_every == 0:
            m = run_policy(dev_env, greedy_chooser(net, dv_idx, dv_X, dv_M))
            print(f"ep {ep + 1:6d}  eps {eps:.2f}  dev reward {m['reward']:+.3f}  "
                  f"InAcc {m['correct']:.3f}  abstain {m['abstained']:.3f}  "
                  f"sel.acc {m['selective_acc']:.3f}  RC {m['retrieval_calls']:.2f}  "
                  f"LMC {m['llm_calls']:.2f}  ({time.time() - t0:.0f}s)")
            if m["reward"] > best_dev:
                best_dev = m["reward"]
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    # Final report on dev with the best checkpoint, next to the oracle.
    if best_state is None:                  # no evaluation ran: keep the final weights
        best_state = net.state_dict()
    net.load_state_dict(best_state)
    m = run_policy(dev_env, greedy_chooser(net, dv_idx, dv_X, dv_M))
    _, best = solve(dev_env)
    o = run_policy(dev_env, lambda n: best[n])
    print("\nbest DQN on dev:", {k: round(v, 3) for k, v in m.items()})
    print("oracle on dev:  ", {k: round(v, 3) for k, v in o.items()})

    if args.save:
        torch.save({"state": best_state, "mean": scaler.mean, "std": scaler.std,
                    "features": FEATURE_NAMES, "args": vars(args)}, args.save)
        print("saved:", args.save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--c_ret", type=float, default=0.05)
    p.add_argument("--c_llm", type=float, default=0.02)
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)    # short episodes, costs are explicit
    p.add_argument("--target_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    train(p.parse_args())