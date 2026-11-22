"""Sweep the wrong-answer penalty lambda and compare policies on dev.

For each lambda: tune the threshold rule and the best fixed path on validation,
train FQI (training length chosen on validation), and evaluate everything on dev
with paired bootstrap CIs against the threshold rule. Results go to a CSV and two
plots: coverage vs selective accuracy, and InAcc vs LLM calls.

Usage (from the repo root):
    python3 -m rl.sweep --train <train file> --dev <dev file> --out outputs/rl/sweep_bm25
"""
import argparse
import csv
import os
import statistics
from types import SimpleNamespace

import torch

from rl import baselines as B
from rl.dqn import featurize
from rl.env import TreeEnv
from rl.features import Scaler, node_features
from rl.fqi import chooser as fqi_chooser, fqi
from rl.oracle import solve
from rl.tree_data import load_trees, split_roots


def train_fqi(tr_nodes, tr_roots, val_env, tr_idx, tr_X, tr_M, cost, grid=(50, 100, 300)):
    """FQI with the number of epochs chosen on validation."""
    best = None
    for epochs in grid:
        args = SimpleNamespace(hidden=64, lr=1e-3, wd=1e-4, epochs=epochs, **cost)
        nets = fqi(tr_nodes, tr_roots, tr_idx, tr_X, tr_M, args)
        r = B.summary(B.play(val_env, fqi_chooser(nets, tr_nodes, tr_idx, tr_X, tr_M)))["reward"]
        if best is None or r > best[0]:
            best = (r, nets)
    return best[1]


def plot(rows, out, min_coverage=0.02):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: selective accuracy vs coverage. Points answering < 2% of questions are
    # dropped (accuracy undefined or based on a handful of questions).
    style = {"threshold": "o-", "FQI": "s-", "best fixed": "^"}
    for name, rs in groups.items():
        if name.startswith("FQI s"):                     # per-seed runs: shown via the mean
            continue
        rs = [r for r in sorted(rs, key=lambda r: r["lam"])
              if 1 - r["abstain"] >= min_coverage]
        if name in ("retrieve once", "oracle"):          # same at every lambda: one marker
            rs = rs[:1]
        if rs:
            left.plot([1 - r["abstain"] for r in rs], [r["sel_acc"] for r in rs],
                      style.get(name, "D"), label=name, markersize=6)
            if name in ("threshold", "FQI"):
                for r in rs:
                    left.annotate(f"λ={r['lam']:g}", (1 - r["abstain"], r["sel_acc"]),
                                  fontsize=7, xytext=(4, 4), textcoords="offset points")
    left.set_xlabel("coverage (share of questions answered)")
    left.set_ylabel("selective accuracy (correct / answered)")
    left.set_title("Accuracy vs coverage")

    # Right: FQI minus threshold reward per lambda, with 95% paired-bootstrap CI.
    fq = sorted(groups["FQI"], key=lambda r: r["lam"])
    lams = [r["lam"] for r in fq]
    d = [r["diff_vs_threshold"] for r in fq]
    err = [[r["diff_vs_threshold"] - r["ci_lo"] for r in fq],
           [r["ci_hi"] - r["diff_vs_threshold"] for r in fq]]
    right.errorbar(lams, d, yerr=err, fmt="s-", capsize=4, label="FQI (seed mean) − threshold")
    right.axhline(0, color="gray", lw=1)
    right.set_xscale("log", base=2)
    right.set_xticks(lams)
    right.set_xticklabels([f"{x:g}" for x in lams])
    right.set_xlabel("λ (wrong-answer penalty)")
    right.set_ylabel("reward difference (dev)")
    right.set_title("FQI vs threshold, 95% CI")

    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150)
    print("saved:", out + ".png")


def main(args):
    tr_nodes, all_roots = load_trees(args.train)
    tr_roots, val_roots = split_roots(all_roots, n_val=args.n_val)
    dv_nodes, dv_roots = load_trees(args.dev)
    B.build_caches(tr_nodes, dv_nodes)
    scaler = Scaler().fit([node_features(tr_nodes, i) for i, n in tr_nodes.items()
                           if n["qid"] in tr_roots])
    tr_idx, tr_X, tr_M = featurize(tr_nodes, scaler)
    dv_idx, dv_X, dv_M = featurize(dv_nodes, scaler)

    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for lam in [float(x) for x in args.lams.split(",")]:
        cost = dict(lam=lam, c_ret=args.c_ret, c_llm=args.c_llm)
        val_env = TreeEnv(tr_nodes, val_roots, **cost)
        dev_env = TreeEnv(dv_nodes, dv_roots, **cost)

        pol = {"retrieve once": B.play(dev_env, B.fixed_path("R"))}
        plan = B.tune_best_fixed(val_env)
        pol["best fixed"] = B.play(dev_env, B.fixed_path(plan))
        t = B.tune_threshold(val_env, tr_nodes)
        pol["threshold"] = B.play(dev_env, B.threshold(dv_nodes, *t))
        # FQI, once per seed. "FQI" is the seed average (per-question rewards averaged).
        runs = []
        for sd in seeds:
            torch.manual_seed(sd)
            nets = train_fqi(tr_nodes, tr_roots, val_env, tr_idx, tr_X, tr_M, cost)
            runs.append(B.play(dev_env, fqi_chooser(nets, dv_nodes, dv_idx, dv_X, dv_M)))
        if len(runs) > 1:
            for sd, res in zip(seeds, runs):
                pol[f"FQI s{sd}"] = res
        pol["FQI"] = [tuple(sum(run[i][j] for run in runs) / len(runs) for j in range(5))
                      for i in range(len(runs[0]))]
        _, best_act = solve(dev_env)
        pol["oracle"] = B.play(dev_env, lambda n: best_act[n])

        print(f"\nlambda={lam}  (best fixed: {plan or 'root'}, threshold: {t[0]}, "
              f"t={t[1]:.3g}, t_last={t[2]:.3g}, k={t[3]})")
        for name, res in pol.items():
            s = B.summary(res)
            d, lo, hi = B.paired_ci(res, pol["threshold"])
            print(f"  {name:14s} reward {s['reward']:+.3f}  InAcc {s['InAcc']:.3f}  "
                  f"abstain {s['abstain']:.3f}  sel.acc {s['sel_acc']:.3f}  "
                  f"LMC {s['LMC']:.2f}   vs threshold {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            rows.append({"lam": lam, "group": name, **s,
                         "diff_vs_threshold": d, "ci_lo": lo, "ci_hi": hi})
        if len(runs) > 1:
            seed_r = [B.summary(r)["reward"] for r in runs]
            wins = sum(1 for r in runs if B.paired_ci(r, pol["threshold"])[1] > 0)
            print(f"  FQI over {len(runs)} seeds: reward {statistics.mean(seed_r):+.3f} "
                  f"± {statistics.stdev(seed_r):.3f} (std);  seeds significantly above "
                  f"threshold: {wins}/{len(runs)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved:", args.out + ".csv")
    plot(rows, args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--lams", default="0.25,0.5,1,2")
    p.add_argument("--c_ret", type=float, default=0.05)
    p.add_argument("--c_llm", type=float, default=0.02)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--seeds", default="0", help="FQI seeds, e.g. 0,1,2,3,4")
    p.add_argument("--out", default="outputs/rl/sweep_bm25")
    main(p.parse_args())
