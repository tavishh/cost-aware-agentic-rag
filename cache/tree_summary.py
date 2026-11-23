"""Tree summaries with strict (whole-word) InAcc as the main metric, substring InAcc as an extra
column. strict matching is identical to rl/env.py: inacc(pred, golds, strict=True).

  python cache/tree_summary.py outputs/tree/nq_arag_test_tree15_*.jsonl            # one file, full table
  python cache/tree_summary.py --select outputs/tree/nq_ddev_dpr_tree15_*.jsonl \\
      outputs/tree/nq_arag_test_tree15_*.jsonl outputs/tree/nq_ddev_dpr_tree15_*.jsonl ...

--select FILE: also report the fixed path chosen (by strict InAcc) on FILE, evaluated on every file.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, json
from collections import defaultdict
from tree_common import norm, em


def inacc(pred, golds, strict=True):
    p = norm(pred)
    for g in golds:
        g = norm(g)
        if g and ((f" {g} " in f" {p} ") if strict else (g in p)):
            return 1
    return 0


def path_of(n):
    return n["node_id"][len(n["qid"]):].lstrip("/") or "root"


def load(path):
    rows, mismatch = [], 0
    for line in open(path):
        n = json.loads(line)
        text = n["answer_call"]["text"]
        r = {"qid": n["qid"], "path": path_of(n),
             "strict": inacc(text, n["gold"], True), "sub": inacc(text, n["gold"], False),
             "em": em(text, n["gold"]), "tok_in": n["cum_cost"]["tokens_in"]}
        if "metrics" in n and n["metrics"]["inacc"] != r["sub"]:
            mismatch += 1
        rows.append(r)
    return rows, mismatch


def summarize(rows):
    by_path = defaultdict(list)
    for r in rows:
        by_path[r["path"]].append(r)
    table = {}
    for p, sub in by_path.items():
        m = len(sub)
        table[p] = {"n": m, **{k: sum(x[k] for x in sub) / m for k in ("strict", "sub", "em", "tok_in")}}
    orc = {}
    for key in ("strict", "sub"):
        full, ronly = defaultdict(int), defaultdict(int)
        for r in rows:
            full[r["qid"]] = max(full[r["qid"]], r[key])
            if "W" not in r["path"]:
                ronly[r["qid"]] = max(ronly[r["qid"]], r[key])
        orc[key] = {"full": sum(full.values()) / len(full), "ronly": sum(ronly.values()) / len(ronly)}
    return table, orc


def best_path(table, key="strict"):
    return max(table, key=lambda p: (table[p][key], -table[p]["tok_in"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--select", default="", help="pick the fixed path on this file (e.g. ddev)")
    ap.add_argument("--full", action="store_true", help="print the per-path table for every file")
    a = ap.parse_args()

    sel_path = None
    if a.select:
        sel_rows, _ = load(a.select)
        sel_path = best_path(summarize(sel_rows)[0])
        print(f"fixed path selected on {os.path.basename(a.select)} (strict): {sel_path}\n")

    headline = []
    for fn in a.files:
        rows, mismatch = load(fn)
        table, orc = summarize(rows)
        name = os.path.basename(fn).split("_tree15")[0]
        if a.full or len(a.files) == 1:
            print(f"== {name}  (substring column vs stored metrics: {mismatch} mismatches)")
            print(f"{'path':10s} {'nodes':>6} {'InAcc':>7} {'InAcc_sub':>9} {'EM':>6} {'cum_tok_in':>10}")
            for p in sorted(table, key=lambda s: (s.count("/"), s != "root", s)):
                t = table[p]
                print(f"{p:10s} {t['n']:>6} {t['strict']:>7.3f} {t['sub']:>9.3f} {t['em']:>6.3f} {t['tok_in']:>10.0f}")
            print(f"oracle  R-only {orc['strict']['ronly']:.3f} (sub {orc['sub']['ronly']:.3f})   "
                  f"full tree {orc['strict']['full']:.3f} (sub {orc['sub']['full']:.3f})\n")
        elif mismatch:
            print(f"WARNING {name}: substring InAcc differs from stored metrics on {mismatch} nodes")
        bp = best_path(table)
        h = {"name": name, "root": table["root"], "R": table.get("R"), "RRR": table.get("R/R/R"),
             "best": (bp, table[bp]), "orc": orc}
        if sel_path:
            h["sel"] = table.get(sel_path)
        headline.append(h)

    # compact summary: strict (substring in parentheses)
    def f(t, k="strict"):
        return f"{t[k]:.3f} ({t['sub']:.3f})" if t else "—"
    cols = ["root", "R", "R/R/R", "best fixed (path)"] + (["selected path"] if sel_path else []) + \
           ["oracle R-only", "oracle full", "oracle − best"]
    print("Summary — strict InAcc (substring in parentheses)")
    print("| split | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for h in headline:
        bp, bt = h["best"]
        cells = [f(h["root"]), f(h["R"]), f(h["RRR"]), f"{f(bt)} {bp}"]
        if sel_path:
            cells.append(f(h["sel"]))
        o = h["orc"]
        cells += [f"{o['strict']['ronly']:.3f} ({o['sub']['ronly']:.3f})",
                  f"{o['strict']['full']:.3f} ({o['sub']['full']:.3f})",
                  f"{100*(o['strict']['full']-bt['strict']):.1f} ({100*(o['sub']['full']-bt['sub']):.1f})"]
        print(f"| {h['name']} | " + " | ".join(cells) + " |")
