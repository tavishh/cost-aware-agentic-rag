import json, sys, argparse, statistics
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--test", required=True)
p.add_argument("--train")
p.add_argument("--dev")
args = p.parse_args()

def load(fn):
    d = defaultdict(list)
    for l in open(fn):
        n = json.loads(l)
        d[n["node_id"][len(n["qid"]):].lstrip("/") or "root"].append(n)
    return d

avg = lambda xs, k: sum(x["metrics"][k] for x in xs) / len(xs)
order = lambda d: sorted(d, key=lambda s: (s.count("/"), s))
test = load(args.test)

# ---------- 任务 1：train/dev 合理性检查 ----------
print("=== Task 1: no-retrieval / retrieve-once InAcc vs test ===")
ref = {"root": avg(test["root"], "inacc"), "R": avg(test["R"], "inacc")}
print(f"{'split':6s} {'root':>6} {'R':>6}   (test: root {ref['root']:.3f}, R {ref['R']:.3f})")
for name, fn in [("train", args.train), ("dev", args.dev)]:
    if not fn:
        print(f"{name:6s} (not provided yet)")
        continue
    d = load(fn)
    vals = {k: avg(d[k], "inacc") for k in ("root", "R")}
    flags = [k for k in vals if abs(vals[k] - ref[k]) > 0.03]
    print(f"{name:6s} {vals['root']:6.3f} {vals['R']:6.3f}   "
          + (f"FLAG: gap > 3 pts on {', '.join(flags)}" if flags else "ok"))

# ---------- 任务 2：答案长度（token） ----------
print("\n=== Task 2: answer length (tokens_out) on test ===")
print(f"{'path':8s} {'mean':>6} {'median':>6}")
for path in order(test):
    lens = [n["answer_call"]["tokens_out"] for n in test[path]]
    print(f"{path:8s} {statistics.mean(lens):6.2f} {statistics.median(lens):6.1f}")
root_len = statistics.mean(n["answer_call"]["tokens_out"] for n in test["root"])
ret_len = statistics.mean(n["answer_call"]["tokens_out"]
                          for p_, ns in test.items() if p_ != "root" for n in ns)
print(f"root mean {root_len:.2f} vs retrieval paths mean {ret_len:.2f}")

# ---------- 任务 3：EM / F1 / InAcc ----------
print("\n=== Task 3: EM / F1 / InAcc by path on test ===")
print(f"{'path':8s} {'EM':>6} {'F1':>6} {'InAcc':>6} {'InAcc-EM':>9}")
for path in order(test):
    ns = test[path]
    e, f, i = avg(ns, "em"), avg(ns, "f1"), avg(ns, "inacc")
    print(f"{path:8s} {e:6.3f} {f:6.3f} {i:6.3f} {i - e:9.3f}")