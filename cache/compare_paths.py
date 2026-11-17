import json, sys
from collections import defaultdict

def load(fn):
    d = defaultdict(list)
    for l in open(fn):
        n = json.loads(l)
        d[n["node_id"][len(n["qid"]):].lstrip("/") or "root"].append(n)
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
avg = lambda xs, k: sum(x["metrics"][k] for x in xs) / len(xs)
print(f"{'path':8s} {'EM_a':>6} {'EM_b':>6} {'InAcc_a':>8} {'InAcc_b':>8}")
for p in sorted(a, key=lambda s: (s.count("/"), s)):
    print(f"{p:8s} {avg(a[p],'em'):6.3f} {avg(b[p],'em'):6.3f} "
          f"{avg(a[p],'inacc'):8.3f} {avg(b[p],'inacc'):8.3f}")
