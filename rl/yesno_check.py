"""Check yes/no questions on a split before deciding the InAcc rule for multi-hop test runs.

For questions whose gold answer is "yes" or "no", whole-word InAcc can still be too lenient:
a draft like "There is no information" contains the word "no". This compares strict InAcc
with a stricter rule (the draft must START with the gold yes/no) and shows disagreements.

Usage (from the repo root, on a ddev file):
    python3 -m rl.yesno_check outputs/tree/2wiki_ddev_tree15_Llama-3.1-8B-Instruct_accumulate.jsonl
"""
import sys

from rl.env import inacc, normalize
from rl.tree_data import load_trees

YN = {"yes", "no"}


def starts_with_gold(pred, golds):
    first = normalize(pred).split()[:1]
    return int(any(first == [normalize(g)] for g in golds))


if __name__ == "__main__":
    nodes, roots = load_trees(sys.argv[1])
    yn_q = {q for q, r in roots.items() if {normalize(g) for g in nodes[r]["gold"]} & YN}
    print(f"questions: {len(roots)}  yes/no gold: {len(yn_q)} ({len(yn_q) / len(roots):.1%})")
    if not yn_q:
        sys.exit("no yes/no questions: the standard strict InAcc rule is fine")

    n = strict_ok = start_ok = 0
    false_pos = []
    for node_id, node in nodes.items():
        if node["qid"] not in yn_q:
            continue
        gold = nodes[roots[node["qid"]]]["gold"]
        pred = node["answer_call"]["text"]
        s, t = inacc(pred, gold, strict=True), starts_with_gold(pred, gold)
        n += 1
        strict_ok += s
        start_ok += t
        if s and not t:
            false_pos.append((pred, gold))

    print(f"yes/no nodes: {n}  correct (strict): {strict_ok}  correct (starts with gold): {start_ok}")
    print(f"credited by strict but not starting with yes/no: {len(false_pos)} "
          f"({len(false_pos) / n:.1%} of yes/no nodes, "
          f"{len(false_pos) / len(nodes):.2%} of all nodes)")
    for pred, gold in false_pos[:10]:
        print(f"  gold={gold}  draft={pred!r}")
