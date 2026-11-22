import json, collections

fn = "outputs/tree/nq_arag_test_tree15_Llama-3.1-8B-Instruct_accumulate.jsonl"
rows = [json.loads(l) for l in open(fn)]
print("lines:", len(rows), " questions:", len({r["qid"] for r in rows}))
print("top-level keys:", sorted(rows[0].keys()))

paths = collections.Counter()
for r in rows:
    p = r["node_id"][len(r["qid"]):].lstrip("/") or "root"
    paths[p] += 1
print("\npaths and counts:")
for p, c in sorted(paths.items(), key=lambda x: (x[0].count("/"), x[0])):
    print(f"  {p:12} {c}")

skip = ("answer_call", "retrievals", "context_docids", "path_rewrite_calls")
child = next(r for r in rows if r["parent_id"] is not None)
print("\nexample child (long fields hidden):")
print({k: v for k, v in child.items() if k not in skip})
print("\nanswer_call keys:", [k for k in child["answer_call"] if k != "token_logprobs"])
