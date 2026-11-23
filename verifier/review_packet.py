"""Build a local score-blind packet for an actual human reviewer; sends nothing."""
import argparse
import json
from pathlib import Path

from verifier.audit import unique
from verifier.score_nodes import read_jsonl


def build_packet(input_paths, output, node_ids=None):
    rows = unique((r for path in input_paths for r in read_jsonl(path)), "node_id")
    if node_ids is not None:
        if set(node_ids) - set(rows):
            raise ValueError("selected reviewer nodes are missing")
        rows = {nid: rows[nid] for nid in node_ids}
    directory = Path(output)
    if directory.exists():
        raise FileExistsError("use a new packet directory")
    directory.mkdir(parents=True)
    text = ["# Independent grounding review\n",
            "Review the question and draft using only the supplied evidence. Do not look at\n"
            "model scores, cached gold answers, or prior assistant labels before completing\n"
            "your judgments. This packet contains none of those fields.\n",
            "Use supported if the evidence establishes every material claim/constraint;\n"
            "unsupported for missing, contradictory or only partial support; unclear for\n"
            "unresolvable ambiguity. Distinguish factual support from answer usefulness.\n"
            "Preserve dates, numeric ranges, first/second, actor/character roles, and the\n"
            "difference between a film's setting and filming location. For supported\n"
            "labels cite the passage IDs. Give a short reason for every label.\n",
            "Fill annotations.template.jsonl with your label, evidence_ids, rationale and\n"
            "annotator name. Set human_validated to true only for judgments you personally\n"
            "reviewed as a human. Do not leave the placeholder fields when submitting.\n"
            "The packet is deliberately selected for adjudication, not a representative benchmark.\n"]
    clean, templates = [], []
    for index, (nid, row) in enumerate(rows.items(), 1):
        item = {key: row[key] for key in ("node_id", "qid", "depth", "question", "answer")}
        item["passages"] = [{key: p.get(key, "") for key in ("docid", "title", "text")}
                            for p in row["passages"]]
        clean.append(item)
        templates.append({"node_id": nid, "label": None, "evidence_ids": [],
                          "rationale": "", "annotator": "", "human_validated": False})
        text += [f"\n## Case {index}: {nid}\n", f"Question: {item['question']}\n\nDraft: {item['answer']}\n"]
        seen = set()
        for passage in item["passages"]:
            if passage["docid"] in seen:
                continue
            seen.add(passage["docid"])
            text += [f"\n### Passage {passage['docid']}: {passage['title']}\n",
                     "\n".join("> "+line for line in passage["text"].splitlines())+"\n"]
    (directory/"review_packet.md").write_text("\n".join(text))
    for name, values in (("review_inputs.jsonl", clean), ("annotations.template.jsonl", templates)):
        (directory/name).write_text("".join(json.dumps(r)+"\n" for r in values))
    return len(clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("Prepared", build_packet(args.inputs, args.out), "review cases")
