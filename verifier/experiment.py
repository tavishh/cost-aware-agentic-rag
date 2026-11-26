"""Experimental QA2D + short-premise variant for development audits only.

This is not wired into score_nodes or the production RL features.
"""
import argparse
import json
from pathlib import Path
import re
import string
import time

from verifier.nli import NLIBackend
from verifier.score_nodes import fingerprint, read_jsonl
from verifier.claims import VERSION as CLAIM_VERSION, build_claim


def normalize(text):
    return " ".join("".join(" " if c in string.punctuation else c for c in text.lower()).split())


def preserves_answer(answer, hypothesis):
    stop = {"a", "an", "the", "and", "in", "of", "to", "is", "are"}
    tokens = lambda text: set(re.findall(r"\w+", text.lower())) - stop
    return bool(tokens(answer)) and tokens(answer).issubset(tokens(hypothesis))


def sentence_premises(passages):
    """All single sentences and adjacent sentence pairs; no answer-based selection."""
    result = []
    for passage in passages:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage["text"]) if s.strip()]
        result.extend(sentences)
        result.extend(a+" "+b for a, b in zip(sentences, sentences[1:]))
    return list(dict.fromkeys(result))


def run(inputs, output, cache_dir, revision="main", claim_mode="v1"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new experiment output")
    name = "domenicrosati/QA2D-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    converter = AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision, cache_dir=cache_dir).eval()
    backend = NLIBackend(revision="fa2804872c3b4bd748f38c0185cc85775361e735", cache_dir=cache_dir)
    config = {**backend.config, "method": "experimental_qa2d_sentencepair_max_v1",
              "converter": name, "converter_revision": converter.config._commit_hash,
              "converter_preprocessing": "lowercase_punctuation_to_space_v1",
              "num_beams": 4, "max_new_tokens": 128,
              "conversion_guard": "all answer content tokens or fallback raw QA"}
    if claim_mode == "guarded-v2":
        config.update(method="experimental_guarded_sentencepair_max_v2",
                      conversion_guard=CLAIM_VERSION, failure_policy="null score; invalid_claim")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for item in read_jsonl(inputs):
            start = time.perf_counter()
            decision = build_claim(item["question"], item["answer"]) if claim_mode == "guarded-v2" else None
            conversion_calls = input_tokens = output_tokens = 0
            if decision is None or decision.issues:
                encoded = tokenizer(normalize(item["question"])+". "+normalize(item["answer"]), return_tensors="pt")
                if encoded["input_ids"].shape[1] > 512:
                    raise ValueError("conversion input too long")
                with torch.inference_mode():
                    generated = converter.generate(**encoded, max_new_tokens=128, num_beams=4, do_sample=False)
                claim = tokenizer.decode(generated[0], skip_special_tokens=True)
                conversion_calls, input_tokens, output_tokens = 1, int(encoded["attention_mask"].sum()), int(generated.shape[1]-1)
                if claim_mode == "guarded-v2":
                    decision = build_claim(item["question"], item["answer"], claim)
            else:
                claim = decision.text
            preserved = not decision.issues if decision else preserves_answer(item["answer"], claim)
            hypothesis = decision.text if decision else (claim if preserved else f"Question: {item['question']}\nAnswer: {item['answer']}")
            conversion_ms = (time.perf_counter()-start)*1000
            if decision and decision.issues:
                result = {"grounding_score": None, "cost": {"verifier_calls": 0,
                          "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}}
            else:
                result = backend.score_premises(hypothesis, sentence_premises(item["passages"]))
            result["cost"]["verifier_calls"] += conversion_calls
            result["cost"]["input_tokens"] += input_tokens
            result["cost"]["output_tokens"] += output_tokens
            result["cost"]["latency_ms"] += conversion_ms
            full_input = {k: item[k] for k in ("node_id", "qid", "question", "answer")}
            full_input["passages"] = [{"docid": p["docid"], "text":
                f"Wikipedia Title: {p['title']}\n{p['text']}" if p.get("title") else p["text"]}
                for p in item["passages"]]
            row = {"schema_version": 1, "node_id": item["node_id"], "qid": item["qid"],
                   "input_hash": fingerprint(full_input), "answer_hash": fingerprint(item["answer"]),
                   "context_docids": [p["docid"] for p in item["passages"]], "has_evidence": True,
                   "status": "invalid_claim" if decision and decision.issues else "ok",
                   "claim_strategy": decision.strategy if decision else "qa2d_v1",
                   "claim_issues": list(decision.issues) if decision else [],
                   "verifier": config, "hypothesis": hypothesis,
                   "converted_hypothesis": claim, "conversion_answer_preserved": preserved, **result}
            f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
            print(item["node_id"], round(row["grounding_score"],4) if row["grounding_score"] is not None else "invalid_claim", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--claim-mode", choices=("v1", "guarded-v2"), default="v1")
    args = parser.parse_args()
    run(args.inputs, args.out, args.cache_dir, args.revision, args.claim_mode)
