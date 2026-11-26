"""Exploratory NLI support signal; not a context-sufficiency classifier."""
import math
import time

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
METHOD = "qa_nli_window_max_v1"


class NLIBackend:
    def __init__(self, model=DEFAULT_MODEL, revision="main", device="cpu",
                 batch_size=8, max_length=512, stride=64, cache_dir=None):
        if batch_size < 1 or not 64 <= max_length <= 512 or not 0 <= stride < max_length // 2:
            raise ValueError("invalid batch size, length or stride")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, cache_dir=cache_dir, use_safetensors=True).to(device).eval()
        labels = {str(label).lower(): int(i) for i, label in self.model.config.id2label.items()}
        if not {"entailment", "contradiction", "neutral"}.issubset(labels):
            raise ValueError("model must expose explicit NLI labels in id2label")
        self.entailment, self.contradiction = labels["entailment"], labels["contradiction"]
        self.device, self.batch_size = device, batch_size
        self.max_length, self.stride = max_length, stride
        self.config = {"method": METHOD, "model": model,
                       "revision": getattr(self.model.config, "_commit_hash", None) or revision,
                       "max_length": max_length, "stride": stride, "batch_size": batch_size,
                       "device": device}

    def score(self, question, answer, passages):
        # A fixed QA hypothesis avoids extra LLM calls, but is an approximation:
        # SNLI/MNLI models are not trained to judge arbitrary QA pairs directly.
        hypothesis = f"Question: {question}\nAnswer: {answer}"
        return self.score_hypothesis(hypothesis, passages)

    def score_hypothesis(self, hypothesis, passages):
        """Research hook for testing frozen, externally converted hypotheses.

        Callers must record the conversion method and check preservation of the
        original claim. The production score() method remains the raw-QA baseline.
        """
        return self.score_premises(hypothesis, ["\n\n".join(p.text for p in passages)])

    def score_premises(self, hypothesis, premises):
        """Research hook: aggregate entailment over explicitly supplied premise units."""
        if not premises:
            raise ValueError("at least one premise is required")
        if len(self.tokenizer.encode(hypothesis, add_special_tokens=False)) > self.max_length // 2:
            raise ValueError("question/answer too long; refusing to truncate the hypothesis")
        start = time.perf_counter()
        encoded = self.tokenizer(
            premises, [hypothesis]*len(premises), truncation="only_first", max_length=self.max_length,
            stride=self.stride, return_overflowing_tokens=True, padding=True, return_tensors="pt")
        encoded.pop("overflow_to_sample_mapping", None)
        count = len(encoded["input_ids"])
        ent, con = [], []
        calls = 0
        for offset in range(0, count, self.batch_size):
            batch = {k: v[offset:offset+self.batch_size].to(self.device) for k, v in encoded.items()}
            with self.torch.inference_mode():
                probabilities = self.model(**batch).logits.softmax(-1).cpu()
            ent.extend(probabilities[:, self.entailment].tolist())
            con.extend(probabilities[:, self.contradiction].tolist())
            calls += 1
        if not ent or any(not math.isfinite(p) for p in ent + con):
            raise ValueError("model produced invalid scores")
        return {"grounding_score": max(ent), "contradiction_score": max(con),
                "windows": count,
                "cost": {"verifier_calls": calls,
                         "input_tokens": int(encoded["attention_mask"].sum()),
                         "output_tokens": 0,
                         "latency_ms": (time.perf_counter()-start)*1000}}
