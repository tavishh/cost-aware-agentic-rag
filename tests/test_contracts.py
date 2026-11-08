import json
import math
from pathlib import Path
import unittest

from common.schemas import (
    Action, Cost, GenerationResult, Observation, Passage, UncertaintyResult,
    Verdict, VerificationResult,
)


class ContractTests(unittest.TestCase):
    def observation(self, **kwargs):
        values = dict(qid="q1", question="Who designed it?", query="bridge designer",
                      passages=(Passage("p1", "Mira designed it.", 4.2),),
                      generation=GenerationResult("Mira", "model-v1", "prompt-v1", (-0.2,)),
                      uncertainty=UncertaintyResult("mean_nll_v1", 0.2))
        values.update(kwargs)
        return Observation(**values)

    def result(self, **kwargs):
        values = dict(verdict=Verdict.YES, evidence_ids=("p1",), explanation="Direct support",
                      verifier_version="verifier-v1", prompt_version="prompt-v1")
        values.update(kwargs)
        return VerificationResult(**values)

    def test_policy_payload_is_json_compatible_and_has_no_gold_labels(self):
        payload = json.loads(json.dumps(self.observation(grounding=self.result()).to_dict(), allow_nan=False))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["grounding"]["verdict"], "yes")
        forbidden = {"gold", "golden_answers", "has_answer", "correctness", "em", "f1", "answerable"}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values(): check(item)
            elif isinstance(value, list):
                for item in value: check(item)
        check(payload)
        with self.assertRaises(TypeError):
            self.observation(gold=["Mira"])

    def test_missing_uncertainty_is_explicit(self):
        self.assertIsNone(UncertaintyResult("mean_nll_v1", None, "backend omits logprobs").value)
        with self.assertRaises(ValueError): UncertaintyResult("mean_nll_v1", None)
        with self.assertRaises(ValueError): UncertaintyResult("mean_nll_v1", 0.1, "missing")

    def test_invalid_numeric_signals_are_rejected(self):
        for value in (math.nan, math.inf, -0.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): UncertaintyResult("nll", value)
        for values in ((), (0.1,), (math.nan,), (-math.inf,)):
            with self.assertRaises(ValueError): GenerationResult("Mira", "m", "p", values)
        for score in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError): self.result(score=score)

    def test_costs_are_nonnegative_and_counts_integral(self):
        for kwargs in ({"verifier_calls": -1}, {"input_tokens": 1.5}, {"latency_ms": math.inf}):
            with self.assertRaises(ValueError): Cost(**kwargs)

    def test_evidence_references_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.observation(grounding=self.result(evidence_ids=("not-retrieved",)))
        with self.assertRaises(ValueError):
            self.observation(passages=(Passage("p1", "a"), Passage("p1", "b")))

    def test_terminal_actions_cannot_appear_in_live_history(self):
        self.observation(action_history=(Action.RETRIEVE, Action.REWRITE), remaining_steps=1)
        with self.assertRaises(ValueError): self.observation(action_history=(Action.ABSTAIN,))
        with self.assertRaises(ValueError): self.observation(remaining_steps=-1)

    def test_unknown_is_not_a_numeric_failure_or_zero(self):
        self.result(verdict=Verdict.UNKNOWN, evidence_ids=())
        with self.assertRaises(ValueError): self.result(verdict=Verdict.UNKNOWN, score=0.0)

    def test_fixture_integrity_and_coverage(self):
        path = Path(__file__).parent / "fixtures" / "verifier_cases.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r["case_id"] for r in rows}), len(rows))
        combinations = set()
        for row in rows:
            inputs, expected = row["input"], row["expected"]
            self.assertEqual(set(inputs), {"question", "answer", "passages"})
            ids = {Passage(**p).docid for p in inputs["passages"]}
            self.assertTrue(set(expected["supporting_evidence_ids"]).issubset(ids))
            s, g = Verdict(expected["sufficiency"]), Verdict(expected["grounding"])
            combinations.add((s, g))
            self.assertTrue(expected["rationale"])
        self.assertIn((Verdict.YES, Verdict.NO), combinations)
        self.assertIn((Verdict.UNKNOWN, Verdict.UNKNOWN), combinations)


if __name__ == "__main__":
    unittest.main()
