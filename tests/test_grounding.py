"""Pipeline tests with a deterministic backend; these do not assert NLI accuracy."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from common.node_adapter import cumulative_cost, node_to_observation, context_passages
from common.schemas import Verdict, VerificationResult
from rl.features import FEATURE_NAMES, node_features
from verifier.integration import grounded_node_features, load_scores
from verifier.score_nodes import run, score_node


def tree():
    call = {"text": "Mira Chen", "tokens_in": 10, "tokens_out": 2,
            "latency_s": .1, "token_logprobs": [[["Mira", -.1], ["Omar", -3]]],}
    root = {"node_id": "q", "qid": "q", "parent_id": None, "question": "Who designed the bridge?",
            "gold": ["Mira Chen"], "step": 0, "action": None,
            "current_query": "bridge designer", "retrievals": [], "context_docids": [],
            "answer_call": copy.deepcopy(call), "metrics": {"inacc": 1}}
    r = copy.deepcopy(root)
    r.update(node_id="q/R", parent_id="q", step=1, action="retrieve", context_docids=["p1"],
             retrievals=[{"query": "bridge designer", "docids": ["p1"], "scores": [4.0],
                          "has_answer": [True]}])
    w = copy.deepcopy(r)
    w.update(node_id="q/R/W", parent_id="q/R", step=2, action="rewrite",
             rewrite_call={"tokens_in": 6, "tokens_out": 3, "latency_s": .2},
             context_docids=["p1", "p1"])
    w["retrievals"].append({"query": "Aster architect", "docids": ["p1"], "scores": [3.0]})
    return {x["node_id"]: x for x in (root, r, w)}


PASSAGES = {"p1": {"docid": "p1", "title": "Aster Bridge", "text": "Mira Chen designed it."}}


class FakeBackend:
    config = {"method": "test_backend", "revision": "test-v1"}
    def __init__(self): self.calls = 0
    def score(self, question, answer, passages):
        self.calls += 1
        return {"grounding_score": .8, "contradiction_score": .1, "windows": 1,
                "cost": {"verifier_calls": 1, "input_tokens": 25,
                         "output_tokens": 0, "latency_ms": 1.0}}


class GroundingTests(unittest.TestCase):
    def adapt(self, nodes, nid):
        return node_to_observation(nodes, nid, PASSAGES, model_version="llama-v1",
                                   prompt_version="simple-v1", greedy=True)

    def test_adapter_preserves_repeated_generator_context_and_units(self):
        obs = self.adapt(tree(), "q/R/W")
        self.assertEqual([p.docid for p in obs.passages], ["p1", "p1"])
        self.assertEqual(obs.remaining_steps, 1)
        self.assertAlmostEqual(obs.generation.cost.latency_ms, 100)
        self.assertAlmostEqual(obs.uncertainty.value, .1)
        self.assertIsNone(obs.generation.token_logprobs)  # never mislabel candidates as selected tokens

    def test_adapter_does_not_expose_gold_fields(self):
        nodes = tree()
        original = self.adapt(nodes, "q/R").to_dict()
        nodes["q/R"]["gold"] = ["WRONG"]
        nodes["q/R"]["metrics"] = {"em": 0}
        nodes["q/R"]["retrievals"][0]["has_answer"] = [False]
        self.assertEqual(original, self.adapt(nodes, "q/R").to_dict())

    def test_cost_counts_each_draft_and_rewrite_once(self):
        c = cumulative_cost(tree(), "q/R/W")
        self.assertEqual((c.retrieval_calls, c.generation_calls), (2, 4))
        self.assertEqual((c.input_tokens, c.output_tokens), (36, 9))
        self.assertAlmostEqual(c.latency_ms, 500)

    def test_missing_parent_and_cycle_fail(self):
        nodes = tree()
        del nodes["q/R"]
        with self.assertRaises(KeyError): cumulative_cost(nodes, "q/R/W")
        nodes = tree()
        nodes["q"]["parent_id"] = "q/R"
        with self.assertRaises(ValueError): cumulative_cost(nodes, "q")

    def test_context_requires_metadata_if_ids_not_saved(self):
        node = tree()["q/R"]
        del node["context_docids"]
        with self.assertRaises(ValueError): context_passages(node, PASSAGES)
        self.assertEqual(len(context_passages(node, PASSAGES, context_mode="latest")), 1)

    def test_nli_verifier_metadata_can_be_absent(self):
        r = VerificationResult(Verdict.YES, ("p1",), "nli-model-v1", score=.8)
        self.assertIsNone(r.explanation)
        self.assertIsNone(r.prompt_version)
        with self.assertRaises(ValueError):
            VerificationResult(Verdict.YES, (), "")

    def test_root_is_no_evidence_not_incorrect(self):
        backend = FakeBackend()
        result = score_node(tree()["q"], PASSAGES, backend)
        self.assertEqual(result["grounding_score"], 0)
        self.assertEqual(result["status"], "no_evidence")
        self.assertFalse(result["has_evidence"])
        self.assertEqual(backend.calls, 0)

    def test_empty_answer_is_explicitly_missing(self):
        node = tree()["q/R"]
        node["answer_call"]["text"] = " "
        result = score_node(node, PASSAGES, FakeBackend())
        self.assertIsNone(result["grounding_score"])
        self.assertEqual(result["status"], "empty_answer")

    def test_gold_changes_do_not_change_verifier_input_hash_or_score(self):
        nodes = tree()
        before = score_node(nodes["q/R"], PASSAGES, FakeBackend())
        nodes["q/R"]["gold"] = ["WRONG"]
        nodes["q/R"]["metrics"] = {"inacc": 0}
        after = score_node(nodes["q/R"], PASSAGES, FakeBackend())
        self.assertEqual(before, after)

    def test_invalid_model_score_is_not_zero_filled(self):
        backend = FakeBackend()
        backend.score = lambda *args: {"grounding_score": math.nan}
        with self.assertRaises(ValueError): score_node(tree()["q/R"], PASSAGES, backend)

    def test_runner_resume_joins_and_opt_in_features(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            nodes = tree()
            (d/"tree").write_text("\n".join(json.dumps(n) for n in nodes.values()))
            (d/"passages").write_text(json.dumps(PASSAGES["p1"]))
            backend = FakeBackend()
            run(d/"tree", d/"passages", d/"scores", backend)
            self.assertEqual(backend.calls, 2)
            run(d/"tree", d/"passages", d/"scores", backend, resume=True)
            self.assertEqual(backend.calls, 2)
            scores = load_scores(d/"scores", nodes)
            base = node_features(nodes, "q/R")
            extended = grounded_node_features(nodes, "q/R", scores)
            self.assertEqual(len(base), len(FEATURE_NAMES))
            self.assertEqual(extended, base + [.8, 1.0, 1.0])
            nodes["q/R"]["answer_call"]["text"] = "Omar"
            with self.assertRaises(ValueError): load_scores(d/"scores", nodes)
            with self.assertRaises(FileExistsError): run(d/"tree", d/"passages", d/"scores", backend)

    def test_missing_evidence_is_an_error_not_a_no_evidence_score(self):
        with self.assertRaises(KeyError): score_node(tree()["q/R"], {}, FakeBackend())

    def test_resume_rejects_stale_context_without_changing_output(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d/"tree").write_text("\n".join(json.dumps(n) for n in tree().values()))
            (d/"passages").write_text(json.dumps(PASSAGES["p1"]))
            run(d/"tree", d/"passages", d/"scores", FakeBackend())
            original = (d/"scores").read_bytes()
            changed = dict(PASSAGES["p1"], text="Different evidence")
            (d/"passages").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                run(d/"tree", d/"passages", d/"scores", FakeBackend(), resume=True)
            self.assertEqual((d/"scores").read_bytes(), original)

    def test_duplicate_and_missing_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"scores"
            row = score_node(tree()["q"], PASSAGES, FakeBackend())
            p.write_text(json.dumps(row)+"\n")
            with self.assertRaises(ValueError): load_scores(p, tree())
            p.write_text((json.dumps(row)+"\n")*2)
            with self.assertRaises(ValueError): load_scores(p, {"q": tree()["q"]})

    def test_rl_features_ignore_gold_and_metrics(self):
        nodes = tree()
        before = node_features(nodes, "q/R/W")
        for n in nodes.values():
            n["gold"] = ["poison"]
            n["metrics"] = {"inacc": "poison"}
            for rt in n["retrievals"]: rt["has_answer"] = ["poison"]
        self.assertEqual(before, node_features(nodes, "q/R/W"))


if __name__ == "__main__":
    unittest.main()
