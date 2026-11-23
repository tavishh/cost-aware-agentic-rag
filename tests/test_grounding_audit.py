import json
from pathlib import Path
import tempfile
import unittest

from verifier.audit import evaluate, prepare
from verifier.score_nodes import score_node
from verifier.experiment import preserves_answer, sentence_premises
from verifier.review_packet import build_packet
from test_grounding import FakeBackend, PASSAGES, tree


class AuditTests(unittest.TestCase):
    def test_human_packet_hides_scores_and_prior_labels(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            inputs, _, _ = self.files(d)
            inputs.update(gold="PRIVATE_GOLD", grounding_score="PRIVATE_SCORE", label="PRIVATE_LABEL")
            (d/"inputs").write_text(json.dumps(inputs))
            self.assertEqual(build_packet([d/"inputs"], d/"packet"), 1)
            text = (d/"packet/review_packet.md").read_text()+(d/"packet/review_inputs.jsonl").read_text()
            for hidden in ("PRIVATE_GOLD", "PRIVATE_SCORE", "PRIVATE_LABEL"):
                self.assertNotIn(hidden, text)
            template = json.loads((d/"packet/annotations.template.jsonl").read_text())
            self.assertIsNone(template["label"])
            self.assertFalse(template["human_validated"])
            self.assertEqual(template["annotator"], "")

    def test_conversion_guard_detects_dropped_answer(self):
        self.assertFalse(preserves_answer("Vibe", "Cisco is in the Flash."))
        self.assertTrue(preserves_answer("David Thewlis", "David Thewlis plays Ares."))

    def test_short_premises_preserve_adjacent_context_and_remove_duplicates(self):
        passages = [{"text": "Mira designed it. She was born in York."}]*2
        self.assertEqual(sentence_premises(passages), ["Mira designed it.",
                         "She was born in York.", "Mira designed it. She was born in York."])

    def files(self, directory):
        node = tree()["q/R"]
        inputs = {"node_id": node["node_id"], "qid": "q", "depth": 1,
                  "question": node["question"], "answer": node["answer_call"]["text"],
                  "passages": list(PASSAGES.values())}
        annotation = {"node_id": node["node_id"], "label": "supported", "evidence_ids": ["p1"],
                      "annotator": "test", "human_validated": False}
        score = score_node(node, PASSAGES, FakeBackend())
        for name, row in (("inputs", inputs), ("annotations", annotation), ("scores", score)):
            (directory/name).write_text(json.dumps(row)+"\n")
        return inputs, annotation, score

    def test_evaluation_has_explicit_annotation_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.files(d)
            result = evaluate(d/"inputs", d/"annotations", d/"scores")
            self.assertEqual(result["confusion"]["tp"], 1)
            self.assertFalse(result["human_validated"])
            self.assertIsNone(result["roc_auc"])  # only one class

    def test_unclear_is_excluded_from_binary_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, annotation, _ = self.files(d)
            annotation["label"] = "unclear"
            (d/"annotations").write_text(json.dumps(annotation))
            result = evaluate(d/"inputs", d/"annotations", d/"scores")
            self.assertEqual(result["excluded_unclear"], [annotation["node_id"]])
            self.assertIsNone(result["accuracy"])

    def test_invalid_claim_is_missing_not_a_negative_score(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, _, score = self.files(d)
            score.update(status="invalid_claim", grounding_score=None,
                         claim_issues=["lost_question_constraint:first"])
            (d/"scores").write_text(json.dumps(score))
            result = evaluate(d/"inputs", d/"annotations", d/"scores")
            self.assertEqual(result["unavailable"], [score["node_id"]])
            self.assertEqual(result["score_coverage"], 0)
            self.assertIsNone(result["accuracy"])
            self.assertEqual(result["confusion"], dict(tp=0, fp=0, tn=0, fn=0))

    def test_bad_joins_and_stale_evidence_fail(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            inputs, _, _ = self.files(d)
            inputs["passages"][0] = dict(inputs["passages"][0], text="Changed")
            (d/"inputs").write_text(json.dumps(inputs))
            with self.assertRaises(ValueError): evaluate(d/"inputs", d/"annotations", d/"scores")
            self.files(d)
            (d/"scores").write_text("")
            with self.assertRaises(ValueError): evaluate(d/"inputs", d/"annotations", d/"scores")

    def test_sampling_is_reproducible_and_gold_blind(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = []
            for i in range(5):
                for depth in (1, 2, 3):
                    n = tree()["q/R"]
                    n.update(qid=f"q{i}", node_id=f"q{i}/"+"/".join(["R"]*depth), step=depth)
                    rows.append(n)
            (d/"tree").write_text("\n".join(json.dumps(n) for n in rows))
            (d/"passages").write_text(json.dumps(PASSAGES["p1"]))
            prepare(d/"tree", d/"passages", d/"a", per_depth=1)
            for n in rows:
                n["gold"] = ["changed"]
                n["metrics"] = {"em": "changed"}
            (d/"tree").write_text("\n".join(json.dumps(n) for n in rows))
            prepare(d/"tree", d/"passages", d/"b", per_depth=1)
            self.assertEqual((d/"a/review_inputs.jsonl").read_bytes(), (d/"b/review_inputs.jsonl").read_bytes())
            selected = [json.loads(l) for l in (d/"a/nodes.jsonl").read_text().splitlines()]
            self.assertEqual(len({n["qid"] for n in selected}), 3)
            self.assertNotIn("gold", selected[0])
            self.assertNotIn("has_answer", selected[0]["retrievals"][0])


if __name__ == "__main__":
    unittest.main()
