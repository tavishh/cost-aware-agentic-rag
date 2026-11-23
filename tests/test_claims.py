import unittest
from verifier.claims import build_claim, restore_answer, template_claim


class ClaimTests(unittest.TestCase):
    def test_cost_range_is_copied_exactly(self):
        result = build_claim("how much does it cost to join a club", "$10,000-$30,000")
        self.assertIn("$10,000-$30,000", result.text)
        self.assertFalse(result.issues)

    def test_generic_restoration_preserves_range_and_sign(self):
        self.assertEqual(restore_answer("$10,000-$30,000", "it costs 10 000 30 000"),
                         "it costs $10,000-$30,000")
        self.assertEqual(restore_answer("$10", "it costs $10"), "it costs $10")
        self.assertEqual(restore_answer("-3.5", "it is 3 5"), "it is -3.5")

    def test_number_order_and_answer_multiplicity_are_checked(self):
        self.assertIsNone(restore_answer("10-30", "the range is 30 10"))
        self.assertIsNone(restore_answer("Mira", "Mira met Mira"))
        self.assertIsNone(restore_answer("Vibe", "Cisco is in the show"))

    def test_filming_relation_is_preserved(self):
        result = build_claim("where was a sitcom in italy filmed", "Anguillara Sabazia outside Rome.")
        self.assertEqual(result.text, "a sitcom in italy was filmed in Anguillara Sabazia outside Rome.")

    def test_comparison_preserves_both_options(self):
        result = build_claim("what came out first game alpha or game beta", "Game Beta")
        self.assertEqual(result.text, "Game Beta was released before game alpha.")
        self.assertIsNone(template_claim("what came out first game alpha or game beta", "Both"))

    def test_lost_constraints_reject_fallback(self):
        cases = [("when did a country first win a cup", "2011", "a country won a cup in 2011"),
                 ("which is the largest and most complex part", "Cerebrum", "the largest part is cerebrum"),
                 ("identify the place where a movie was filmed", "Rome", "the movie is in rome")]
        for question, answer, generated in cases:
            with self.subTest(question=question):
                self.assertTrue(build_claim(question, answer, generated).issues)

    def test_date_in_question_is_not_silently_changed(self):
        result = build_claim("identify the recipient in 1801", "France", "the recipient in 1800 was france")
        self.assertIn("lost_question_number:1801", result.issues)
        result = build_claim("identify the recipient in 1801", "France", "the recipient in 18010 was france")
        self.assertIn("lost_question_number:1801", result.issues)


if __name__ == "__main__":
    unittest.main()
