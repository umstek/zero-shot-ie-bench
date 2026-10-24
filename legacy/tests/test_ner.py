import unittest

from bench import NER, ner_repeat_loop, spans_of, strict_prf


class NerScoringTests(unittest.TestCase):
    def test_wrong_document_cannot_supply_a_true_positive(self):
        cases = [("Alice", {(0, 5, "person")}),
                 ("Paris", {(0, 5, "location")})]
        predicted, _ = ner_repeat_loop(
            lambda text: frozenset({(0, 5, "location" if text == "Alice"
                                    else "person")}), cases, 2)
        gold = {(0, 0, 5, "person"), (1, 0, 5, "location")}
        self.assertEqual(strict_prf(predicted, gold), (0.0, 0.0, 0.0))

    def test_identical_offsets_in_different_documents_count_twice(self):
        cases = [("Alice", {(0, 5, "person")}),
                 ("James", {(0, 5, "person")})]
        predicted, stats = ner_repeat_loop(
            lambda text: frozenset({(0, 5, "person")}) if text == "Alice"
            else frozenset(), cases, 2)
        gold = {(0, 0, 5, "person"), (1, 0, 5, "person")}
        self.assertEqual(strict_prf(predicted, gold), (1.0, 0.5, 0.6667))
        self.assertEqual(stats["stability"], 1.0)

    def test_full_gold_set_keeps_all_thirty_entities(self):
        cases = [(text, spans_of(text, truth)) for text, truth in NER]
        by_text = dict(cases)
        predicted, _ = ner_repeat_loop(
            lambda text: frozenset(by_text[text]), cases, 1)
        self.assertEqual(len(predicted), 30)


if __name__ == "__main__":
    unittest.main()
