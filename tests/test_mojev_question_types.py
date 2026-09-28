"""Offline tests for the MoJev typed-question surface (the vendored
serve.py option_texts/build_answer pair and the multi-field packer).

No checkpoint, no network: the pure question->candidates/answer mappings,
the packer span math against a character-level fake tokenizer, and the
demo module's import hygiene. engines.mojev_engine imports torch at module
scope, which the main venv ships, so torch is NOT blocked here (only
transformers, which must stay out of the demo module and the packer).

    .venv/Scripts/python -m unittest tests.test_mojev_question_types
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demos import mojev_demo as demo
from engines.mojev_engine import build_answer, option_texts
from engines.mojev_engine import packer


class FakeTokenizer:
    """Character-level stand-in for the Qwen tokenizer: one integer id per
    character, so every span window is checkable with len()/range()."""

    pad_token_id = 0

    def __call__(self, texts, truncation=False, max_length=None):
        rows = [[(ord(ch) % 251) + 1 for ch in text] for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}


class ImportTests(unittest.TestCase):
    def test_demo_imports_pull_no_heavy_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            import importlib

            importlib.reload(demo)

    def test_packer_imports_without_transformers(self):
        import importlib

        with patch.dict(sys.modules, {"transformers": None}):
            importlib.reload(packer)


class FieldPromptTests(unittest.TestCase):
    def test_kind_choice_and_options_join(self):
        self.assertEqual(
            packer.field_prompt("urgency", "How urgent?", ("high", "low")),
            "urgency | How urgent? | kind: choice | options: high, low")

    def test_empty_description_is_dropped(self):
        self.assertEqual(packer.field_prompt("refund", "", ("no", "yes")),
                         "refund | kind: choice | options: no, yes")

    def test_wide_menus_drop_the_options_list(self):
        options = tuple(f"o{i}" for i in range(9))
        prompt = packer.field_prompt("t", "d", options)
        self.assertNotIn("options:", prompt)


class CandidateOrderTests(unittest.TestCase):
    def test_sort_and_unsort_round_trip(self):
        options = ["c", "a", "b"]
        order = packer.sort_candidates(options)
        self.assertEqual([options[i] for i in order], ["a", "b", "c"])
        # unsort maps sorted-position results back to caller order
        self.assertEqual(packer.unsort(order, [0.1, 0.2, 0.3]),
                         [0.3, 0.1, 0.2])


class OptionTextsTests(unittest.TestCase):
    def test_choice_descriptions_become_label_colon_description(self):
        options, kind = option_texts("team", {
            "type": "choice",
            "criteria": {"billing": "payments", "technical": None,
                         "account": ""}})
        self.assertEqual(kind, "choice")
        self.assertEqual(options,
                         ["billing: payments", "technical", "account"])

    def test_noul_defaults_to_no_then_yes(self):
        options, kind = option_texts("refund", {"type": "noul"})
        self.assertEqual(kind, "noul")
        self.assertEqual(options, ["no", "yes"])

    def test_noul_candidates_stay_false_then_true(self):
        options, _ = option_texts("refund", {
            "type": "noul",
            "criteria": {"true": "money back", "false": "nothing owed"}})
        self.assertEqual(options, ["nothing owed", "money back"])

    def test_non_string_criteria_render_as_json(self):
        options, _ = option_texts("s", {"type": "score", "criteria": [1, 2.5]})
        self.assertEqual(options, ["1", "2.5"])

    def test_score_keeps_rubric_order(self):
        options, kind = option_texts("urgency", {
            "type": "score",
            "criteria": ["low", "medium", "high", "urgent"]})
        self.assertEqual(kind, "score")
        self.assertEqual(options, ["low", "medium", "high", "urgent"])

    def test_bad_questions_name_the_problem(self):
        for name, question in (
                ("q", {}),                                    # no type
                ("q", {"type": "rank", "criteria": ["a"]}),   # unknown type
                ("q", {"type": "choice", "criteria": []}),    # empty criteria
                ("q", {"type": "choice", "criteria": ["a"]}),  # not a dict
                ("q", {"type": "score", "criteria": "low"}),  # not a list
                ("q", {"type": "score"}),                     # no rubric
                ("q", {"type": "noul", "criteria": ["no"]}),  # not a dict
        ):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    option_texts(name, question)


class BuildAnswerTests(unittest.TestCase):
    def test_noul_reads_the_yes_probability(self):
        answer = build_answer("refund", {"type": "noul"}, "noul",
                              [0.3, 0.7])
        self.assertEqual(answer, {"type": "noul", "noul": 0.7})

    def test_choice_argmaxes_and_maps_probabilities(self):
        question = {"type": "choice",
                    "criteria": {"billing": None, "technical": "d"}}
        answer = build_answer("team", question, "choice", [0.2, 0.8])
        self.assertEqual(answer["choice"], "technical")
        self.assertEqual(answer["confidence"], 0.8)
        self.assertEqual(answer["probabilities"],
                         {"billing": 0.2, "technical": 0.8})

    def test_score_expected_value_and_legend(self):
        question = {"type": "score", "criteria": ["low", "mid", "high"]}
        answer = build_answer("urgency", question, "score", [0.2, 0.5, 0.3])
        self.assertAlmostEqual(answer["score"], 1.1)
        self.assertEqual(answer["confidence"], 0.5)
        self.assertEqual(answer["legend"],
                         {"0": "low", "1": "mid", "2": "high"})
        self.assertEqual(answer["probabilities"],
                         {"0": 0.2, "1": 0.5, "2": 0.3})


class MultiFieldPackTests(unittest.TestCase):
    def pack(self):
        # fake ids are one per character: ctx=10, p one=5, a/b=1 each,
        # p two=5, x/y/z=1 each -> total 25
        return packer.pack_fields(
            FakeTokenizer(), "state text",
            [("p one", ["a", "b"]), ("p two", ["x", "y", "z"])],
            context_tokens=64)

    def test_option_axis_is_padded_to_the_widest_menu(self):
        batch = self.pack()
        self.assertEqual(batch["option_span"].shape, (1, 2, 3, 25))
        self.assertEqual(batch["option_mask"].shape, (1, 2, 3))
        self.assertEqual(batch["option_mask"][0, 0].tolist(),
                         [True, True, False])
        self.assertEqual(batch["option_mask"][0, 1].tolist(),
                         [True, True, True])
        # the padded column owns no positions, so it cannot enter a softmax
        self.assertFalse(batch["option_span"][0, 0, 2].any())

    def test_field_spans_follow_the_prompt_option_layout(self):
        batch = self.pack()
        self.assertEqual(batch["field_span"].shape, (1, 2, 25))
        self.assertEqual(batch["context_span"][0, :10].tolist(), [1.0] * 10)
        self.assertFalse(batch["context_span"][0, 10:].any())
        # [context][field 0 prompt][its options][field 1 prompt][its options]
        self.assertEqual(
            batch["field_span"][0, 0].nonzero().flatten().tolist(),
            list(range(10, 15)))
        self.assertEqual(
            batch["field_span"][0, 1].nonzero().flatten().tolist(),
            list(range(17, 22)))
        self.assertEqual(
            batch["option_span"][0, 0, 1].nonzero().flatten().tolist(), [16])
        self.assertEqual(
            batch["option_span"][0, 1, 0].nonzero().flatten().tolist(), [22])
        self.assertEqual(
            batch["option_span"][0, 1, 2].nonzero().flatten().tolist(), [24])

    def test_every_real_position_is_masked_in(self):
        batch = self.pack()
        self.assertEqual(int(batch["packed_mask"].sum()), 25)
        self.assertEqual(int(batch["option_mask"].sum()), 5)

    def test_no_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            packer.pack_fields(FakeTokenizer(), "s", [], context_tokens=64)


class SingleFieldPackTests(unittest.TestCase):
    def test_pack_batch_keeps_the_benchmark_shape(self):
        # the benches and score() unpack this exact shape; do not change it
        batch = packer.pack_batch(FakeTokenizer(), "state", "p",
                                  ["a", "b"], context_tokens=64)
        self.assertEqual(batch["option_span"].shape, (1, 1, 2, 8))
        self.assertEqual(batch["field_span"].shape, (1, 1, 8))
        self.assertEqual(batch["option_mask"].tolist(), [[[True, True]]])


class TourQuestionsTests(unittest.TestCase):
    def test_typed_questions_cover_the_three_kinds(self):
        self.assertEqual(set(demo.TYPED_QUESTIONS),
                         {"team", "refund", "urgency"})
        self.assertEqual(demo.TYPED_QUESTIONS["team"]["type"], "choice")
        self.assertEqual(demo.TYPED_QUESTIONS["refund"]["type"], "noul")
        self.assertEqual(demo.TYPED_QUESTIONS["urgency"]["type"], "score")

    def test_score_rubric_is_ordered_four_levels(self):
        rubric = demo.TYPED_QUESTIONS["urgency"]["criteria"]
        self.assertEqual([level.split(":")[0] for level in rubric],
                         ["low", "medium", "high", "urgent"])
        self.assertEqual(demo.TYPED_QUESTIONS["urgency"]["criteria"],
                         demo.URGENCY_RUBRIC)


class ParseTypedPayloadTests(unittest.TestCase):
    """--serve mode 'typed': the payload gate in front of answer_typed."""

    def _ok(self):
        return {"mode": "typed", "state": "charged twice",
                "questions": {"refund": {"type": "noul",
                                         "instructions": "refund?"}}}

    def test_good_payload_passes_through(self):
        state, questions = demo.parse_typed_payload(self._ok())
        self.assertEqual(state, "charged twice")
        self.assertEqual(questions, self._ok()["questions"])

    def test_missing_or_blank_state_is_named(self):
        for bad in ({}, {"state": "  "},
                    {"state": 7, "questions": self._ok()["questions"]}):
            with self.assertRaises(ValueError) as ctx:
                demo.parse_typed_payload(bad)
            self.assertIn("state", str(ctx.exception))

    def test_bad_questions_container_is_named(self):
        payload = {"state": "s", "questions": []}
        with self.assertRaises(ValueError) as ctx:
            demo.parse_typed_payload(payload)
        self.assertIn("questions", str(ctx.exception))

    def test_unknown_type_names_the_question(self):
        payload = self._ok()
        payload["questions"]["bad"] = {"type": "tuple", "instructions": "x"}
        with self.assertRaises(ValueError) as ctx:
            demo.parse_typed_payload(payload)
        self.assertIn("'bad'", str(ctx.exception))
        self.assertIn("choice", str(ctx.exception))

    def test_non_string_instructions_is_named(self):
        payload = self._ok()
        payload["questions"]["refund"]["instructions"] = 5
        with self.assertRaises(ValueError) as ctx:
            demo.parse_typed_payload(payload)
        self.assertIn("instructions", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
