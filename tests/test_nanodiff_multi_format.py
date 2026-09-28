"""Offline tests for the nanodiff multi-decision format (the
build_multi_prompt/response pair predict_multi relies on).

No checkpoint, no network - only the tiktoken-based format math (the
engine package import does pull torch, which the main venv ships).
Run with the main venv:

    .venv/Scripts/python -m unittest tests.test_nanodiff_multi_format
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.nanodiff_engine.decision_format import (
    build_multi_prompt, build_multi_response, encode_example)


class MultiFormatTests(unittest.TestCase):
    def test_multi_prompt_numbers_questions_and_inline_options(self):
        prompt = build_multi_prompt("the state", [
            ("first question?", ["a1", "a2"]),
            ("second question?", ["b1", "b2", "b3"]),
        ])
        self.assertIn("### Questions:", prompt)
        self.assertIn("1) first question? [A) a1, B) a2]", prompt)
        self.assertIn("2) second question? [A) b1, B) b2, C) b3]", prompt)

    def test_multi_response_offsets_point_at_the_answer_letters(self):
        response, offsets = build_multi_response([0, 2, 1])
        self.assertEqual(response, "1) A 2) C 3) B")
        self.assertEqual([response[o] for o in offsets], ["A", "C", "B"])

    def test_encode_example_masks_each_letter_as_its_own_token(self):
        questions = [("q one?", ["a", "b"]), ("q two?", ["x", "y", "z"]),
                     ("q three?", ["m", "n"])]
        response, offsets = build_multi_response([0] * len(questions))
        prompt = build_multi_prompt("state text", questions)
        _, response_ids, answer_tokens = encode_example(prompt, response,
                                                        offsets)
        self.assertEqual(len(answer_tokens), len(questions))
        # each answer slot is distinct and inside the response window
        self.assertEqual(len(set(answer_tokens)), len(answer_tokens))
        for token in answer_tokens:
            self.assertLess(token, len(response_ids))

    def test_multi_prompt_stays_under_the_prompt_budget(self):
        questions = [(f"question number {i}?", ["opt one", "opt two"])
                     for i in range(8)]
        prompt = build_multi_prompt("s" * 200, questions)
        # PROMPT_LEN is 480; encode_example hard-rejects oversize prompts
        encode_example(prompt, *build_multi_response([0] * 8))


if __name__ == "__main__":
    unittest.main()
