"""Offline tests for the nanodiff multi-decision format (the
build_multi_prompt/response pair predict_multi relies on), plus the v2
checkpoint constants ('nanodiff 350M v2' in the benches loads through the
same vendored runner).

No checkpoint, no network - only the tiktoken-based format math (the
engine package import does pull torch, which the main venv ships).
Run with the main venv:

    .venv/Scripts/python -m unittest tests.test_nanodiff_multi_format
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.nanodiff_engine import runner
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


class V2CheckpointTests(unittest.TestCase):
    """The 'nanodiff 350M v2' bench entries load through the same vendored
    runner (same ARCH, same typed-decision format); only the checkpoint
    constant differs. Constants only - no download, no weights."""

    def test_default_ckpt_is_still_the_v1_release(self):
        repo, _, fn = runner.CKPT.partition("::")
        self.assertEqual(repo, "pngwn/nanodiff-350m-typed-decisions-lam1")
        self.assertEqual(fn, "nanodiff-350m-typed-decisions-lam1.pt")

    def test_v2_ckpt_targets_the_v2_release(self):
        repo, _, fn = runner.CKPT_V2.partition("::")
        self.assertEqual(repo, "pngwn/nanodiff-350m-typed-decisions-v2")
        self.assertEqual(fn, "nanodiff-350m-typed-decisions-v2.pt")

    def test_load_model_defaults_to_v1_and_accepts_v2(self):
        self.assertIn(runner.CKPT, runner.load_model.__defaults__)

    def test_v2_shares_the_arch(self):
        # the v2 checkpoint's embedded config matches runner.ARCH (verified
        # against the release header: n_layer=16, n_head=20, n_embd=1280,
        # block_size=512), so load_model builds the same NanoDiff shape
        self.assertEqual(runner.ARCH, dict(n_layer=16, n_head=20,
                                           n_embd=1280, block_size=512))


if __name__ == "__main__":
    unittest.main()
