"""Offline tests for the nanodiff multi-decision format (the
build_multi_prompt/response pair predict_multi relies on), plus the v2
checkpoint constants ('nanodiff 350M v2' in the benches loads through the
same engine module).

Ported from legacy/tests/test_nanodiff_multi_format.py — the unittest
classes became pytest classes, the imports moved to the island package,
the test bodies are unchanged. No checkpoint, no network — only the
tiktoken-based format math (importing the engine pulls torch, a hard dep
of this island, but never touches the HF cache).
"""

from nanodiff_runner import engine
from nanodiff_runner.decision_format import (
    build_multi_prompt,
    build_multi_response,
    encode_example,
)


class TestMultiFormat:
    def test_multi_prompt_numbers_questions_and_inline_options(self):
        prompt = build_multi_prompt(
            "the state",
            [
                ("first question?", ["a1", "a2"]),
                ("second question?", ["b1", "b2", "b3"]),
            ],
        )
        assert "### Questions:" in prompt
        assert "1) first question? [A) a1, B) a2]" in prompt
        assert "2) second question? [A) b1, B) b2, C) b3]" in prompt

    def test_multi_response_offsets_point_at_the_answer_letters(self):
        response, offsets = build_multi_response([0, 2, 1])
        assert response == "1) A 2) C 3) B"
        assert [response[o] for o in offsets] == ["A", "C", "B"]

    def test_encode_example_masks_each_letter_as_its_own_token(self):
        questions = [
            ("q one?", ["a", "b"]),
            ("q two?", ["x", "y", "z"]),
            ("q three?", ["m", "n"]),
        ]
        response, offsets = build_multi_response([0] * len(questions))
        prompt = build_multi_prompt("state text", questions)
        _, response_ids, answer_tokens = encode_example(prompt, response, offsets)
        assert len(answer_tokens) == len(questions)
        # each answer slot is distinct and inside the response window
        assert len(set(answer_tokens)) == len(answer_tokens)
        for token in answer_tokens:
            assert token < len(response_ids)

    def test_multi_prompt_stays_under_the_prompt_budget(self):
        questions = [(f"question number {i}?", ["opt one", "opt two"]) for i in range(8)]
        prompt = build_multi_prompt("s" * 200, questions)
        # PROMPT_LEN is 480; encode_example hard-rejects oversize prompts
        encode_example(prompt, *build_multi_response([0] * 8))


class TestV2Checkpoint:
    """The 'nanodiff 350M v2' bench entries load through the same engine
    module (same ARCH, same typed-decision format); only the checkpoint
    constant differs. Constants only — no download, no weights."""

    def test_default_ckpt_is_still_the_v1_release(self):
        repo, _, fn = engine.CKPT.partition("::")
        assert repo == "pngwn/nanodiff-350m-typed-decisions-lam1"
        assert fn == "nanodiff-350m-typed-decisions-lam1.pt"

    def test_v2_ckpt_targets_the_v2_release(self):
        repo, _, fn = engine.CKPT_V2.partition("::")
        assert repo == "pngwn/nanodiff-350m-typed-decisions-v2"
        assert fn == "nanodiff-350m-typed-decisions-v2.pt"

    def test_load_model_defaults_to_v1_and_accepts_v2(self):
        defaults = engine.load_model.__defaults__ or ()
        assert engine.CKPT in defaults

    def test_v2_shares_the_arch(self):
        # the v2 checkpoint's embedded config matches engine.ARCH (verified
        # against the release header: n_layer=16, n_head=20, n_embd=1280,
        # block_size=512), so load_model builds the same NanoDiff shape
        assert engine.ARCH == dict(n_layer=16, n_head=20, n_embd=1280, block_size=512)

    def test_served_model_ids_are_the_checkpoint_repos(self):
        # the wire identity (server.py) resolves through the same constants
        assert engine.repo_of(engine.CKPT) == "pngwn/nanodiff-350m-typed-decisions-lam1"
        assert engine.repo_of(engine.CKPT_V2) == "pngwn/nanodiff-350m-typed-decisions-v2"

    def test_provenance_is_stable_for_run_records(self):
        record = engine.provenance(engine.CKPT)
        assert record["model"] == "pngwn/nanodiff-350m-typed-decisions-lam1"
        assert record["checkpoint"] == "nanodiff-350m-typed-decisions-lam1.pt"
        assert record["format"] == "typed-decisions"
