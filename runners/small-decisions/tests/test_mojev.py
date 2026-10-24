"""Offline tests for the MoJev module (models/mojev.py + the vendored
packer) — a port of legacy/tests/test_mojev_question_types.py minus the
demo-module classes (the demo surface became the runner payload contract,
tested through predict()).

No checkpoint, no network: the pure question->candidates/answer mappings,
the packer span math against a character-level fake tokenizer, and the
import hygiene. The vendored packer imports torch at module scope, which
this island ships, so torch is NOT blocked here (only transformers, which
must stay out of the mapping functions and the packer)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from conftest import CHOICE

import small_decisions.models.mojev as mojev
from small_decisions.models._vendored import mojev_packer


class FakeTokenizer:
    """Character-level stand-in for the Qwen tokenizer: one integer id per
    character, so every span window is checkable with len()/range()."""

    pad_token_id = 0

    def __call__(self, texts, truncation=False, max_length=None):
        rows = [[(ord(ch) % 251) + 1 for ch in text] for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}


class ImportTests:
    def test_module_imports_pull_no_heavy_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            importlib.reload(mojev)

    def test_packer_imports_without_transformers(self):
        with patch.dict(sys.modules, {"transformers": None}):
            importlib.reload(mojev_packer)


class FieldPromptTests:
    def test_kind_choice_and_options_join(self):
        assert (
            mojev_packer.field_prompt("urgency", "How urgent?", ("high", "low"))
            == "urgency | How urgent? | kind: choice | options: high, low"
        )

    def test_empty_description_is_dropped(self):
        assert mojev_packer.field_prompt("refund", "", ("no", "yes")) == "refund | kind: choice | options: no, yes"

    def test_wide_menus_drop_the_options_list(self):
        options = tuple(f"o{i}" for i in range(9))
        prompt = mojev_packer.field_prompt("t", "d", options)
        assert "options:" not in prompt


class CandidateOrderTests:
    def test_sort_and_unsort_round_trip(self):
        options = ["c", "a", "b"]
        order = mojev_packer.sort_candidates(options)
        assert [options[i] for i in order] == ["a", "b", "c"]
        # unsort maps sorted-position results back to caller order
        assert mojev_packer.unsort(order, [0.1, 0.2, 0.3]) == [0.3, 0.1, 0.2]


class OptionTextsTests:
    def test_choice_descriptions_become_label_colon_description(self):
        options, kind = mojev.option_texts("team", {
            "type": "choice",
            "criteria": {"billing": "payments", "technical": None, "account": ""}})
        assert kind == "choice"
        assert options == ["billing: payments", "technical", "account"]

    def test_noul_defaults_to_no_then_yes(self):
        options, kind = mojev.option_texts("refund", {"type": "noul"})
        assert kind == "noul"
        assert options == ["no", "yes"]

    def test_noul_candidates_stay_false_then_true(self):
        options, _ = mojev.option_texts("refund", {
            "type": "noul",
            "criteria": {"true": "money back", "false": "nothing owed"}})
        assert options == ["nothing owed", "money back"]

    def test_non_string_criteria_render_as_json(self):
        options, _ = mojev.option_texts("s", {"type": "score", "criteria": [1, 2.5]})
        assert options == ["1", "2.5"]

    def test_score_keeps_rubric_order(self):
        options, kind = mojev.option_texts("urgency", {
            "type": "score",
            "criteria": ["low", "medium", "high", "urgent"]})
        assert kind == "score"
        assert options == ["low", "medium", "high", "urgent"]

    def test_bad_questions_name_the_problem(self):
        for name, question in (
            ("q", {}),  # no type
            ("q", {"type": "rank", "criteria": ["a"]}),  # unknown type
            ("q", {"type": "choice", "criteria": []}),  # empty criteria
            ("q", {"type": "choice", "criteria": ["a"]}),  # not a dict
            ("q", {"type": "score", "criteria": "low"}),  # not a list
            ("q", {"type": "score"}),  # no rubric
            ("q", {"type": "noul", "criteria": ["no"]}),  # not a dict
        ):
            with pytest.raises(ValueError):
                mojev.option_texts(name, question)

    def test_noul_rejects_identical_false_true_texts(self):
        question = {"type": "noul", "criteria": {"false": "x", "true": "x"}}
        with pytest.raises(ValueError, match="must differ"):
            mojev.option_texts("refund", question)

    def test_choice_rejects_duplicate_rendered_candidates(self):
        # "x: y" (bare key) renders to the same text as "x" described
        # as "y" - the model cannot tell the candidates apart
        question = {"type": "choice", "criteria": {"x: y": None, "x": "y"}}
        with pytest.raises(ValueError, match="distinct"):
            mojev.option_texts("team", question)


class BuildAnswerTests:
    def test_noul_reads_the_yes_probability(self):
        answer = mojev.build_answer("refund", {"type": "noul"}, "noul", [0.3, 0.7])
        assert answer == {"type": "noul", "noul": 0.7}

    def test_choice_argmaxes_and_maps_probabilities(self):
        question = {"type": "choice", "criteria": {"billing": None, "technical": "d"}}
        answer = mojev.build_answer("team", question, "choice", [0.2, 0.8])
        assert answer["choice"] == "technical"
        assert answer["confidence"] == 0.8
        assert answer["probabilities"] == {"billing": 0.2, "technical": 0.8}

    def test_score_expected_value_and_legend(self):
        question = {"type": "score", "criteria": ["low", "mid", "high"]}
        answer = mojev.build_answer("urgency", question, "score", [0.2, 0.5, 0.3])
        assert answer["score"] == pytest.approx(1.1)
        assert answer["confidence"] == 0.5
        assert answer["legend"] == {"0": "low", "1": "mid", "2": "high"}
        assert answer["probabilities"] == {"0": 0.2, "1": 0.5, "2": 0.3}


class MultiFieldPackTests:
    def pack(self):
        # fake ids are one per character: ctx=10, p one=5, a/b=1 each,
        # p two=5, x/y/z=1 each -> total 25
        return mojev_packer.pack_fields(
            FakeTokenizer(), "state text",
            [("p one", ["a", "b"]), ("p two", ["x", "y", "z"])],
            context_tokens=64)

    def test_option_axis_is_padded_to_the_widest_menu(self):
        batch = self.pack()
        assert tuple(batch["option_span"].shape) == (1, 2, 3, 25)
        assert tuple(batch["option_mask"].shape) == (1, 2, 3)
        assert batch["option_mask"][0, 0].tolist() == [True, True, False]
        assert batch["option_mask"][0, 1].tolist() == [True, True, True]
        # the padded column owns no positions, so it cannot enter a softmax
        assert not batch["option_span"][0, 0, 2].any()

    def test_field_spans_follow_the_prompt_option_layout(self):
        batch = self.pack()
        assert tuple(batch["field_span"].shape) == (1, 2, 25)
        assert batch["context_span"][0, :10].tolist() == [1.0] * 10
        assert not batch["context_span"][0, 10:].any()
        # [context][field 0 prompt][its options][field 1 prompt][its options]
        assert batch["field_span"][0, 0].nonzero().flatten().tolist() == list(range(10, 15))
        assert batch["field_span"][0, 1].nonzero().flatten().tolist() == list(range(17, 22))
        assert batch["option_span"][0, 0, 1].nonzero().flatten().tolist() == [16]
        assert batch["option_span"][0, 1, 0].nonzero().flatten().tolist() == [22]
        assert batch["option_span"][0, 1, 2].nonzero().flatten().tolist() == [24]

    def test_every_real_position_is_masked_in(self):
        batch = self.pack()
        assert int(batch["packed_mask"].sum()) == 25
        assert int(batch["option_mask"].sum()) == 5

    def test_no_fields_is_rejected(self):
        with pytest.raises(ValueError):
            mojev_packer.pack_fields(FakeTokenizer(), "s", [], context_tokens=64)


class SingleFieldPackTests:
    def test_pack_batch_keeps_the_benchmark_shape(self):
        # the benches and score() unpack this exact shape; do not change it
        batch = mojev_packer.pack_batch(FakeTokenizer(), "state", "p", ["a", "b"], context_tokens=64)
        assert tuple(batch["option_span"].shape) == (1, 1, 2, 8)
        assert tuple(batch["field_span"].shape) == (1, 1, 8)
        assert batch["option_mask"].tolist() == [[[True, True]]]


class PredictTests:
    def test_choice_mapping_through_a_stubbed_answer_typed(self):
        def answer_typed(state, questions):
            return (
                {"q": {"type": "choice", "choice": "positive", "confidence": 0.9,
                       "probabilities": {"positive": 0.9, "negative": 0.1}}},
                {"input_tokens": 42, "output_tokens": 0},
            )

        mojev._ENGINE["answer_typed"] = answer_typed
        try:
            result = mojev.predict(CHOICE)
        finally:
            mojev._ENGINE.clear()
        assert result == {"prediction": "positive", "prob": 0.9}

    def test_catalog_identity(self):
        assert mojev.MODEL_ID == "molemo-lab/mojev"
        assert mojev.VERSION == "mojev@0c8695b6252f4205907433d4e196a94f032e60c3"
