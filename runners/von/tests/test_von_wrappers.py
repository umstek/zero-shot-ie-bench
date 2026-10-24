"""Judge/rate wrapper + row tests, ported from legacy
tests/test_von_judge_rate.py (JudgeWrapperTests, RateWrapperTests,
RowTests).

The wrappers must stay thin over the pinned SDK — evaluate_noul /
evaluate_score / evaluate_choice with the backend's calibration
temperature, one shared preloaded backend — and the answer->row mappings
(ported from demos/von_demo.py) must keep the 0.5 verdict threshold and
the score/legend/probabilities passthrough.
"""

import sys
from unittest.mock import Mock, patch

from conftest import VON_TYPES_STUB, choice_answer, noul_answer, score_answer, sdk_stubs

import von_runner.von as von_module
from von_runner.von import judgment_row, load_von_decider, load_von_judge, load_von_rate, rating_row


class TestJudgeWrappers:
    def test_judge_calls_evaluate_noul_with_the_backend_temperature(self):
        backend = Mock(_default_temp=0.7)
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}):
            judge = load_von_judge(backend)
            answer = judge(state="ticket text", instructions="Is it a refund?")
        backend.evaluate_noul.assert_called_once_with(
            "judgment", "ticket text", {"instructions": "Is it a refund?"}, temperature=0.7
        )
        assert answer is backend.evaluate_noul.return_value

    def test_judge_without_a_backend_loads_one(self):
        sentinel = Mock(_default_temp=1.0)
        with (
            patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}),
            patch.object(von_module, "load_von_backend", return_value=sentinel) as load,
        ):
            judge = load_von_judge()
            judge(state="s", instructions="i")
        load.assert_called_once_with()
        sentinel.evaluate_noul.assert_called_once()


class TestRateWrappers:
    def test_rate_calls_evaluate_score_with_the_ordered_rubric(self):
        backend = Mock(_default_temp=1.0)
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}):
            rate = load_von_rate(backend)
            answer = rate(state="ticket text", rubric=["low", "high"], instructions="How urgent?")
        backend.evaluate_score.assert_called_once_with(
            "rating",
            "ticket text",
            {"instructions": "How urgent?", "criteria": ["low", "high"]},
            temperature=1.0,
        )
        assert answer is backend.evaluate_score.return_value

    def test_wrappers_share_one_preloaded_backend(self):
        backend = Mock(_default_temp=1.0)
        with (
            patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}),
            patch.object(von_module, "load_von_backend") as load,
        ):
            load_von_judge(backend)
            load_von_rate(backend)
            load_von_decider(backend)
        load.assert_not_called()

    def test_decider_wraps_evaluate_choice_with_criteria_dict(self):
        backend = Mock(_default_temp=0.9)
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}):
            decide = load_von_decider(backend)
            answer = decide(
                state="text", choices={"positive": "positive attitude"}, instructions="sentiment"
            )
        backend.evaluate_choice.assert_called_once_with(
            "decision",
            "text",
            {"instructions": "sentiment", "criteria": {"positive": "positive attitude"}},
            temperature=0.9,
        )
        assert answer is backend.evaluate_choice.return_value


class TestRows:
    def test_judgment_row_maps_noul_and_derives_the_verdict(self):
        assert judgment_row(noul_answer(0.87)) == {"noul": 0.87, "verdict": "yes"}
        assert judgment_row(noul_answer(0.2)) == {"noul": 0.2, "verdict": "no"}
        # the 0.5 threshold itself counts as yes
        assert judgment_row(noul_answer(0.5))["verdict"] == "yes"

    def test_rating_row_maps_score_confidence_legend_probabilities(self):
        answer = score_answer(2.4, 0.4, {"0": "low", "1": "high"}, {"0": 0.1, "1": 0.9})
        assert rating_row(answer) == {
            "score": 2.4,
            "confidence": 0.4,
            "legend": {"0": "low", "1": "high"},
            "probabilities": {"0": 0.1, "1": 0.9},
        }

    def test_row_maps_choice_probabilities_confidence(self):
        answer = choice_answer("positive", {"positive": 0.9, "negative": 0.1}, 0.9)
        assert von_module.row(answer) == {
            "choice": "positive",
            "probabilities": {"positive": 0.9, "negative": 0.1},
            "confidence": 0.9,
        }


def test_loaded_wrappers_read_the_stubbed_sdk_path():
    # end-to-end over the fake dependencies: load_von_judge() with no
    # backend goes through snapshot_download + OptionMarkerBackend just
    # like load_von_backend does (the loading tests pin the details).
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as checkpoint:
        Path(checkpoint, "option_marker.pt").touch()
        modules, backend, _, _ = sdk_stubs(checkpoint=checkpoint)
        with patch.dict(sys.modules, modules):
            judge = load_von_judge()
            judge(state="s", instructions="i")
        backend.evaluate_noul.assert_called_once()
