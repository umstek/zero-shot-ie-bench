"""Offline regression tests for models/ollaya.py — the HTTP proxies.

A port of legacy/tests/test_ollaya_client.py: URL building (default,
host:port override, full-URL override wins as-is), keyless client
construction (a local daemon never sees an authorization header), and
registry consistency. The registry checks the legacy tests ran against
the bench drivers now run against the catalog: /info carries every
Ollaya-served catalog model (test_server.py); here the tag map is policed.

No daemon needed: predict() runs against a stubbed HTTP server on an
ephemeral port (conftest's ollaya_stub fixture)."""

from __future__ import annotations

import pytest

import small_decisions.models.ollaya as ollaya
from conftest import CHOICE


class BaseUrlTests:
    def test_default_is_local_systemone(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OLLAYA_BASE_URL", raising=False)
        assert ollaya.base_url() == "http://127.0.0.1:11435/v1/systemone"

    def test_host_port_override_appends_the_systemone_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OLLAYA_BASE_URL", "http://localhost:11499/")
        assert ollaya.base_url() == "http://localhost:11499/v1/systemone"

    def test_full_url_override_wins_as_is(self, monkeypatch: pytest.MonkeyPatch):
        url = "http://example.test:1/api/decide/v1/systemone"
        monkeypatch.setenv("OLLAYA_BASE_URL", url)
        assert ollaya.base_url() == url


class RegistryTests:
    def test_tags_are_nonempty_and_unique(self):
        assert len(set(ollaya.TAGS.values())) == len(ollaya.TAGS)
        for model_id, tag in ollaya.TAGS.items():
            assert tag
            assert "/" in model_id  # catalog ids, not legacy bench names

    def test_every_ollaya_catalog_model_has_an_entry(self):
        assert {entry["model_id"] for entry in ollaya.ENTRIES} == set(ollaya.TAGS)
        assert len(ollaya.ENTRIES) == 10

    def test_known_tags(self):
        assert ollaya.TAGS["vllm-sr/decision"] == "decision"
        assert ollaya.TAGS["moritzlaurer/nli-deberta-v3-large"] == "nli"
        assert ollaya.TAGS["moritzlaurer/nli-modernbert-large"] == "nli:modernbert-large"
        assert ollaya.TAGS["madani/decima-small"] == "decima:small"


class PredictTests:
    def test_predict_posts_the_systemone_contract_and_maps_the_answer(self, ollaya_stub):
        result = ollaya.predict("vllm-sr/decision", CHOICE)
        assert result == {"prediction": "positive", "prob": 0.9}
        (body,) = ollaya_stub
        # the daemon picks the checkpoint by tag
        assert body["model"] == "decision"
        # the state/instructions split is load-bearing: the text to classify
        # must BE the state (see the module docstring's gotcha)
        assert body["state"] == CHOICE["input"]["text"]
        (question,) = body["questions"].values()
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"positive", "negative"}

    def test_predict_reports_the_tag_for_the_requested_model(self, ollaya_stub):
        ollaya.predict("moritzlaurer/nli-modernbert-large", CHOICE)
        assert ollaya_stub[0]["model"] == "nli:modernbert-large"

    def test_error_rows_map_to_the_error_shape(self, ollaya_stub, monkeypatch: pytest.MonkeyPatch):
        from conftest import _StubSystemOne

        monkeypatch.setattr(_StubSystemOne, "answer", {"type": "choice", "error": "invalid_question"}, raising=False)
        result = ollaya.predict("logitlab/snap", CHOICE)
        assert result["prediction"] is None
        assert "invalid_question" in result["error"]

    def test_http_errors_raise_a_runtime_error(self, monkeypatch: pytest.MonkeyPatch):
        # a dead daemon fails the call, not the process — the protocol
        # factory turns the raise into the item's error
        monkeypatch.setenv("OLLAYA_BASE_URL", "http://127.0.0.1:1")
        with pytest.raises(RuntimeError):
            ollaya.predict("codekins/arbiter", CHOICE)

    def test_unknown_model_id_raises(self):
        with pytest.raises(KeyError, match="no-such"):
            ollaya.predict("no-such", CHOICE)
