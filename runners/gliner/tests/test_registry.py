"""The island's registry: the catalog slice it must serve, and the track
resolution that stands in for the protocol's (non-echoed) track."""

from __future__ import annotations

import pytest

from gliner_runner import models
from gliner_runner.models import SERVED_MODELS, MODELS, ModelSpec, resolve_track

# catalog/models/{knowledgator,jackboyla,convaiinnovations}/*.yaml — the
# deployments with runner: gliner, id -> (version id, hf_repo).
CATALOG = {
    "gliformer-base-v1.local": ("gliformer-base-v1@unversioned",
                                "knowledgator/gliformer-base-v1"),
    "gliformer-large-v1.local": ("gliformer-large-v1@unversioned",
                                 "knowledgator/gliformer-large-v1"),
    "gliner-relex-multi-v1.0.local": ("gliner-relex-multi-v1.0@unversioned",
                                      "knowledgator/gliner-relex-multi-v1.0"),
    "glirel-large-v0.local": ("glirel-large-v0@unversioned",
                              "jackboyla/glirel-large-v0"),
    "gliclass-base-v3.0.local": ("gliclass-base-v3.0@unversioned",
                                 "knowledgator/gliclass-base-v3.0"),
    "gliclass-modern-base-v3.0.local": ("gliclass-modern-base-v3.0@unversioned",
                                        "knowledgator/gliclass-modern-base-v3.0"),
    "gliclass-large-v3.0.local": ("gliclass-large-v3.0@unversioned",
                                  "knowledgator/gliclass-large-v3.0"),
    "laya.local": ("laya@unversioned", "convaiinnovations/laya"),
    "laya.router": ("laya@unversioned", "convaiinnovations/laya"),
    "laya-typed-decisions.local": ("laya-typed-decisions@unversioned",
                                   "convaiinnovations/laya-typed-decisions"),
}

# catalog claims per deployment (claims.tracks; relex's classify row is the
# census-row multitask mapping, see models/gliner_relex.py).
CAPABILITIES = {
    "gliformer-base-v1.local": {"classify", "ner"},
    "gliformer-large-v1.local": {"classify", "ner"},
    "gliner-relex-multi-v1.0.local": {"ner", "relation", "classify"},
    "glirel-large-v0.local": {"relation"},
    "gliclass-base-v3.0.local": {"classify"},
    "gliclass-modern-base-v3.0.local": {"classify"},
    "gliclass-large-v3.0.local": {"classify"},
    "laya.local": {"decision"},
    "laya.router": {"decision"},
    "laya-typed-decisions.local": {"decision"},
}


class TestRegistry:
    def test_every_catalog_deployment_is_served(self):
        assert set(MODELS) == set(CATALOG)

    def test_versions_and_hf_repos_match_the_catalog(self):
        for model_id, (version, hf_repo) in CATALOG.items():
            assert MODELS[model_id].version == version
            assert MODELS[model_id].hf_repo == hf_repo

    def test_served_models_shape_for_create_app(self):
        # the protocol validates entries against {model_id, version, runtime}
        assert len(SERVED_MODELS) == len(CATALOG)
        for entry in SERVED_MODELS:
            assert set(entry) == {"model_id", "version", "runtime"}
            assert entry["runtime"]["engine"] == "torch"
            assert entry["runtime"]["device"] == "cpu"

    def test_every_capability_has_an_implementation(self):
        for spec in MODELS.values():
            assert set(spec.tracks) == CAPABILITIES[spec.model_id]
            for track in spec.tracks:
                assert callable(getattr(spec.module, f"run_{track}"))
            assert callable(getattr(spec.module, spec.loader))

    def test_model_ids_are_unique(self):
        assert len(SERVED_MODELS) == len({spec.model_id for spec in MODELS.values()})

    def test_router_uses_the_router_loader(self):
        assert MODELS["laya.router"].loader == "load_router"
        assert MODELS["laya.local"].loader == "load"


class TestResolveTrack:
    """The v0 question shapes decide the track when the protocol does not
    echo it; a model's capabilities constrain the answer."""

    SPEC = ModelSpec("x", "x@unversioned", "org/x", models.gliclass,
                     ("classify", "ner"))

    def test_jev_typed_question_is_decision(self):
        for qtype in ("choice", "score", "noul"):
            question = {"type": qtype, "instructions": "i", "criteria": {}}
            assert resolve_track(self.SPEC, question) == "decision"

    def test_relation_keys_ask_for_relations(self):
        spec = ModelSpec("x", "x@unversioned", "org/x", models.gliner_relex,
                         ("ner", "relation", "classify"))
        for key in ("relations", "relation_labels", "ner"):
            assert resolve_track(spec, {key: []}) == "relation"

    def test_relation_keys_fall_back_when_the_model_cannot_do_relations(self):
        # the gliformer-shaped spec has no relation capability: an empty
        # relations key falls through to the first capability
        assert resolve_track(self.SPEC, {"relations": []}) == "classify"

    def test_dict_labels_are_classification(self):
        assert resolve_track(self.SPEC, {"labels": {"a": "desc"}}) == "classify"

    def test_list_labels_are_ner_when_capable(self):
        assert resolve_track(self.SPEC, {"labels": ["a", "b"]}) == "ner"

    def test_list_labels_fall_back_to_classification_on_a_classify_only_model(self):
        spec = ModelSpec("x", "x@unversioned", "org/x", models.gliclass, ("classify",))
        assert resolve_track(spec, {"labels": ["a"]}) == "classify"

    def test_unrecognised_question_falls_back_to_the_first_capability(self):
        spec = ModelSpec("x", "x@unversioned", "org/x", models.laya, ("decision",))
        assert resolve_track(spec, {}) == "decision"

    def test_relation_only_model_stays_on_relation(self):
        spec = ModelSpec("x", "x@unversioned", "org/x", models.glirel, ("relation",))
        assert resolve_track(spec, {"labels": ["a"]}) == "relation"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
