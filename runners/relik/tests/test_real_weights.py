"""The tests that load the real relik checkpoints.

Skipped unless the checkpoint is already in the local HF cache
(cached_snapshot() uses local_files_only=True, so CI never downloads
weights). Marked slow (docs/plan/02: "tests that load real weights are
marked slow and run locally on a machine that can hold the model"). To
seed the caches locally:

    uv run python -c "from relik_runner.relik import load_relik; load_relik()"
    uv run python -c "from relik_runner.relik import load_relik, NER_MODEL_ID; load_relik(NER_MODEL_ID)"

The snapshot_download inside the gate is pinned to local-only mode here,
so even these tests cannot pull weights from the network.
"""

from __future__ import annotations

from functools import partial

import pytest

from relik_runner.relik import (
    CATALOG_MODEL_ID,
    MODEL_ID,
    NER_MODEL_ID,
    cached_snapshot,
    load_relik,
    predict,
    provenance,
)

pytestmark = pytest.mark.slow

# the canonical sample sentences (legacy demos/demo_relik.py showcases 1+5)
TRIPLET_TEXT = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
NER_TEXT = "Minecraft is a video game developed by Mojang Studios."


def _require_cached_checkpoint(model_id: str, monkeypatch: pytest.MonkeyPatch) -> str:
    snapshot = cached_snapshot(model_id)
    if snapshot is None:
        pytest.skip(
            f"the {model_id} checkpoint is not in the local HF cache; seed "
            "it with load_relik() once, locally"
        )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        partial(huggingface_hub.snapshot_download, local_files_only=True),
    )
    return snapshot


def test_real_relation_pipeline_loads_and_annotates(
    monkeypatch: pytest.MonkeyPatch,
):
    _require_cached_checkpoint(MODEL_ID, monkeypatch)
    relik = load_relik(MODEL_ID)  # the benched deployment, csv shim + all
    # num_workers=0, like the engine's _run: the retriever's default 4
    # worker processes cannot re-install the csv shim under spawn and hang
    # the DataLoader when a child dies
    out = relik(TRIPLET_TEXT, num_workers=0)
    # the reader labels its spans --NME-- (RE-only checkpoint) and scores
    # span pairs against retrieved Wikidata properties
    for s in out.spans:
        assert s.text == TRIPLET_TEXT[s.start:s.end]
    for t in out.triplets:
        assert 0.0 <= t.confidence <= 1.0
        assert t.subject.text and t.object.text


def test_real_predict_relation_rows_match_the_legacy_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    _require_cached_checkpoint(MODEL_ID, monkeypatch)
    out = predict(CATALOG_MODEL_ID, {"input": TRIPLET_TEXT, "question": {}})
    assert out["error"] is None and out["prob"] is None
    for row in out["prediction"]:
        assert set(row) == {"subject", "relation", "object", "confidence"}
        assert set(row["subject"]) == {"start", "end", "label", "text"}


def test_real_ner_sibling_loads_through_the_two_fixes(
    monkeypatch: pytest.MonkeyPatch,
):
    # the joint NER+relation config needs BOTH relik-1.0.7 load fixes
    # (GoldenSillyRetriever's nn.Module init, the span index tolerance);
    # a load here is the regression test for them
    _require_cached_checkpoint(NER_MODEL_ID, monkeypatch)
    relik = load_relik(NER_MODEL_ID)
    out = relik(NER_TEXT, num_workers=0)  # see the relation test above
    for s in out.spans:
        assert s.text == NER_TEXT[s.start:s.end]
        # typed from the fixed 10-type Wikipedia set, never --NME--
        assert s.label != "--NME--"


def test_real_provenance_is_stable_for_run_records():
    assert provenance()["model"] == MODEL_ID
    assert provenance(NER_MODEL_ID)["model"] == NER_MODEL_ID
    assert provenance()["backend"] == (
        "retriever-reader (faiss index + DeBERTa reader)"
    )
