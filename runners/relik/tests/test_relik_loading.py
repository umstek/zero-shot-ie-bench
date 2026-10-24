"""Loading tests, ported from legacy/tests/test_relik_loader_fixes.py.

The engine (relik_runner/relik.py, port of legacy engines/relik_client.py)
must keep: the Windows csv shim installed at import (no-arg queries stay
queries; sys.maxsize never raises), the two relik-1.0.7 load fixes
(GoldenSillyRetriever's missing nn.Module init, the index's documents +
junk-kwargs intolerance) applied idempotently, load_relik() with no
argument forwarding exactly MODEL_ID while the sibling passes
NER_MODEL_ID, and a module that imports without torch / relik / the hub.

Nothing here imports real relik: tests/fake_relik.py replicates the two
defects the engine patches, so the patch functions themselves are
exercised.
"""

import csv
import importlib
import sys
from unittest.mock import patch

from fake_relik import FakeListConfig, install_fake_relik, uninstall_fake_relik

import relik_runner.relik as relik_module
from relik_runner.relik import MODEL_ID, NER_MODEL_ID, provenance


def test_default_provenance_is_the_benched_model():
    assert provenance()["model"] == MODEL_ID


def test_ner_override_names_the_sibling():
    assert provenance(NER_MODEL_ID)["model"] == NER_MODEL_ID


def test_the_two_models_are_distinct_repos():
    assert MODEL_ID != NER_MODEL_ID
    assert MODEL_ID.startswith("relik-ie/")
    assert NER_MODEL_ID.startswith("relik-ie/")


def test_importing_the_engine_installs_the_shim():
    assert csv.field_size_limit is relik_module._safe_field_size_limit


def test_no_arg_query_stays_a_query():
    # the shim's sentinel dance: a no-arg call must be a query, never a
    # set (forwarding a default like -1 would set the limit to -1)
    before = csv.field_size_limit()
    assert csv.field_size_limit() == before


def test_sys_maxsize_never_raises_and_clamps_on_windows():
    original = csv.field_size_limit()
    try:
        # the exact call relik's document.py makes at import — the crash
        # this shim exists for (SapienzaNLP/relik#39)
        returned = csv.field_size_limit(sys.maxsize)
        assert isinstance(returned, int)
        if sys.platform == "win32":
            # Windows' C long is 32-bit: the shim clamped to INT32_MAX
            assert csv.field_size_limit() == 2**31 - 1
    finally:
        csv.field_size_limit(original)


def test_missing_relik_names_the_shim_and_the_sync(monkeypatch):
    # None in sys.modules forces the ImportError even where relik IS
    # installed (e.g. a suite run inside this island's venv); monkeypatch
    # restores the real entry afterwards
    monkeypatch.setitem(sys.modules, "relik", None)
    raised = None
    try:
        relik_module.load_relik()
    except RuntimeError as exc:
        raised = exc
    assert raised is not None
    assert "csv shim" in str(raised)
    assert "uv sync" in str(raised)


def test_default_load_forwards_the_benched_model_verbatim():
    calls = []
    _silly, _index, _document, _store, saved = install_fake_relik(calls)
    try:
        relik_module.load_relik()
        assert calls == [(MODEL_ID, "cpu")]
    finally:
        uninstall_fake_relik(saved)


def test_ner_load_forwards_the_sibling_model():
    calls = []
    _silly, _index, _document, _store, saved = install_fake_relik(calls)
    try:
        relik_module.load_relik(NER_MODEL_ID)
        assert calls == [(NER_MODEL_ID, "cpu")]
    finally:
        uninstall_fake_relik(saved)


def test_silly_retriever_eval_broken_before_fixed_after_load():
    calls = []
    silly, _index, _document, _store, saved = install_fake_relik(calls)
    try:
        # the replicated 1.0.7 crash: nn.Module bookkeeping missing
        raised = None
        try:
            silly(["person"]).eval()
        except AttributeError as exc:
            raised = exc
        assert raised is not None
        relik_module.load_relik()  # applies the nn.Module-init fix
        retriever = silly(["person", "location"])
        retriever.eval()  # must no longer raise
        retriever.train(False)
        assert retriever.documents == ["person", "location"]
    finally:
        uninstall_fake_relik(saved)


def test_index_fix_normalizes_config_documents_and_junk_kwargs():
    calls = []
    _silly, index, document, _store, saved = install_fake_relik(calls)
    try:
        relik_module.load_relik()  # applies the index fix
        built = index(documents=FakeListConfig(["person", "location"]),
                      use_faiss=False, precision="fp16")
        assert [d.text for d in built.documents] == ["person", "location"]
        assert all(isinstance(d, document) for d in built.documents)
    finally:
        uninstall_fake_relik(saved)


def test_index_fix_leaves_documents_and_stores_untouched():
    calls = []
    _silly, index, document, store, saved = install_fake_relik(calls)
    try:
        relik_module.load_relik()
        documents = [document("a"), document("b")]
        built = index(documents=documents)
        assert built.documents is documents
        documents_store = store([document("a")])
        built_store = index(documents=documents_store)
        assert built_store.documents is documents_store
    finally:
        uninstall_fake_relik(saved)


def test_the_two_patches_are_idempotent():
    calls = []
    silly, index, _document, _store, saved = install_fake_relik(calls)
    try:
        relik_module.load_relik()
        silly_init, index_init = silly.__init__, index.__init__
        relik_module.load_relik()
        assert silly.__init__ is silly_init
        assert index.__init__ is index_init
        assert len(calls) == 2
    finally:
        uninstall_fake_relik(saved)


def test_engine_module_never_imports_heavy_dependencies():
    # another test (or venv) may already hold torch, so the delta around a
    # fresh import of the engine is what matters: importing it must not
    # pull torch, relik or huggingface_hub (the csv shim is stdlib-only;
    # the load fixes import relik lazily inside the functions).
    saved = {name: module for name, module in sys.modules.items()
             if name == "relik_runner.relik"}
    for name in saved:
        del sys.modules[name]
    before = set(sys.modules)
    try:
        module = importlib.import_module("relik_runner.relik")
        added = set(sys.modules) - before
        assert module is not None
        for heavy in ("torch", "relik", "huggingface_hub"):
            assert heavy not in added, heavy
    finally:
        sys.modules.update(saved)


def test_get_pipeline_loads_once_per_model_and_reset_forces_a_reload():
    # get_pipeline takes the WIRE model ids (predict's contract) and caches
    # per HF repo: the catalog id and the sibling are distinct pipelines
    with patch.object(relik_module, "load_relik", return_value="pipeline") as load:
        first = relik_module.get_pipeline("relik-ie/relik")
        assert relik_module.get_pipeline("relik-ie/relik") is first
        assert load.call_count == 1
        load.assert_called_once_with(MODEL_ID)
        sibling = relik_module.get_pipeline(NER_MODEL_ID)
        assert load.call_count == 2 and sibling == "pipeline"
        load.assert_called_with(NER_MODEL_ID)
        relik_module.reset_pipelines()
        relik_module.get_pipeline("relik-ie/relik")
        assert load.call_count == 3


def test_get_pipeline_rejects_unknown_model_ids():
    for bad in ("relik-ie/nope", MODEL_ID):  # wire ids only — not the repo id
        raised = None
        try:
            relik_module.get_pipeline(bad)
        except ValueError as exc:
            raised = exc
        assert raised is not None and "unknown model_id" in str(raised)
