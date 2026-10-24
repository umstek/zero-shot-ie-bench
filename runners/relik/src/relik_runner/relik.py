"""The relik engine: loading + predict, ported from legacy/engines/relik_client.py.

relik 1.0.7 cannot even be imported on Windows: relik/retriever/indexers/
document.py:11 runs `csv.field_size_limit(sys.maxsize)` at module import
and Windows' C long is 32-bit (upstream issues SapienzaNLP/relik#14, #15
and #39). Importing THIS module installs the shim below, so every entry
point (server, tests, UI) that imports the engine first can then import
relik. Two load-time fixes repair the joint NER+relation model's span
retriever and span document index, which relik 1.0.7 cannot load as
published. We do not patch site-packages.

Two pipelines are served (catalog/models/relik-ie/relik.yaml):

- relik-ie/relik-relation-extraction-small — the benched deployment,
  reachable on the wire as the catalog model id "relik-ie/relik"; closed
  Wikidata-property relation vocabulary, untyped --NME-- reader spans.
- relik-ie/relik-relation-extraction-ner-small — the demoed joint
  NER+relation sibling (10 Wikipedia-trained entity types); needs the two
  load-time fixes. On the wire it is keyed by its HF repo id, the same
  string legacy's tour passed to load_relik.

Heavy imports (torch, relik, huggingface_hub) stay inside the load path so
this module imports without them — the csv shim is stdlib-only — and
offline tests stub the `relik` package exactly like legacy
tests/test_relik_loader_fixes.py did.
"""

from __future__ import annotations

import csv as _csv
import threading
from typing import Any

# ---------------------------------------------------------- Windows csv shim
# Windows-safe csv.field_size_limit: sys.maxsize (2**63-1) overflows the
# 32-bit C long there; clamp to INT32_MAX like the stdlib docs suggest.
_field_size_limit = _csv.field_size_limit
_UNSET = object()


def _set_field_size_limit(limit: Any) -> int:
    # Any, deliberately: whatever we got is forwarded raw to the C
    # function (which does its own type checking), as in the legacy shim
    try:
        return _field_size_limit(limit)
    except (OverflowError, ValueError):
        return _field_size_limit(2**31 - 1)


def _safe_field_size_limit(new_limit: int | object = _UNSET) -> int:
    # no-arg query stays a query: forwarding -1 would set the limit to -1
    if new_limit is _UNSET:
        return _field_size_limit()
    return _set_field_size_limit(new_limit)


_csv.field_size_limit = _safe_field_size_limit


# ------------------------------------------------------------ model identity
# HF identity of the two checkpoints (download + provenance; legacy
# MODEL_ID / NER_MODEL_ID).
MODEL_ID = "relik-ie/relik-relation-extraction-small"
NER_MODEL_ID = "relik-ie/relik-relation-extraction-ner-small"

# Wire identity served on /info and accepted as TrackRequest.model_id.
# The benched deployment answers to the catalog model id
# (catalog/models/relik-ie/relik.yaml, deployment
# relik-relation-extraction-small.local); the NER sibling has no catalog
# deployment (demoed only — v0.benched is false), so it answers to its HF
# repo id, the string the legacy tour passed to load_relik().
CATALOG_MODEL_ID = "relik-ie/relik"

# wire model_id -> (hf repo, default track when the question shape is
# silent). Both pipelines produce spans AND triplets; the defaults just
# match each checkpoint's primary purpose (the catalog claims tracks
# [relation, ner] with the relation deployment first).
WIRE_MODELS: dict[str, tuple[str, str]] = {
    CATALOG_MODEL_ID: (MODEL_ID, "relation"),
    NER_MODEL_ID: (NER_MODEL_ID, "ner"),
}


def provenance(model_id: str | None = None) -> dict[str, str]:
    """The identity row the legacy bench recorded alongside results."""
    return {"model": model_id or MODEL_ID,
            "backend": "retriever-reader (faiss index + DeBERTa reader)"}


# ------------------------------------------- relik-1.0.7 load-time fixes


def _complete_golden_silly_init() -> None:
    """Fix relik 1.0.7's GoldenSillyRetriever before it is instantiated.

    The joint NER+relation config (NER_MODEL_ID) declares its span
    retriever as GoldenSillyRetriever, which subclasses the nn.Module
    GoldenRetriever but never calls super().__init__() - so the nn.Module
    bookkeeping (_modules, _parameters, ...) is missing and the loader's
    `retriever.eval()` crashes with AttributeError. Completing
    nn.Module.__init__ first makes eval()/train() work (the loader calls
    .to() only on already-instantiated retrievers, never on configs).
    Like the csv shim above, this patches the class in our process, not
    site-packages."""
    try:
        import torch.nn as nn
        from relik.retriever.pytorch_modules.model import GoldenSillyRetriever
    except ImportError:
        return  # relik is optional; load_relik reports install problems
    original_init = GoldenSillyRetriever.__init__

    def _init_with_module_state(self, documents, *args, **kwargs):
        nn.Module.__init__(self)
        original_init(self, documents, *args, **kwargs)

    _init_with_module_state._relik_bench_patch = True  # type: ignore[attr-defined]
    if getattr(GoldenSillyRetriever.__init__, "_relik_bench_patch", False):
        return
    GoldenSillyRetriever.__init__ = _init_with_module_state  # type: ignore[method-assign]


def _tolerate_index_kwargs() -> None:
    """Fix relik 1.0.7's index loading before the NER config loads.

    The joint NER+relation config (NER_MODEL_ID) builds its span document
    index from a plain list of the 10 entity-type strings, but relik
    1.0.7's BaseDocumentIndex expects ready-made Document objects - and
    receives them from hydra as an OmegaConf ListConfig on top; finally,
    _instantiate_index (relik/inference/utils.py) unconditionally merges
    a `use_faiss` key - and `precision`, when set - into every index
    config, which BaseDocumentIndex.__init__ also rejects. Only the
    triplet index's from_pretrained tolerates all of this, which is why
    the benched relation-only model loads fine. The wrapper normalizes
    documents (ListConfig -> list, strings -> Documents, like
    GoldenSillyRetriever does for its own store) and retries once
    without the keys the signature rejects, so a future relik that
    actually accepts them keeps them."""
    try:
        from relik.retriever.indexers.base import BaseDocumentIndex
        from relik.retriever.indexers.document import Document, DocumentStore
    except ImportError:
        return  # relik is optional; load_relik reports install problems
    original_init = BaseDocumentIndex.__init__

    def _init_tolerant_of_loader_kwargs(self, *args, **kwargs):
        if "documents" in kwargs:
            documents = kwargs["documents"]
            # listify ONLY config-like iterables - None, a path string
            # or bytes must reach relik's own handling untouched
            if (documents is not None
                    and not isinstance(documents, (list, str, bytes,
                                                   DocumentStore))
                    and hasattr(documents, "__iter__")):
                documents = list(documents)  # hydra hands over a ListConfig
            if (isinstance(documents, list) and documents
                    and all(isinstance(d, str) for d in documents)):
                documents = [Document(d) for d in documents]
            kwargs["documents"] = documents
        try:
            original_init(self, *args, **kwargs)
        except TypeError:
            for loader_junk in ("use_faiss", "precision"):
                kwargs.pop(loader_junk, None)
            original_init(self, *args, **kwargs)

    _init_tolerant_of_loader_kwargs._relik_bench_patch = True  # type: ignore[attr-defined]
    if getattr(BaseDocumentIndex.__init__, "_relik_bench_patch", False):
        return
    BaseDocumentIndex.__init__ = _init_tolerant_of_loader_kwargs  # type: ignore[method-assign]


# ------------------------------------------------------------------ loading
# Weights live here after the first load, keyed by HF repo; tests stub
# entries in this dict instead of touching the network. Never populated at
# import time.
_PIPELINES: dict[str, Any] = {}
_PIPELINE_LOCK = threading.Lock()


def load_relik(model_id: str | None = None) -> Any:
    """Download and load the full retriever-reader pipeline, strict.
    model_id defaults to MODEL_ID so the benchmark path and the app tab
    stay bit-identical; the NER sibling passes NER_MODEL_ID for the joint
    NER+relation pipeline (same family, 10 Wikipedia-trained entity
    types instead of arbitrary GLiNER-style labels)."""
    try:
        from relik import Relik
    except ImportError as exc:
        raise RuntimeError(
            "relik is not installed; sync this island's pins "
            "(uv sync: relik[faiss]==1.0.7 on Python 3.11) - relik is "
            "imported only behind the Windows csv shim in relik_runner.relik."
        ) from exc

    _complete_golden_silly_init()
    _tolerate_index_kwargs()
    return Relik.from_pretrained(model_id or MODEL_ID, device="cpu")


def get_pipeline(model_id: str) -> Any:
    """Process-wide lazy singleton per wire model: the pipeline loads on
    the first predict (HTTP request), never at import. Lock-guarded
    because FastAPI runs sync handlers on a threadpool."""
    repo, _ = _wire_model(model_id)
    with _PIPELINE_LOCK:
        if repo not in _PIPELINES:
            _PIPELINES[repo] = load_relik(repo)
        return _PIPELINES[repo]


def reset_pipelines() -> None:
    """Drop the cached pipelines (tests swap in stubs between cases)."""
    with _PIPELINE_LOCK:
        _PIPELINES.clear()


def cached_snapshot(model_id: str = MODEL_ID) -> str | None:
    """The checkpoint's HF cache snapshot if it is already downloaded.

    local_files_only=True — this never downloads. Returns None when the
    snapshot is absent; the weight tests gate on it so CI never pulls
    weights."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover — huggingface-hub is a hard dep
        return None
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 — any cache miss means "not cached"
        return None


def _wire_model(model_id: str) -> tuple[str, str]:
    """Resolve a wire model_id to (hf repo, default track); unknown ids
    raise the ValueError predict() turns into a per-item error row."""
    if model_id not in WIRE_MODELS:
        raise ValueError(
            f"unknown model_id: {model_id!r} (this runner serves "
            f"{CATALOG_MODEL_ID!r} and {NER_MODEL_ID!r})"
        )
    return WIRE_MODELS[model_id]


# ------------------------------------------------------------- wire rows
# Ported from legacy demos/relik_demo.py — the relik output -> JSON row
# mappings the legacy web runner and demo UI relied on.


def span_row(span: Any) -> dict[str, Any]:
    """A relik Span (NamedTuple) -> {start, end, label, text}, the
    legacy span_to_dict shape (char offsets)."""
    return {"start": int(span.start), "end": int(span.end),
            "label": str(span.label), "text": str(span.text)}


def triplet_row(triplet: Any) -> dict[str, Any]:
    """A relik triplet -> {subject, relation, object, confidence}, the
    legacy web-runner row (subject/object as span_rows)."""
    return {"subject": span_row(triplet.subject),
            "relation": str(triplet.label),
            "object": span_row(triplet.object),
            "confidence": float(triplet.confidence)}


# ------------------------------------------------- payload -> track + text


def resolve_track(model_id: str, question: dict[str, Any],
                  params: dict[str, Any]) -> str:
    """Pick the track an item is asking for. The protocol does not echo
    the track to predict(), so:

    - an explicit "track": "ner" | "relation" on the question or params wins
    - relation keys ("relations", "relation_labels", "triplets") ask for
      relation extraction
    - span keys ("entities", "entity_labels", "labels", "spans") ask for
      NER — relik's vocabulary is CLOSED (Wikidata properties / the 10
      Wikipedia entity types), so any labels riding along are not shown
      to the model
    - otherwise the checkpoint's primary: relation for the benched
      deployment, NER for the joint sibling.
    """
    for source in (question, params):
        explicit = source.get("track")
        if explicit in ("ner", "relation"):
            return str(explicit)
    if any(key in question or key in params
           for key in ("relations", "relation_labels", "triplets")):
        return "relation"
    if any(key in question or key in params
           for key in ("entities", "entity_labels", "labels", "spans")):
        return "ner"
    return _wire_model(model_id)[1]


def _text_of(payload: dict[str, Any]) -> str:
    """The item's text: a plain string input (the ner/relation islands'
    convention; anything else is a caller bug and gets a clear error)."""
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("relik expects a non-empty text string as `input`")
    return text


def _run(pipeline: Any, text: str) -> Any:
    """One inference call; relik returns a bare RelikOutput for a single
    string and a list for a list — normalize to the single output.

    num_workers=0 is pinned: the retriever's inference DataLoader defaults
    to 4 worker PROCESSES, and on Windows a spawned child re-imports relik
    before anything can install our csv shim (the legacy one-shot demo
    only survived it because `import engines.relik_client` sat at the top
    of __main__, which spawn children re-execute — a server or pytest
    process has no such __main__). relik's own gradio frontend passes
    num_workers=0 for the same reason; on CPU the in-process batches are
    the right trade anyway."""
    out = pipeline(text, num_workers=0)
    if isinstance(out, list):
        out = out[0]
    return out


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """runner-protocol predict(model_id, payload) -> {prediction, prob, error}.

    payload is the item's {"input", "question", "params"} (the protocol
    strips item_id). The prediction is the track's rows: relation ->
    triplet rows [{subject, relation, object, confidence}], ner -> span
    rows [{start, end, label, text}] (untyped --NME-- on the benched
    deployment, typed on the sibling). prob is None — per-row confidences
    ride inside the rows, like the gliner island's list predictions.
    Never raises: every failure becomes an {"error": "Type: message"} row
    so one bad item cannot kill a run."""
    try:
        text = _text_of(payload)
        question = payload.get("question") or {}
        params = payload.get("params") or {}
        if not isinstance(question, dict):
            raise ValueError("payload 'question' must be an object")
        if not isinstance(params, dict):
            raise ValueError("payload 'params' must be an object")
        track = resolve_track(model_id, question, params)
        pipeline = get_pipeline(model_id)
        if params.get("top_k") is not None:
            # the retriever's candidate budget (legacy demo's --top-k);
            # set on the shared pipeline like the demo did, just before
            # the single call that reads it
            pipeline.top_k = int(params["top_k"])
        out = _run(pipeline, text)
        if track == "relation":
            rows: list[dict[str, Any]] = [triplet_row(t) for t in out.triplets]
        else:
            rows = [span_row(s) for s in out.spans]
        return {"prediction": rows, "prob": None, "error": None}
    except Exception as exc:  # noqa: BLE001 — per-item isolation by contract
        return {"prediction": None, "prob": None, "error": f"{type(exc).__name__}: {exc}"}
