"""Load the ReLiK retriever-reader relation-extraction pipeline on CPU.

relik 1.0.7 cannot even be imported on Windows: relik/retriever/indexers/
document.py:11 runs `csv.field_size_limit(sys.maxsize)` at module import
and Windows' C long is 32-bit (upstream issues SapienzaNLP/relik#14, #15
and #39). Importing this module installs the shim below, so every entry
point (worker, tour, tab) that imports the client first can then import
relik. Two load-time fixes repair the joint NER+relation model's span
retriever and span document index, which relik 1.0.7 cannot load as
published. We do not patch site-packages.
"""

import csv as _csv
import sys as _sys

MODEL_ID = "relik-ie/relik-relation-extraction-small"
NER_MODEL_ID = "relik-ie/relik-relation-extraction-ner-small"

# Windows-safe csv.field_size_limit: sys.maxsize (2**63-1) overflows the
# 32-bit C long there; clamp to INT32_MAX like the stdlib docs suggest.
_field_size_limit = _csv.field_size_limit
_UNSET = object()


def _safe_field_size_limit(limit=_UNSET):
    # no-arg query stays a query: forwarding -1 would set the limit to -1
    if limit is _UNSET:
        return _field_size_limit()
    try:
        return _field_size_limit(limit)
    except (OverflowError, ValueError):
        return _field_size_limit(2**31 - 1)


_csv.field_size_limit = _safe_field_size_limit


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

    _init_with_module_state._relik_bench_patch = True  # idempotency marker
    if getattr(GoldenSillyRetriever.__init__, "_relik_bench_patch", False):
        return
    GoldenSillyRetriever.__init__ = _init_with_module_state


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

    _init_tolerant_of_loader_kwargs._relik_bench_patch = True
    if getattr(BaseDocumentIndex.__init__, "_relik_bench_patch", False):
        return
    BaseDocumentIndex.__init__ = _init_tolerant_of_loader_kwargs


def provenance(model_id: str | None = None):
    return {"model": model_id or MODEL_ID,
            "backend": "retriever-reader (faiss index + DeBERTa reader)"}


def load_relik(model_id: str | None = None):
    """Download and load the full retriever-reader pipeline, strict.
    model_id defaults to MODEL_ID so the benchmark path and the app tab
    stay bit-identical; the tour passes NER_MODEL_ID for the joint
    NER+relation sibling (same pipeline family, 10 Wikipedia-trained
    entity types instead of arbitrary GLiNER-style labels)."""
    try:
        from relik import Relik
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-relik.txt in .venv-relik; relik is "
            "imported only behind the Windows csv shim above.") from exc

    _complete_golden_silly_init()
    _tolerate_index_kwargs()
    return Relik.from_pretrained(model_id or MODEL_ID, device="cpu")
