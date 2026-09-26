"""Load the ReLiK retriever-reader relation-extraction pipeline on CPU.

relik 1.0.7 cannot even be imported on Windows: relik/retriever/indexers/
document.py:11 runs `csv.field_size_limit(sys.maxsize)` at module import
and Windows' C long is 32-bit (upstream issues SapienzaNLP/relik#14, #15
and #39). Importing this module installs the shim below, so every entry
point (worker, tour, tab) that imports the client first can then import
relik. We do not patch site-packages.
"""

import csv as _csv
import sys as _sys

MODEL_ID = "relik-ie/relik-relation-extraction-small"

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


def provenance():
    return {"model": MODEL_ID,
            "backend": "retriever-reader (faiss index + DeBERTa reader)"}


def load_relik():
    """Download and load the full retriever-reader pipeline, strict."""
    try:
        from relik import Relik
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-relik.txt in .venv-relik; relik is "
            "imported only behind the Windows csv shim above.") from exc

    return Relik.from_pretrained(MODEL_ID, device="cpu")
