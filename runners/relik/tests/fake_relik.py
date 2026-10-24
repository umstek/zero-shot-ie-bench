"""A fake `relik` package replicating the two relik-1.0.7 defects the
engine patches — ported from legacy tests/test_relik_loader_fixes.py's
install_fake_relik.

GoldenSillyRetriever subclasses an nn.Module but skips super().__init__();
BaseDocumentIndex rejects the use_faiss/precision kwargs its own loader
injects. Installing the fake modules through sys.modules lets the loader
fixes themselves be exercised without the real (torch-stacked) relik.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class FakeListConfig:
    """omegaconf.ListConfig stand-in as hydra delivers it: iterable, but
    not a list subclass — the part of the real class the wrapper's
    normalization actually exercises."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)


def install_fake_relik(calls: list) -> tuple[Any, Any, Any, Any, dict]:
    """Install the minimal fake `relik` package and capture Relik.
    from_pretrained calls. Returns (silly, index, Document, DocumentStore,
    saved) — saved restores sys.modules on cleanup."""
    import torch.nn as nn

    class Relik:
        @staticmethod
        def from_pretrained(model_id, device=None):
            calls.append((model_id, device))
            return object()

    class GoldenRetriever(nn.Module):
        def __init__(self):
            super().__init__()

    class GoldenSillyRetriever(GoldenRetriever):
        def __init__(self, documents, *args, **kwargs):
            # the 1.0.7 bug: nn.Module.__init__ is never called
            self.documents = list(documents)

    class Document:
        def __init__(self, text, **kwargs):
            self.text = text

    class DocumentStore(list):
        pass

    class BaseDocumentIndex:
        # no **kwargs: use_faiss/precision raise TypeError, as in 1.0.7
        def __init__(self, documents=None, embeddings=None,
                     metadata_fields=None, separator=None,
                     name_or_path=None, device="cpu"):
            self.documents = documents

    relik_mod: Any = types.ModuleType("relik")
    relik_mod.Relik = Relik
    retriever_mod: Any = types.ModuleType("relik.retriever")
    pytorch_mod: Any = types.ModuleType("relik.retriever.pytorch_modules")
    model_mod: Any = types.ModuleType("relik.retriever.pytorch_modules.model")
    model_mod.GoldenSillyRetriever = GoldenSillyRetriever
    indexers_mod: Any = types.ModuleType("relik.retriever.indexers")
    base_mod: Any = types.ModuleType("relik.retriever.indexers.base")
    base_mod.BaseDocumentIndex = BaseDocumentIndex
    document_mod: Any = types.ModuleType("relik.retriever.indexers.document")
    document_mod.Document = Document
    document_mod.DocumentStore = DocumentStore

    saved = {k: v for k, v in sys.modules.items()
             if k == "relik" or k.startswith("relik.")}
    sys.modules.update({
        "relik": relik_mod,
        "relik.retriever": retriever_mod,
        "relik.retriever.pytorch_modules": pytorch_mod,
        "relik.retriever.pytorch_modules.model": model_mod,
        "relik.retriever.indexers": indexers_mod,
        "relik.retriever.indexers.base": base_mod,
        "relik.retriever.indexers.document": document_mod,
    })
    return GoldenSillyRetriever, BaseDocumentIndex, Document, DocumentStore, saved


def uninstall_fake_relik(saved: dict) -> None:
    for name in [k for k in sys.modules if k == "relik" or k.startswith("relik.")]:
        del sys.modules[name]
    sys.modules.update(saved)
