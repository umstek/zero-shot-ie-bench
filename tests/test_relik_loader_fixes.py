"""Offline tests for the relik client's parametrized loader and its
relik-1.0.7 load fixes.

Nothing here imports real relik (the main venv does not have it): a
fake package replicates the two defects the client patches, so the
patch functions themselves are exercised. The benched model must keep
loading bit-identically - load_relik() with no argument forwards
exactly MODEL_ID - while the tour's NER sibling passes NER_MODEL_ID.

    .venv/Scripts/python -m unittest discover -s tests -p "test_relik_loader_fixes.py"
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines import relik_client


class FakeListConfig:
    """omegaconf.ListConfig stand-in as hydra delivers it: iterable,
    but not a list subclass - the part of the real class the wrapper's
    normalization actually exercises."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)


def install_fake_relik(calls):
    """Install a minimal fake `relik` package whose classes replicate
    the two relik 1.0.7 defects the client fixes: GoldenSillyRetriever
    subclasses an nn.Module but skips super().__init__(), and
    BaseDocumentIndex rejects the use_faiss/precision kwargs its own
    loader injects. Returns (silly, index, Document, DocumentStore,
    saved) - saved restores sys.modules on cleanup."""
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

    relik_mod = types.ModuleType("relik")
    relik_mod.Relik = Relik
    retriever_mod = types.ModuleType("relik.retriever")
    pytorch_mod = types.ModuleType("relik.retriever.pytorch_modules")
    model_mod = types.ModuleType("relik.retriever.pytorch_modules.model")
    model_mod.GoldenSillyRetriever = GoldenSillyRetriever
    indexers_mod = types.ModuleType("relik.retriever.indexers")
    base_mod = types.ModuleType("relik.retriever.indexers.base")
    base_mod.BaseDocumentIndex = BaseDocumentIndex
    document_mod = types.ModuleType("relik.retriever.indexers.document")
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


def uninstall_fake_relik(saved):
    for name in [k for k in sys.modules if k == "relik" or k.startswith("relik.")]:
        del sys.modules[name]
    sys.modules.update(saved)


class ProvenanceTests(unittest.TestCase):
    def test_default_provenance_is_the_benched_model(self):
        self.assertEqual(relik_client.provenance()["model"],
                         relik_client.MODEL_ID)

    def test_ner_override_names_the_sibling(self):
        self.assertEqual(
            relik_client.provenance(relik_client.NER_MODEL_ID)["model"],
            relik_client.NER_MODEL_ID)

    def test_the_two_models_are_distinct_repos(self):
        self.assertNotEqual(relik_client.MODEL_ID, relik_client.NER_MODEL_ID)
        self.assertTrue(relik_client.MODEL_ID.startswith("relik-ie/"))
        self.assertTrue(relik_client.NER_MODEL_ID.startswith("relik-ie/"))


class MissingRelikTests(unittest.TestCase):
    def test_missing_relik_names_the_requirements_file(self):
        saved = {k: v for k, v in sys.modules.items()
                 if k == "relik" or k.startswith("relik.")}
        for name in list(saved):
            del sys.modules[name]
        try:
            with self.assertRaises(RuntimeError) as ctx:
                relik_client.load_relik()
            self.assertIn("requirements-relik.txt", str(ctx.exception))
        finally:
            sys.modules.update(saved)


class LoaderFixTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        (self.silly, self.index, self.document, self.store,
         saved) = install_fake_relik(self.calls)
        self.addCleanup(uninstall_fake_relik, saved)

    def test_default_load_forwards_the_benched_model_verbatim(self):
        relik_client.load_relik()
        self.assertEqual(self.calls, [(relik_client.MODEL_ID, "cpu")])

    def test_ner_load_forwards_the_sibling_model(self):
        relik_client.load_relik(relik_client.NER_MODEL_ID)
        self.assertEqual(self.calls, [(relik_client.NER_MODEL_ID, "cpu")])

    def test_silly_retriever_eval_broken_before_fixed_after_load(self):
        with self.assertRaises(AttributeError):
            self.silly(["person"]).eval()  # the replicated 1.0.7 crash
        relik_client.load_relik()  # applies the nn.Module-init fix
        retriever = self.silly(["person", "location"])
        retriever.eval()  # must no longer raise
        retriever.train(False)
        self.assertEqual(retriever.documents, ["person", "location"])

    def test_index_fix_normalizes_config_documents_and_junk_kwargs(self):
        relik_client.load_relik()  # applies the index fix
        built = self.index(documents=FakeListConfig(["person", "location"]),
                           use_faiss=False, precision="fp16")
        self.assertEqual([d.text for d in built.documents],
                         ["person", "location"])
        self.assertTrue(all(isinstance(d, self.document)
                            for d in built.documents))

    def test_index_fix_leaves_documents_and_stores_untouched(self):
        relik_client.load_relik()
        documents = [self.document("a"), self.document("b")]
        built = self.index(documents=documents)
        self.assertIs(built.documents, documents)
        store = self.store([self.document("a")])
        built_store = self.index(documents=store)
        self.assertIs(built_store.documents, store)

    def test_the_two_patches_are_idempotent(self):
        relik_client.load_relik()
        silly_init, index_init = self.silly.__init__, self.index.__init__
        relik_client.load_relik()
        self.assertIs(self.silly.__init__, silly_init)
        self.assertIs(self.index.__init__, index_init)
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
