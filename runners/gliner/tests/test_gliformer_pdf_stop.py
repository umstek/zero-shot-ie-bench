"""Ported offline tests for the GLiFormer parse_pdf tour stop.

No checkpoint, no gliformer inference: the deterministic sample-PDF
generator (now gliner.models.gliformer.build_sample_pdf, ported from
legacy/demos/demo_gliformer.py) is pinned — same bytes when rebuilt, two
pages, the expected text on each page. pymupdf draws the sample and ships
as an island dependency (gliformer only pulls it in under its pdf extras).
"""

from __future__ import annotations

import re
import tempfile

from gliner_runner.models.gliformer import build_sample_pdf


class TestSamplePdf:
    def test_rebuild_is_byte_identical(self):
        directory = tempfile.mkdtemp(prefix="pdf-stop-test-")
        path = build_sample_pdf(directory)
        with open(path, "rb") as fh:  # the FIRST build's bytes
            original = fh.read()
        again = build_sample_pdf(directory)
        with open(again, "rb") as fh:
            rebuilt = fh.read()
        # pymupdf writes a fresh random /ID into the trailer on every
        # save, so strict byte equality is impossible; everything else
        # must be identical
        def normalize(blob):
            return re.sub(rb"/ID\[<[0-9A-Fa-f]+>\s*<[0-9A-Fa-f]+>\]",
                          b"/ID[X]", blob)

        assert normalize(original) == normalize(rebuilt)

    def test_two_pages_with_expected_text(self):
        import pymupdf

        doc = pymupdf.open(build_sample_pdf(tempfile.mkdtemp(prefix="pdf-stop-test-")))
        try:
            assert len(doc) == 2
            page1 = doc[0].get_text()
            page2 = doc[1].get_text()
        finally:
            doc.close()
        for needle in ("Acme Cloud Services", "Dana Whitfield",
                       "$1,240.00", "dana.whitfield@northwind.example"):
            assert needle in page1
        assert "Refund ledger" in page2
        assert "refund pending" in page2

    def test_pages_are_card_size_not_a4(self):
        # the layout encoder's memory scales with page pixels: A4 pages at
        # its fixed 144 dpi want ~8 GB on CPU, card pages stay runnable
        import pymupdf

        doc = pymupdf.open(build_sample_pdf(tempfile.mkdtemp(prefix="pdf-stop-test-")))
        try:
            assert doc[0].rect.width < 400
            assert doc[0].rect.height < 200
        finally:
            doc.close()
