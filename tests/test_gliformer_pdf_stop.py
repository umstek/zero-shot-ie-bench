"""Offline tests for the GLiFormer parse_pdf tour stop (demos/demo_gliformer.py).

No checkpoint, no gliformer inference: the deterministic sample-PDF
generator is pinned (same bytes when rebuilt, two pages, the expected
text on each page). pymupdf draws the sample, so it must be installed in
the main venv (the demo's docstring says how); torch is imported by the
demo module itself and ships in the main venv.

    .venv/Scripts/python -m unittest tests.test_gliformer_pdf_stop
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SamplePdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # gliformer itself must stay out of the test path: the generator
        # imports pymupdf inside the function, the model loader far away
        from demos import demo_gliformer as demo

        cls.demo = demo
        import tempfile

        cls.directory = tempfile.mkdtemp(prefix="pdf-stop-test-")
        cls.path = demo.build_sample_pdf(cls.directory)

    def test_rebuild_is_byte_identical(self):
        again = self.demo.build_sample_pdf(self.directory)
        with open(self.path, "rb") as fh:
            first = fh.read()
        with open(again, "rb") as fh:
            self.assertEqual(first, fh.read())

    def test_two_pages_with_expected_text(self):
        import pymupdf

        doc = pymupdf.open(self.path)
        try:
            self.assertEqual(len(doc), 2)
            page1 = doc[0].get_text()
            page2 = doc[1].get_text()
        finally:
            doc.close()
        for needle in ("Acme Cloud Services", "Dana Whitfield",
                       "$1,240.00", "dana.whitfield@northwind.example"):
            self.assertIn(needle, page1)
        self.assertIn("Refund ledger", page2)
        self.assertIn("refund pending", page2)

    def test_pages_are_card_size_not_a4(self):
        # the layout encoder's memory scales with page pixels: A4 pages at
        # its fixed 144 dpi want ~8 GB on CPU, card pages stay runnable
        import pymupdf

        doc = pymupdf.open(self.path)
        try:
            self.assertLess(doc[0].rect.width, 400)
            self.assertLess(doc[0].rect.height, 200)
        finally:
            doc.close()


if __name__ == "__main__":
    unittest.main()
