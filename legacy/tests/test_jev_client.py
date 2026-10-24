"""Offline tests for engines/jev_client.py's helpers - no network, no
API key. The danger-zone builders (box, image_asset) are pure local
code; ask() itself is exercised live by demos/jev_image_demo.py."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.jev_client import box, image_asset


def tiny_png(path) -> str:
    # 1x1 transparent PNG - content doesn't matter, only the encoding
    import base64

    with open(path, "wb") as fh:
        fh.write(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="))
    return str(path)


class BoxBuilderTests(unittest.TestCase):
    def test_box_is_type_instructions_only(self):
        self.assertEqual(box("the mug in `photo`"),
                         {"type": "box", "instructions": "the mug in `photo`"})


class ImageAssetTests(unittest.TestCase):
    def test_png_resolves_to_png_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = image_asset(tiny_png(os.path.join(tmp, "scene.png")))
        self.assertEqual(asset["type"], "image")
        self.assertEqual(asset["format"], "png")
        self.assertTrue(asset["data_base64"])

    def test_jpeg_extensions_resolve_to_jpg(self):
        # bytes are opaque to the builder; only the extension is typed
        for name in ("scene.jpg", "scene.JPEG", "scene.jpeg"):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, name)
                with open(path, "wb") as fh:
                    fh.write(b"\xff\xd8\xff")
                asset = image_asset(path)
            self.assertEqual(asset["format"], "jpg", name)

    def test_unsupported_types_raise_before_encoding(self):
        for name in ("scene.webp", "scene.heic", "scene", "scene.txt"):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, name)
                with open(path, "wb") as fh:
                    fh.write(b"not really an image")
                with self.assertRaises(ValueError) as raised:
                    image_asset(path)
            self.assertIn("PNG and JPEG", str(raised.exception), name)


if __name__ == "__main__":
    unittest.main()
