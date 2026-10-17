"""Jev IMAGE-input + box demo - the danger-zone surface of System One.

Jev's image path is a danger-zone opt-in (Jev 1.13 and 1.14): the
TypeSafe-Danger-Zone-Images: true header unlocks a top-level `assets`
map on POST /v1/systemone. Each entry is one base64 image keyed by a
reference string, and that reference must appear somewhere in the
state, instructions or criteria - the model receives the image in its
place. Jev 1.14 adds the `box` question type behind the same header:
instructions name the target ("the red mug in `photo`") and the answer
is four edge fractions {top, bottom, left, right} of the image, 0-1,
always returned (even for an absent target - pair it with a noul).
docs.typesafe.ai/images and docs.typesafe.ai/primitives/box.

The demo draws one deterministic desk scene with Pillow (red mug at a
KNOWN fractional position, blue book, green plant - no binary assets
ship in the repo) and asks one batched request with three questions:
a choice over the mug's color, a noul for the mug's presence, and the
box for the mug's position - printed next to the drawn ground truth.
Images bill as input tokens; PNG/JPEG only, 280-1568 px per side (the
API rejects rather than resizes - the scene is 640x480).

The box question is 1.14-only, so the demo defaults to jev-1.14.0
(pass --model to override; on 1.13 the request itself fails on the box
row). 1.14 first failed the 12-question trial gate, but a full-suite
parity run (45/48 mixed pool = the incumbent's recorded score, 54/54
multilingual) overturned the miss on 2026-10-17 and the benchmark's
Jev row now pins jev-1.14.0 - this demo exercises the danger zone of
the version the bench records.

Run (paid API, one request; TYPESAFE_API_KEY in env or repo .env):
    .venv/Scripts/python demos/jev_image_demo.py
    .venv/Scripts/python demos/jev_image_demo.py --model jev-latest
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import tempfile
from pathlib import Path

from engines.jev_client import (DANGER_IMAGES_HEADERS, JevClient, box,
                                choice, image_asset, noul)

# the drawn mug's BODY rectangle, fractional image coordinates
# (x0, y0, x1, y1) - x fractions of the width, y fractions of the
# height (Jev's box edges are height fractions vertically). The printed
# ground truth expands this to the full mug, handle included.
MUG_FRACTIONS = (0.30, 0.45, 0.55, 0.80)


def mug_truth() -> tuple[float, float, float, float]:
    """The full-mug bounding box (left, top, right, bottom) in fractions:
    the body rectangle plus the handle's rightward reach. The handle is
    drawn a sixth of the body width out and half a body width wide."""
    x0, y0, x1, _ = MUG_FRACTIONS
    body_w = x1 - x0
    return (x0, y0, x1 + body_w / 6 + body_w / 2, MUG_FRACTIONS[3])


def build_scene(directory=None) -> str:
    """Draw the deterministic 640x480 desk scene; return its file path
    (a fresh temp dir unless one is given, so the repo ships no PNGs)."""
    from PIL import Image, ImageDraw

    root = Path(directory or tempfile.mkdtemp(prefix="jev-image-demo-"))
    root.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    image = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(image)

    # desk surface line
    draw.rectangle((0, int(height * 0.82), width, height), fill="#c9a97e")
    # blue book, upper-left third
    draw.rectangle((int(width * 0.08), int(height * 0.20),
                    int(width * 0.28), int(height * 0.38)), fill="#3a6ea5")
    # green plant, right edge
    draw.rectangle((int(width * 0.78), int(height * 0.55),
                    int(width * 0.95), int(height * 0.82)), fill="#4c7a3f")
    draw.ellipse((int(width * 0.74), int(height * 0.38),
                  int(width * 0.99), int(height * 0.62)), fill="#5d9149")
    # the red mug (body + handle); x from width, y from height, so the
    # fractional MUG_FRACTIONS land where they claim
    fx0, fy0, fx1, fy1 = MUG_FRACTIONS
    x0, y0, x1, y1 = (int(width * fx0), int(height * fy0),
                      int(width * fx1), int(height * fy1))
    draw.rectangle((x0, y0, x1, y1), fill="#b3392e")
    draw.ellipse((x0, y0, x1, y0 + (y1 - y0) // 6), fill="#8f2b23")
    handle_x0 = x1 + (x1 - x0) // 6
    draw.arc((handle_x0, y0 + (y1 - y0) // 4,
              handle_x0 + (x1 - x0) // 2, y1 - (y1 - y0) // 4),
             start=-90, end=90, fill="#b3392e", width=8)

    path = root / "desk.png"
    image.save(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="jev-1.14.0",
                        help="versioned model id (box needs 1.14+)")
    args = parser.parse_args()

    scene_path = build_scene()
    print(f"scene: {scene_path} (mug body at x "
          f"{MUG_FRACTIONS[0]:.2f}-{MUG_FRACTIONS[2]:.2f}, y "
          f"{MUG_FRACTIONS[1]:.2f}-{MUG_FRACTIONS[3]:.2f}, handle "
          "reaching further right)")

    client = JevClient(model=args.model,
                       extra_headers=DANGER_IMAGES_HEADERS)
    state = ("A photo of a desk. The image is attached as the asset "
             "referenced by `photo`.")
    questions = {
        "mug_color": choice("What color is the mug in `photo`?",
                            {"red": None, "blue": None, "green": None}),
        "mug_present": noul("Is there a mug visible in `photo`?"),
        "mug_box": box("the mug in `photo`"),
    }
    payload = client.ask(state, questions,
                         assets={"photo": image_asset(scene_path)})

    print(f"model that answered: {payload.get('model') or client.last_model}")
    print(f"latency: {payload.get('_latency_s')} s, "
          f"usage: {json.dumps(payload.get('usage'), default=str)}")
    answers = payload["answers"]
    color = answers["mug_color"]
    print(f"mug_color   -> {color.get('choice')} "
          f"(confidence {color.get('confidence')})")
    print(f"mug_present -> yes-probability "
          f"{answers['mug_present'].get('noul')}")
    found = answers["mug_box"].get("box") or {}
    if found:
        left, top, right, bottom = mug_truth()
        print("mug_box     -> left {:.2f} right {:.2f} top {:.2f} "
              "bottom {:.2f} (confidence {:.2f})".format(
                  found.get("left", float("nan")),
                  found.get("right", float("nan")),
                  found.get("top", float("nan")),
                  found.get("bottom", float("nan")),
                  answers["mug_box"].get("confidence", float("nan"))))
        print("drawn truth -> left {:.2f} right {:.2f} top {:.2f} "
              "bottom {:.2f} (full mug, handle included)".format(
                  left, right, top, bottom))


if __name__ == "__main__":
    main()
