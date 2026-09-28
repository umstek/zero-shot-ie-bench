"""Nemotron Rerank VL 1B image-document demo - the "VL" input the
benchmarks never exercised (they scored it on text-only pairs, its worst
row).

nvidia/llama-nemotron-rerank-vl-1b-v2 (via OpenRouter, free tier) is a
vision-language cross-encoder: OpenRouter's rerank endpoint accepts
documents as structured {"image": ...} objects - a remote URL or a
base64-encoded data URI - for multimodal models, where every other
reranker in this repo only ever sees plain strings. The card expects
"screenshots of document pages or slides" with text, tables, charts.

This demo draws three deterministic "page screenshots" with Pillow (an
invoice page, a refund-policy page and a revenue chart page), sends
them as base64 data-URI documents against one query, and prints the
relevance ranking - plus the same model on the same three pages as
plain text, so the image path and the text path sit side by side.

Run (main venv; needs OPENROUTER_API_KEY in .env, network, ~free):
    .venv/Scripts/python demos/nemotron_vl_demo.py
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import base64
import io
import sys

from engines.openrouter_client import rerank

MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
QUERY = "Which page states the refund policy for cancelled orders?"

PAGES = [
    ("invoice", "INVOICE #77\nBilled to: Acme Corp\n3 widgets @ $12.00\n"
                "TOTAL: $36.00\nDue in 30 days"),
    ("refund-policy", "REFUND POLICY\nCancelled orders are refunded in "
                      "full within 14 days.\nNo refund after delivery."),
    ("revenue-chart", "QUARTERLY REVENUE\nQ1 $40k  Q2 $60k\n"
                      "Q3 $50k  Q4 $70k"),
]


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def page_image(text: str, width=480, line_height=26) -> str:
    """One page-screenshot-style PNG (white page, dark text) as a
    base64 data URI - the OpenRouter image-document form."""
    from PIL import Image, ImageDraw

    lines = text.splitlines()
    height = 60 + line_height * len(lines)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, width - 5, height - 5), outline="black", width=2)
    for offset, line in enumerate(lines):
        draw.text((28, 34 + offset * line_height), line, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()).decode("ascii")


def show_ranking(rows: list, kind: str) -> None:
    scores = {row["index"]: row["relevance_score"] for row in rows}
    order = sorted(scores, key=lambda i: -scores[i])
    print(f"  {kind} ranking for: {QUERY!r}")
    for rank, index in enumerate(order, 1):
        name, text = PAGES[index]
        print(f"  {rank}. {name:<13} score={scores[index]:.4f}"
              f'  ("{text.splitlines()[0]}")')


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ------------------------------------------------------------- images
    banner(f"1. {MODEL} - three PIL-drawn page SCREENSHOTS as documents")
    print("  (documents carry base64 data-URI images; the text path the "
          "benches used never sends these)")
    # neutral ids only: the page names below must stay out of the
    # text field, or the ranking proves text matching, not image reading
    documents = [{"image": page_image(text), "text": f"page {i + 1}"}
                 for i, (_, text) in enumerate(PAGES)]
    rows = rerank(MODEL, QUERY, documents)
    show_ranking(rows, "image")

    # -------------------------------------------------------------- text
    banner("2. Same model, same pages as PLAIN TEXT (the benched path)")
    rows = rerank(MODEL, QUERY, [text for _, text in PAGES])
    show_ranking(rows, "text")

    print("\nDone. The other hosted rerankers are text-only (Cohere and "
          "Voyage docs list no image field); the local cross-encoder "
          "trio: demos/reranker_demo.py\n")


if __name__ == "__main__":
    main()
