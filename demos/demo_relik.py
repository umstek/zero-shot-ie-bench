"""ReLiK demo - relik-ie/relik-relation-extraction-small tour (runs under .venv-relik).

Same sample texts as demo.py / demo_glirel.py so outputs can be compared
directly. ReLiK (SapienzaNLP, ACL 2024) is a retriever-reader pipeline, not a
label-prompted encoder: an E5-small retriever fetches the most similar relation
definitions from a fixed index, and a DeBERTa-v3 reader scores span pairs
against only those candidates. The vocabulary is therefore CLOSED - Wikidata
properties (618 of them in this small checkpoint, e.g. "headquarters location"
P159), with their official descriptions as the retrieval documents. You cannot
pass free-text relation labels like GLiREL/GLiNER-relex; the flip side is that
every prediction maps onto a real Wikidata property.
  1. the canonical sample sentences   (closed vocabulary in action)
  2. Wikidata-friendly sentences      (developer, position held, ...)
  3. multilingual probe               (English Wikipedia checkpoints - expect
                                       degraded or empty output, stated up front)
  4. reader entity spans              (untyped --NME-- mentions, char offsets)

Measured on this machine: ~180M params total (33.4M E5-small retriever +
146.5M reader), ~700 MB on disk (reader 570M, encoder 129M, index 0.6M).

License: the HF card says Apache 2.0; the relik repo has no LICENSE file and
its README footer calls the data and software CC BY-NC-SA 4.0. Recorded as-is.

Model: https://huggingface.co/relik-ie/relik-relation-extraction-small
Docs:  https://github.com/SapienzaNLP/relik

Run:
    .venv-relik/Scripts/python demos/demo_relik.py
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys
import time

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

# importing the client first installs the Windows csv shim relik needs
# (SapienzaNLP/relik#39); load_relik() itself is called inside main so the
# multiprocessing workers spawned during inference don't re-run the load
from engines.relik_client import load_relik


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def print_output(text: str, out) -> None:
    print(f'  text: "{text}"')
    if out.spans:
        spans = ", ".join(f"{s.text} [{s.start}:{s.end}]" for s in out.spans)
        print(f"  spans: {spans}")
    else:
        print("  spans: (none)")
    if not out.triplets:
        print("  (no triplets predicted)")
        return
    for t in out.triplets:
        print(f"  {t.subject.text!r:<16} -[{t.label}]-> {t.object.text!r:<16} "
              f"{t.confidence:.3f}")


def annotate(relik, texts: list[str]) -> None:
    """One timed batched call per section, printed line by line."""
    outs = timed(relik, texts)
    if not isinstance(outs, list):
        outs = [outs]
    for text, out in zip(texts, outs):
        print_output(text, out)


# ---------------------------------------------------------------- showcase 1
def canonical(relik) -> None:
    """The two shared sample sentences. Honest contrast: GLiREL/
    GLiNER-relex accept any free-text label we type; ReLiK scores against
    the 618 Wikidata properties it retrieved - 'works for' as a phrase does
    not exist here, its nearest property decides."""
    banner("1. Canonical sample sentences - closed Wikidata vocabulary")
    annotate(relik, ["Alice works for Acme in Paris.",
                     "Apple CEO Tim Cook announced the iPhone 15 in "
                     "Cupertino yesterday."])


# ---------------------------------------------------------------- showcase 2
def wikidata_friendly(relik) -> None:
    """Sentences whose facts map cleanly onto real Wikidata properties
    (developer P178, position held P39, inception P571, ...)."""
    banner("2. Wikidata-friendly sentences - properties the index holds")
    annotate(relik, ["Minecraft is a video game developed by Mojang Studios.",
                     "Angela Merkel was the Chancellor of Germany from 2005 "
                     "to 2021."])


# ---------------------------------------------------------------- showcase 3
def multilingual(relik) -> None:
    """Retriever and reader are English Wikipedia checkpoints; non-English
    text is out of distribution. Same Spanish/German sentences as the GLiREL
    tour - printed honestly, empty output is a real result here."""
    banner("3. Multilingual probe - English-only checkpoints, expect little")
    annotate(relik, ["Miguel de Cervantes escribió Don Quijote en 1605.",
                     "Angela Merkel war Bundeskanzlerin von Deutschland."])


# ---------------------------------------------------------------- showcase 4
def reader_spans(relik) -> None:
    """The reader finds mention spans itself and labels them --NME--:
    this checkpoint is RE-only (no typed NER; the -wikipedia-ner sibling
    adds that). Char offsets, straight from the Span fields."""
    banner("4. Reader entity spans - untyped --NME-- mentions")
    text = "Minecraft is a video game developed by Mojang Studios."
    print(f'  text: "{text}"')
    out = timed(relik, text)
    if isinstance(out, list):
        out = out[0]
    if not out.spans:
        print("  (the reader found no mention spans in this text)")
        return
    for s in out.spans:
        print(f"  Span(start={s.start}, end={s.end}, label={s.label!r}, "
              f"text={s.text!r})")
    print(f"  excerpt check: {text[out.spans[0].start:out.spans[0].end]!r} "
          f"== {out.spans[0].text!r}")


SECTIONS = [canonical, wikidata_friendly, multilingual, reader_spans]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="ReLiK showcase")
    parser.add_argument("--top-k", type=int, default=None,
                        help="relation candidates the retriever hands the "
                             "reader (model default: 30)")
    args = parser.parse_args()

    print("Loading ReLiK (relik-ie/relik-relation-extraction-small, ~180M; "
          "first run downloads ~700 MB: reader + retriever + relation "
          "index)...")
    t0 = time.perf_counter()
    relik = load_relik()
    if args.top_k is not None:
        relik.top_k = args.top_k
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    for section in SECTIONS:
        try:
            section(relik)
        except Exception as exc:  # keep the tour going if one API drifts
            print(f"  [skipped] {section.__name__}: {type(exc).__name__}: {exc}")

    print("\nDone. Free-label counterparts:  python demos/demo_glirel.py  |  "
          "python demos/demo_gliner_relex.py\n")


if __name__ == "__main__":
    main()
