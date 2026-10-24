"""GLiFormer — Knowledgator's layout-aware multi-task encoder.

Serves the two benched checkpoints (catalog/models/knowledgator/gliformer-*.yaml,
both `runner: gliner` deployments):
  - gliformer-base-v1.local   -> knowledgator/gliformer-base-v1   (~190M)
  - gliformer-large-v1.local  -> knowledgator/gliformer-large-v1  (~575.6M)

Tracks: classify + ner, threshold 0.5, one call per question — the exact
call paths of legacy/bench_spectrum.py's classify_extractor (gliformer
branch):
  - classify: model.classify(text, labels, threshold=0.5) -> out[0]["class_name"]
  - ner:      model.predict_entities(text, labels, threshold=0.5), deduped
              and sorted (start, end, label) tuples like every NER answer

`build_sample_pdf` is the deterministic two-page sample the parse_pdf tour
stop (legacy/demos/demo_gliformer.py) draws for layout-aware extraction;
it is kept here because the PDF-stop unit tests pin it without weights.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from gliner_runner.models._common import as_text, label_keys

CLASSIFY_THRESHOLD = 0.5  # model.classify's bench threshold
NER_THRESHOLD = 0.5  # predict_entities' default, same as the GLiNER-X bench

# Weights live here after the first load; tests stub entries in this dict
# instead of touching the network. Never populated at import time.
_LOADED: dict[str, Any] = {}


def load(repo: str) -> Any:
    """Load one GLiFormer checkpoint on the CPU (fp32 weights, eval mode).

    First use downloads the checkpoint into the HF cache (~0.8 GB base,
    ~2.3 GB large) — never at import, never in tests without weights.
    """
    if repo not in _LOADED:
        from gliformer import GLiFormer  # the gliformer package (>=0.1.2)

        _LOADED[repo] = GLiFormer.from_pretrained(repo, load_tokenizer=True).to("cpu").eval()
    return _LOADED[repo]


def run_classify(model: Any, input_: Any, question: dict[str, Any],
                 params: dict[str, Any]) -> tuple[str | None, float | None]:
    """Zero-shot classification over one flat label group.

    Bench semantics: labels are the bare keys of the request's label dict,
    the top-scoring class above the threshold wins, nothing above the
    threshold answers None.
    """
    labels = label_keys(question)
    threshold = float(params.get("threshold", CLASSIFY_THRESHOLD))
    text = as_text(input_, "gliformer classify")  # validated before any model call
    out = model.classify(text, labels, threshold=threshold)
    if not out:
        return None, None
    return out[0]["class_name"], out[0].get("score")


def run_ner(model: Any, input_: Any, question: dict[str, Any],
            params: dict[str, Any]) -> tuple[list[tuple[int, int, str]], None]:
    """Zero-shot entity spans: sorted, deduped (start, end, label) tuples."""
    labels = question.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("gliformer NER expects `question.labels` as a list of strings")
    threshold = float(params.get("threshold", NER_THRESHOLD))
    text = as_text(input_, "gliformer NER")  # validated before any model call
    entities = model.predict_entities(text, labels, threshold=threshold)
    spans = sorted({(int(e["start"]), int(e["end"]), str(e["label"])) for e in entities})
    return spans, None


def build_sample_pdf(directory: str | None = None) -> str:
    """A deterministic two-page support-ticket PDF drawn with pymupdf:
    page 1 names a person, company, money and an email in body lines;
    page 2 is a monospaced refund-ledger table. parse_pdf reads the words,
    their boxes and a rendered page image from the same file, so the
    sample needs a real text layer, not a screenshot. Pages stay card-size
    (340x150 pt): the layout encoder's memory scales with page pixels, and
    A4 pages at its fixed 144 dpi want ~8 GB on CPU.

    Ported verbatim from legacy/demos/demo_gliformer.py so the PDF-stop
    tests pin the generator without weights.
    """
    import pymupdf

    if directory is None:
        directory = tempfile.mkdtemp(prefix="gliformer-pdf-")
    path = os.path.join(directory, "ticket.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=340, height=150)
    y = 40
    for line in (
        "Support Ticket #4821 - Acme Cloud Services",
        "Customer: Dana Whitfield",
        "Account ENT-2210 billed $1,240.00 on 2026-09-02",
        "and again on 2026-09-03.",
        "Contact: dana.whitfield@northwind.example",
    ):
        page.insert_text((24, y), line, fontname="helv", fontsize=8)
        y += 15
    page2 = doc.new_page(width=340, height=150)
    page2.insert_text((24, 36), "Refund ledger", fontname="hebo", fontsize=9)
    y = 54
    for row in (
        "Date        Amount     Status",
        "2026-09-02  $1,240.00  captured",
        "2026-09-03  $1,240.00  captured",
        "2026-09-04  $1,240.00  refund pending",
    ):
        page2.insert_text((24, y), row, fontname="cour", fontsize=7)
        y += 13
    doc.save(path)
    doc.close()
    return path
