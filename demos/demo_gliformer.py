"""GLiFormer demo - knowledgator/gliformer-large-v1 tour (runs next to demo.py).

Same sample texts as demo.py (GLiNER 2.5) so outputs can be compared directly.
GLiFormer is Knowledgator's layout-aware multi-task encoder:
  1. entity extraction            (predict_entities)
  2. text classification          (classify, incl. named label groups)
  3. joint relation extraction    (inference + joint_relations)
  4. structured records           (structure, incl. nested Pydantic schemas)
  5. several tasks in one call    (inference)
  6. text embeddings              (embed_text)
  7. PDF document input           (parse_pdf over a pymupdf-drawn sample)

Model: https://huggingface.co/knowledgator/gliformer-large-v1
Docs:   https://github.com/Knowledgator/GLiFormer

parse_pdf additionally needs PyMuPDF (the package imports it lazily):
    uv pip install --python .venv pymupdf

Run:
    python demos/demo_gliformer.py                 # large (575.6M, English)
    python demos/demo_gliformer.py --model base    # base checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import torch

MODELS = {
    "base": "knowledgator/gliformer-base-v1",
    "large": "knowledgator/gliformer-large-v1",
}


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


# ---------------------------------------------------------------- showcase 1
def entities(model):
    banner("1a. Entity extraction - plain labels")
    text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
    print(f'  text: "{text}"')
    for ent in timed(
        model.predict_entities,
        text,
        ["company", "person", "product", "location"],
        threshold=0.5,
    ):
        print(f"  {ent['text']!r:<15} {ent['label']:<10} {ent['score']:.3f}")

    banner("1b. Entity extraction - clinical domain labels")
    text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
    print(f'  text: "{text}"')
    for ent in timed(
        model.predict_entities,
        text,
        ["medication", "dosage", "symptom", "time"],
        threshold=0.5,
    ):
        print(f"  {ent['text']!r:<15} {ent['label']:<10} {ent['score']:.3f}")


# ---------------------------------------------------------------- showcase 2
def classification(model):
    banner("2a. Text classification - single label group")
    preds = timed(
        model.classify,
        "This laptop has amazing performance but terrible battery life!",
        ["positive", "negative", "neutral"],
        threshold=0.5,
    )
    show(preds)

    banner("2b. Text classification - named label groups")
    preds = timed(
        model.classify,
        "The new search feature is fast and easy to use.",
        {"sentiment": ["positive", "negative"], "topic": ["product", "support"]},
        threshold=0.5,
    )
    show(preds)


# ---------------------------------------------------------------- showcase 3
def relations(model):
    banner("3. Joint relation extraction")
    text = "Alice works for Acme in Paris."
    print(f'  text: "{text}"')
    results = timed(
        model.inference,
        text,
        joint_relations={
            "employment": {
                "entities": ["person", "organization", "location"],
                "relations": ["works_for", "located_in"],
            }
        },
        threshold=0.5,
    )
    for rel in results["joint_relex"][0]:
        head, tail = rel["head"], rel["tail"]
        print(f"  {head['text']} -{rel['relation']}-> {tail['text']}  "
              f"(score {rel.get('score', float('nan')):.2f})")


# ---------------------------------------------------------------- showcase 4
def records(model):
    banner("4a. Structured records - flat schema")
    text = "Alice bought apples and Bob bought oranges."
    print(f'  text: "{text}"')
    result = timed(
        model.structure,
        text,
        {"purchase": ["buyer", "item"]},
    )
    show(result)

    banner("4b. Structured records - nested Pydantic schema")
    from pydantic import BaseModel

    class Employee(BaseModel):
        name: str
        role: str

    class Department(BaseModel):
        name: str
        employees: list[Employee]

    class Company(BaseModel):
        name: str
        departments: list[Department]

    text = (
        "At Acme, Engineering includes Alice, a software engineer, and Bob, "
        "a designer. Sales includes Carol, an account manager."
    )
    print(f'  text: "{text}"')
    result = timed(
        model.structure,
        text,
        {"company": Company},
        validate_output=True,
    )
    show(result)


# ---------------------------------------------------------------- showcase 5
def multitask(model):
    banner("5. Multiple tasks in one inference call")
    text = "Alice joined Acme as a software engineer."
    print(f'  text: "{text}"')
    results = timed(
        model.inference,
        text,
        entities=["person", "organization"],
        classes=["business", "sports", "technology"],
        structures={"employee": ["name", "company"]},
    )
    print("  ner:");           show(results["ner"][0])
    print("  classification:"); show(results["classification"][0])
    print("  structuring:");   show(results["structuring"][0])


# ---------------------------------------------------------------- showcase 6
def embeddings(model):
    banner("6. Text embeddings (unique to GLiFormer)")
    vectors = timed(
        model.embed_text,
        [
            "A scientist works in a laboratory.",
            "A researcher conducts an experiment.",
            "I would like a pizza with extra cheese.",
        ],
    )
    print(f"  shape: {tuple(vectors.shape)}")
    cos = torch.nn.functional.cosine_similarity
    print(f"  cos(lab, research) = {cos(vectors[0:1], vectors[1:2]).item():.3f}")
    print(f"  cos(lab, pizza)    = {cos(vectors[0:1], vectors[2:3]).item():.3f}")


# ---------------------------------------------------------------- showcase 7
def build_sample_pdf(directory=None):
    """A deterministic two-page support-ticket PDF drawn with pymupdf:
    page 1 names a person, company, money and an email in body lines;
    page 2 is a monospaced refund-ledger table. parse_pdf reads the words,
    their boxes and a rendered page image from the same file, so the
    sample needs a real text layer, not a screenshot. Pages stay card-size
    (340x150 pt): the layout encoder's memory scales with page pixels, and
    A4 pages at its fixed 144 dpi want ~8 GB on CPU."""
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


def pdf_layout(model):
    banner("7. parse_pdf - layout-aware extraction over a PDF document")
    # the tour cleans up after itself; build_sample_pdf keeps its mkdtemp
    # default for direct one-off use (the PDF tests rely on it)
    with tempfile.TemporaryDirectory(prefix="gliformer-pdf-") as directory:
        path = build_sample_pdf(directory)
        print(f"  generated sample: {path}")
        print("  (the plain-text stops above never touch layout; this path reads")
        print("   words + boxes + a rendered page image from the PDF itself)")
        res = timed(
            model.parse_pdf,
            path,
            entities=["company", "person", "money", "email", "date"],
            return_pages=True,
        )
    print(f"  result keys: {sorted(res)}")
    pages = res.get("pages") or []
    for page_no, page_ents in enumerate(res.get("ner") or []):
        label = (f"page {pages[page_no]}" if page_no < len(pages)
                 else f"page {page_no}")
        for ent in page_ents:
            print(f"  {label}: {ent['text']!r:<26} "
                  f"{ent['label']:<10} {ent['score']:.3f}")


SECTIONS = [entities, classification, relations, records, multitask,
            embeddings, pdf_layout]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="GLiFormer showcase")
    parser.add_argument("--model", choices=MODELS, default="large")
    args = parser.parse_args()

    sizes = {"base": "~0.8 GB", "large": "~2.3 GB"}
    print(f"Loading {MODELS[args.model]} "
          f"(first run downloads {sizes[args.model]})...")
    t0 = time.perf_counter()
    from gliformer import GLiFormer

    model = GLiFormer.from_pretrained(MODELS[args.model], load_tokenizer=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Loaded on {device} in {time.perf_counter() - t0:.1f}s")

    for section in SECTIONS:
        try:
            section(model)
        except Exception as exc:  # keep the tour going if one API drifts
            print(f"  [skipped] {section.__name__}: {type(exc).__name__}: {exc}")

    print("\nDone. Compare with GLiNER 2.5:  python demos/demo.py\n")


if __name__ == "__main__":
    main()
