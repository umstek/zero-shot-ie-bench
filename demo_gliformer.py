"""GLiFormer demo - knowledgator/gliformer-large-v1 tour (runs next to demo.py).

Same sample texts as demo.py (GLiNER 2.5) so outputs can be compared directly.
GLiFormer is Knowledgator's layout-aware multi-task encoder:
  1. entity extraction            (predict_entities)
  2. text classification          (classify, incl. named label groups)
  3. joint relation extraction    (inference + joint_relations)
  4. structured records           (structure, incl. nested Pydantic schemas)
  5. several tasks in one call    (inference)
  6. text embeddings              (embed_text)

Model: https://huggingface.co/knowledgator/gliformer-large-v1
Docs:   https://github.com/Knowledgator/GLiFormer

Run:
    python demo_gliformer.py                 # large (575.6M, English)
    python demo_gliformer.py --model base    # base checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
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


SECTIONS = [entities, classification, relations, records, multitask, embeddings]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="GLiFormer showcase")
    parser.add_argument("--model", choices=MODELS, default="large")
    args = parser.parse_args()

    print(f"Loading {MODELS[args.model]} (first run downloads ~2.3 GB for large)...")
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

    print("\nDone. Compare with GLiNER 2.5:  python demo.py\n")


if __name__ == "__main__":
    main()
