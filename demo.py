"""GLiNER 2.5 demo - a guided tour of zero-shot information extraction.

One boundary-architecture model, six tasks, no fine-tuning:
  1. entity extraction (plain labels + labels with descriptions)
  2. text classification (single-label and multi-label aspects)
  3. relation extraction
  4. joint entity+relation extraction (globally consistent graph)
  5. span attributes (sentiment attached to people, not organizations)
  6. structured records (who bought what, kept as records)

Model family: fastino/gliner2.5-{small,base,multi}-v1
Docs: https://github.com/fastino-ai/GLiNER2

Run:
    python demo.py                 # base checkpoint (194M, English)
    python demo.py --model small   # 74M, fastest on CPU
    python demo.py --model multi   # 287M, multilingual
"""

from __future__ import annotations

import argparse
import json
import sys
import time

MODELS = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}

MODEL_ID = MODELS["base"]  # set in main(); sections that reload use this


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  ({dt:.2f}s on this machine)")
    return result


# ---------------------------------------------------------------- showcase 1
def entities(model):
    banner("1a. Entity extraction - plain labels")
    text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
    print(f'  text: "{text}"')
    result = timed(
        model.extract_entities,
        text,
        ["company", "person", "product", "location"],
        include_confidence=True,
        include_spans=True,
    )
    show(result.get("entities"))

    banner("1b. Entity extraction - labels with descriptions (clinical domain)")
    text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
    print(f'  text: "{text}"')
    result = timed(
        model.extract_entities,
        text,
        {
            "medication": "Names of drugs or pharmaceutical substances",
            "dosage": "Amounts such as 400mg, 2 tablets, or 5ml",
            "symptom": "Reported symptoms or conditions",
            "time": "Clock times or relative times",
        },
        include_spans=True,
    )
    show(result.get("entities"))


# ---------------------------------------------------------------- showcase 2
def classification(model):
    banner("2a. Text classification - single label")
    result = timed(
        model.classify_text,
        "This laptop has amazing performance but terrible battery life!",
        {"sentiment": ["positive", "negative", "neutral"]},
    )
    show(result)

    banner("2b. Text classification - multi-label aspects")
    result = timed(
        model.classify_text,
        "Great camera quality, decent performance, but poor battery life.",
        {
            "aspects": {
                "labels": ["camera", "performance", "battery", "display", "price"],
                "multi_label": True,
                "cls_threshold": 0.4,
            }
        },
    )
    show(result)


# ---------------------------------------------------------------- showcase 3
def relations(model):
    banner("3. Relation extraction - independent decoding")
    text = "Alice works for Acme in Paris."
    print(f'  text: "{text}"')
    result = timed(
        model.extract_relations,
        text,
        ["works_for", "located_in"],
        include_spans=True,
        include_confidence=True,
    )
    show(result.get("relation_extraction"))


# ---------------------------------------------------------------- showcase 4
def joint(model):
    banner("4. Joint entity + relation extraction - typed, globally consistent")
    from gliner2.joint_ie import JointIE, JointIEConfig

    joint = JointIE.from_pretrained(MODEL_ID)
    schema = (
        joint.create_schema()
        .entities(["person", "organization", "location"])
        .relation("works_for", "person", "organization", unique_head=True)
        .relation("located_in", "organization", "location")
        .no_self_loops()
    )
    text = "Alice works for Acme in Paris. Bob joined Acme last year."
    print(f'  text: "{text}"')
    result = timed(
        joint.extract,
        text,
        schema,
        config=JointIEConfig(optimizer="beam", beam_size=32),
    )
    print(f"  feasible: {result.feasible}")
    for rel in result.relations:
        head = result.entity(rel.head)
        tail = result.entity(rel.tail)
        print(f"  {head.text} -{rel.type}-> {tail.text}")


# ---------------------------------------------------------------- showcase 5
def attributes(model):
    banner("5. Span attributes - sentiment attached to people only")
    from gliner2 import AttributeGroup

    schema = (
        model.create_schema()
        .entities(["person", "organization"])
        .entity_attributes(
            {
                "sentiment": AttributeGroup(
                    ["positive", "negative", "neutral"],
                    applies_to=["person"],
                    qualify_labels=True,
                )
            }
        )
    )
    text = "Alice praised Microsoft, but Bob criticized OpenAI."
    print(f'  text: "{text}"')
    result = timed(
        model.extract,
        text,
        schema,
        include_spans=True,
        include_confidence=True,
    )
    show(result.get("entities"))


# ---------------------------------------------------------------- showcase 6
def records(model):
    banner("6. Structured records - instance identity kept (who bought what)")
    schema = (
        model.create_schema()
        .structure("purchase", mode="natural", anchor="buyer")
        .field("buyer", dtype="str", cardinality="required_one")
        .field("item", dtype="str", cardinality="required_one")
    )
    text = "Alice bought apples and Bob bought oranges."
    print(f'  text: "{text}"')
    result = timed(model.extract, text, schema)
    show(result)


SECTIONS = [entities, classification, relations, joint, attributes, records]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="GLiNER 2.5 showcase")
    parser.add_argument("--model", choices=MODELS, default="base")
    args = parser.parse_args()

    global MODEL_ID
    MODEL_ID = MODELS[args.model]

    print(f"Loading {MODEL_ID} (first run downloads the checkpoint)...")
    t0 = time.perf_counter()
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(MODEL_ID, map_location="cpu")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    for section in SECTIONS:
        try:
            section(model)
        except Exception as exc:  # keep the tour going if one API drifts
            print(f"  [skipped] {section.__name__}: {type(exc).__name__}: {exc}")

    print("\nDone. Try the interactive UI:  python app.py\n")


if __name__ == "__main__":
    main()
