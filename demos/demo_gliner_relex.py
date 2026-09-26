"""GLiNER-relex demo - knowledgator/gliner-relex-multi-v1.0 tour.

Knowledgator's zero-shot JOINT NER + relation extractor from the classic
`gliner` line: one mDeBERTa-v3 pass predicts entity spans and scores every
(head, tail) pair against free-text relation labels, so no separate entity
model is needed (unlike GLiREL, which is handed spans). Multilingual -
unlike GLiREL's English DeBERTa backbone, scores hold up on European
languages. Runs through the `gliner` package's meta GLiNER class, which
auto-detects the relex architecture from the checkpoint config:
  1. entities only          (return_relations=False)
  2. joint entities + relations, relation predictions filtered by
                            entity-type constraints (applied post-hoc - the
                            model itself takes plain relation strings)
  3. multilingual texts     (mDeBERTa backbone - scores stay high)
  4. clinical labels        (free-form domain labels, in-distribution
                            enough to score - contrast with demo_glirel)

Model: https://huggingface.co/knowledgator/gliner-relex-multi-v1.0
       (~319M measured, Apache 2.0)
Docs:  https://github.com/Knowledgator/gliner (package source)

Run:
    python demos/demo_gliner_relex.py                  # threshold 0.4
    python demos/demo_gliner_relex.py --threshold 0.5  # stricter
"""

from __future__ import annotations

import argparse
import sys
import time

MODEL_ID = "knowledgator/gliner-relex-multi-v1.0"


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def extract(model, text: str, entity_labels: list[str],
            relation_labels: list[str], threshold: float,
            flat_ner: bool = False, return_relations: bool = True):
    """One inference call; relations sorted by score (they come per text)."""
    result = timed(
        model.inference,
        texts=[text], labels=entity_labels, relations=relation_labels,
        threshold=threshold, flat_ner=flat_ner,
        return_relations=return_relations,
    )
    if isinstance(result, tuple):
        entities, relations = result
        relations = sorted(relations[0], key=lambda r: r["score"], reverse=True)
    else:  # return_relations=False hands back just the per-text entity list
        entities, relations = result, []
    return entities[0], relations


def print_entities(entities) -> None:
    if not entities:
        print("  (no entities scored above the threshold)")
        return
    for ent in entities:
        print(f"  [{ent['start']:>3}:{ent['end']:<3}] {ent['text']!r:<24} "
              f"{ent['label']:<12} {ent['score']:.3f}")


def print_relations(relations, types: bool = True) -> None:
    if not relations:
        print("  (no relations scored above the threshold - nothing to print)")
        return
    for rel in relations:
        head, tail = rel["head"], rel["tail"]
        if types:
            print(f"  {head['text']!r} ({head['type']}) "
                  f"-[{rel['relation']}]-> {tail['text']!r} ({tail['type']})  "
                  f"{rel['score']:.3f}")
        else:
            print(f"  {head['text']!r} -[{rel['relation']}]-> "
                  f"{tail['text']!r}  {rel['score']:.3f}")


# ---------------------------------------------------------------- showcase 1
def entities_only(model, threshold: float) -> None:
    """return_relations=False skips the relation head entirely."""
    banner("1a. Entities only - the canonical sample sentence")
    text = "Alice works for Acme in Paris."
    print(f'  text: "{text}"')
    entities, _ = extract(model, text, ["person", "company", "location"], [],
                          threshold, return_relations=False)
    print_entities(entities)

    banner("1b. Entities only - the Apple sample sentence (flat vs nested)")
    text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
    print(f'  text: "{text}"')
    entities, _ = extract(model, text,
                          ["company", "person", "product", "location"], [],
                          threshold, flat_ner=False)
    print("  flat_ner=False (overlapping spans allowed):")
    print_entities(entities)
    entities, _ = extract(model, text,
                          ["company", "person", "product", "location"], [],
                          threshold, flat_ner=True)
    print("  flat_ner=True (no overlaps - the duplicate 'iPhone' goes):")
    print_entities(entities)


# ---------------------------------------------------------------- showcase 2
def joint_entities_relations(model, threshold: float) -> None:
    """One pass predicts entities AND relations; the relation dicts carry
    head/tail types straight from the entity label set. allowed_head/
    allowed_tail are a wrapper-level filter, not model input (the model
    takes plain relation strings) - same post-hoc filter as the GLiREL
    demo's spaCy-wrapper showcase."""
    banner("2. Joint entities + relations, entity-type-constrained")
    text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
    print(f'  text: "{text}"')
    constraints = {
        "works for": {"allowed_head": ["person"], "allowed_tail": ["company"]},
        "ceo of": {"allowed_head": ["person"], "allowed_tail": ["company"]},
        "announced": {"allowed_head": ["person"], "allowed_tail": ["product"]},
        "produced by": {"allowed_head": ["product"], "allowed_tail": ["company"]},
        "headquartered in": {"allowed_head": ["company"],
                             "allowed_tail": ["location"]},
    }
    entities, relations = extract(model, text,
                                  ["company", "person", "product", "location"],
                                  list(constraints), threshold)
    print("  entities:")
    print_entities(entities)
    print("  relations:")
    print_relations(relations, types=False)
    # constraint dicts that omit a key keep every type of the other rules
    all_heads = sorted({t for c in constraints.values()
                        for t in c.get("allowed_head", [])})
    all_tails = sorted({t for c in constraints.values()
                        for t in c.get("allowed_tail", [])})
    kept = [rel for rel in relations
            if rel["head"]["type"] in constraints.get(
                rel["relation"], {}).get("allowed_head", all_heads)
            and rel["tail"]["type"] in constraints.get(
                rel["relation"], {}).get("allowed_tail", all_tails)]
    dropped = [rel for rel in relations if rel not in kept]
    print("  kept (types match allowed_head/allowed_tail):")
    print_relations(kept)
    if dropped:
        print("  dropped by the constraints (would mislead without them):")
        print_relations(dropped)
    if not kept:
        print("  (every prediction was dropped by the constraints)")


# ---------------------------------------------------------------- showcase 3
def multilingual(model, threshold: float) -> None:
    """English labels on non-English text. The mDeBERTa backbone keeps
    scores high - the opposite of GLiREL's English-only sag."""
    banner("3a. Multilingual - Spanish (English labels)")
    text = "Miguel de Cervantes escribió Don Quijote en 1605."
    print(f'  text: "{text}"')
    _, relations = extract(model, text, ["person", "book", "date"],
                           ["wrote", "author", "publication date"], threshold)
    print_relations(relations)

    banner("3b. Multilingual - German (English labels)")
    text = "Angela Merkel war Bundeskanzlerin von Deutschland."
    print(f'  text: "{text}"')
    _, relations = extract(model, text, ["person", "country", "position"],
                           ["position held", "country of citizenship"],
                           threshold)
    print_relations(relations)

    banner("3c. Multilingual - French (English labels)")
    text = "Emmanuel Macron est le président de la France."
    print(f'  text: "{text}"')
    _, relations = extract(model, text, ["person", "country", "position"],
                           ["president of", "head of state"], threshold)
    print_relations(relations)


# ---------------------------------------------------------------- showcase 4
def clinical(model, threshold: float) -> None:
    """Same clinical sentence as demo.py / demo_gliformer.py /
    demo_glirel.py. Free-form labels score here (the joint relex training
    generalizes further than GLiREL's pair scorer did), so this is not the
    graceful-empty case."""
    banner("4. Clinical labels - free-form domain example")
    text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
    print(f'  text: "{text}"')
    entities, relations = extract(model, text,
                                  ["medication", "dosage", "symptom", "time"],
                                  ["has dosage", "treats",
                                   "time of administration"], threshold)
    print("  entities:")
    print_entities(entities)
    print("  relations:")
    print_relations(relations)


SECTIONS = [entities_only, joint_entities_relations, multilingual, clinical]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="GLiNER-relex showcase")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="score cutoff for entities and relations")
    args = parser.parse_args()

    print(f"Loading GLiNER-relex ({MODEL_ID}, ~319M; first run downloads "
          "~1.3 GB)...")
    t0 = time.perf_counter()
    from gliner import GLiNER

    model = GLiNER.from_pretrained(MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    for section in SECTIONS:
        try:
            section(model, args.threshold)
        except Exception as exc:  # keep the tour going if one API drifts
            print(f"  [skipped] {section.__name__}: {type(exc).__name__}: {exc}")

    print("\nDone. For comparison: python demos/demo.py (GLiNER 2.5 joint "
          "relations), python demos/demo_glirel.py (pair scorer fed "
          "external spans)\n")


if __name__ == "__main__":
    main()
