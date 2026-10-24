"""GLiREL demo - jackboyla/glirel-large-v0 tour (runs next to demo.py).

Same sample texts as demo.py (GLiNER 2.5) so outputs can be compared directly.
GLiREL is a zero-shot RELATION extractor from the GLiNER lineage: a
label-prompted encoder (DeBERTa-v3-large, ~467M) that scores every
(head, tail) entity pair against free-text relation labels in one pass.
It needs entity spans as input, so this tour runs the repo's GLiNER 2.5
base checkpoint first and maps its character spans onto GLiREL's regex
tokens (its ner format is [start_token, end_token, type, text] with an
inclusive end index - and its predict_relations wants a token list,
because the string shortcut garbles head_text/tail_text):
  1. plain free-text relation labels   (predict_relations)
  2. entity-type-constrained labels    (allowed_head/allowed_tail, applied
                                       post-hoc like the spaCy wrapper)
  3. multilingual texts                (English-trained backbone - expect
                                       lower scores than English)
  4. clinical labels                   (graceful case: nothing scores high)

Model: https://huggingface.co/jackboyla/glirel-large-v0 (~467M, CC BY-NC-SA 4.0)
Docs:  https://github.com/jackboyla/GLiREL

Run:
    python demos/demo_glirel.py                    # threshold 0.3
    python demos/demo_glirel.py --threshold 0.1    # looser
"""

from __future__ import annotations

import argparse
import re
import sys
import time

# GLiREL's own tokenizer shortcut (model.py): same regex must be used here
# so the token indices we hand it line up with its head_pos/tail_pos.
TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokens plus character offsets, matching GLiREL's internal split."""
    tokens, offsets = [], []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def char_span_to_token_span(start: int, end: int,
                            offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    idx = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
    return (idx[0], idx[-1]) if idx else None


def ner_from_gliner(gmodel, text: str, entity_labels: list[str]) -> tuple[
        list[str], list[list]]:
    """GLiNER 2.5 char spans -> GLiREL ner ([start, end, type, text],
    end index inclusive)."""
    result = gmodel.extract_entities(text, entity_labels,
                                     include_confidence=True,
                                     include_spans=True)
    tokens, offsets = tokenize(text)
    ner = []
    for label, items in result.get("entities", {}).items():
        for item in items:
            span = char_span_to_token_span(item["start"], item["end"], offsets)
            if span:
                ner.append([span[0], span[1], label, item["text"]])
    return tokens, ner


def predict_sorted(gmodel, rel_model, text: str, entity_labels: list[str],
                   relation_labels: list[str], threshold: float, top_k: int = 1):
    """NER + relation extraction in one go, relations sorted by score."""
    tokens, ner = ner_from_gliner(gmodel, text, entity_labels)
    if len(ner) < 2:
        return tokens, ner, []
    relations = timed(
        rel_model.predict_relations,
        tokens, relation_labels,
        threshold=threshold, ner=ner, top_k=top_k,
    )
    relations.sort(key=lambda r: r["score"], reverse=True)
    return tokens, ner, relations


def print_relations(relations) -> None:
    if not relations:
        print("  (no relations scored above the threshold - nothing to print)")
        return
    for rel in relations:
        head, tail = " ".join(rel["head_text"]), " ".join(rel["tail_text"])
        print(f"  {head!r:<22} -[{rel['label']}]-> {tail!r:<22} "
              f"{rel['score']:.3f}")


# ---------------------------------------------------------------- showcase 1
def plain_labels(gmodel, rel_model, threshold: float) -> None:
    banner("1a. Free-text relation labels - the canonical sample sentence")
    text = "Alice works for Acme in Paris."
    print(f'  text: "{text}"')
    _, _, relations = predict_sorted(
        gmodel, rel_model, text,
        ["company", "person", "location"],
        ["works_for", "located_in", "lives in"],
        threshold,
    )
    print_relations(relations)

    banner("1b. Free-text relation labels - the Apple sample sentence")
    text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
    print(f'  text: "{text}"')
    tokens, ner, relations = predict_sorted(
        gmodel, rel_model, text,
        ["company", "person", "product", "location"],
        ["works_for", "located_in", "announced", "ceo_of", "produced by"],
        threshold,
    )
    print(f"  GLiNER entities (token indices, inclusive end): {ner}")
    print_relations(relations)


# ---------------------------------------------------------------- showcase 2
def constrained_labels(gmodel, rel_model, threshold: float) -> None:
    """allowed_head/allowed_tail are a wrapper-level filter, not model input:
    the spaCy integration predicts first, then drops pairs whose entity
    types don't match. Same post-hoc filter here, on our own ner list."""
    banner("2. Relation labels with entity-type constraints")
    text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday."
    print(f'  text: "{text}"')
    constraints = {
        "ceo_of": {"allowed_head": ["person"], "allowed_tail": ["company"]},
        "produced by": {"allowed_head": ["product"], "allowed_tail": ["company"]},
        "headquartered in": {"allowed_head": ["company"],
                             "allowed_tail": ["location"]},
        "located_in": {"allowed_head": ["company"], "allowed_tail": ["location"]},
    }
    tokens, ner, relations = predict_sorted(
        gmodel, rel_model, text,
        ["company", "person", "product", "location"],
        list(constraints),
        threshold,
    )
    # head_pos is [start, end_inclusive + 1] - same key shape the spaCy
    # wrapper builds from (ent.start, ent.end)
    type_of = {(ent[0], ent[1] + 1): ent[2] for ent in ner}
    kept, dropped = [], []
    for rel in relations:
        rule = constraints[rel["label"]]
        head_t = type_of.get(tuple(rel["head_pos"]), "?")
        tail_t = type_of.get(tuple(rel["tail_pos"]), "?")
        record = (rel, head_t, tail_t)
        # constraint dicts that omit a key keep every type, not none
        types = sorted({t for rule in constraints.values()
                        for key in ("allowed_head", "allowed_tail")
                        for t in rule.get(key, [])})
        if head_t in rule.get("allowed_head", types) and \
                tail_t in rule.get("allowed_tail", types):
            kept.append(record)
        else:
            dropped.append(record)

    print("  kept (types match allowed_head/allowed_tail):")
    for rel, head_t, tail_t in kept:
        print(f"    {' '.join(rel['head_text'])} ({head_t}) "
              f"-[{rel['label']}]-> {' '.join(rel['tail_text'])} ({tail_t})  "
              f"{rel['score']:.3f}")
    if dropped:
        print("  dropped by the constraints (would mislead without them):")
        for rel, head_t, tail_t in dropped:
            print(f"    {' '.join(rel['head_text'])} ({head_t}) "
                  f"-[{rel['label']}]-> {' '.join(rel['tail_text'])} ({tail_t})  "
                  f"{rel['score']:.3f}")
    if not kept:
        print("  (every prediction was dropped by the constraints)")


# ---------------------------------------------------------------- showcase 3
def multilingual(gmodel, rel_model, threshold: float) -> None:
    """English labels on non-English text. GLiREL's backbone is the
    English DeBERTa-v3-large, so scores sag - watch them next to
    showcase 1."""
    banner("3a. Multilingual - Spanish (English labels)")
    text = "Miguel de Cervantes escribió Don Quijote en 1605."
    print(f'  text: "{text}"')
    _, _, relations = predict_sorted(
        gmodel, rel_model, text,
        ["person", "book", "date"],
        ["wrote", "author", "publication date"],
        threshold,
    )
    print_relations(relations)

    banner("3b. Multilingual - German (English labels)")
    text = "Angela Merkel war Bundeskanzlerin von Deutschland."
    print(f'  text: "{text}"')
    _, _, relations = predict_sorted(
        gmodel, rel_model, text,
        ["person", "country", "position"],
        ["head of government", "position held", "country"],
        threshold,
    )
    print_relations(relations)


# ---------------------------------------------------------------- showcase 4
def clinical(gmodel, rel_model, threshold: float) -> None:
    """Same clinical sentence as demo.py / demo_gliformer.py. These labels
    sit outside GLiREL's training distribution - the graceful no-output
    case, which the demo prints instead of pretending."""
    banner("4. Clinical labels - outside the training distribution")
    text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
    print(f'  text: "{text}"')
    _, _, relations = predict_sorted(
        gmodel, rel_model, text,
        ["medication", "dosage", "symptom", "time"],
        ["treats", "has dosage", "symptom of", "time of administration"],
        threshold,
    )
    print_relations(relations)


SECTIONS = [plain_labels, constrained_labels, multilingual, clinical]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="GLiREL showcase")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="score cutoff for reported relations")
    args = parser.parse_args()

    print("Loading GLiREL (jackboyla/glirel-large-v0, ~467M; first run "
          "downloads ~1.7 GB) and GLiNER 2.5 base for the entity step...")
    t0 = time.perf_counter()
    from glirel import GLiREL
    from gliner2 import AutoExtractor

    rel_model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    gmodel = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1",
                                           map_location="cpu")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    for section in SECTIONS:
        try:
            section(gmodel, rel_model, args.threshold)
        except Exception as exc:  # keep the tour going if one API drifts
            print(f"  [skipped] {section.__name__}: {type(exc).__name__}: {exc}")

    print("\nDone. Compare with GLiNER 2.5's joint relations:  "
          "python demos/demo.py\n")


if __name__ == "__main__":
    main()
