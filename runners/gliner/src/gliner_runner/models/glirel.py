"""GLiREL — jackboyla's zero-shot RELATION extractor (GLiNER lineage).

Serves the catalog checkpoint (catalog/models/jackboyla/glirel.yaml,
`runner: gliner` deployment):
  - glirel-large-v0.local -> jackboyla/glirel-large-v0 (~467M, CC BY-NC-SA 4.0)

Track: relation only. GLiREL scores (head, tail) entity pairs against
free-text relation labels but does NOT find entities itself — it needs
entity spans as input (its catalog note). The call paths are ported from
legacy/app.py's GLiREL tab and legacy/demos/demo_glirel.py:

  - its ner format is [start_token, end_token_INCLUSIVE, type, text] over
    GLiREL's own regex tokenization (model.py) — the same TOKEN_RE here so
    the token indices we hand it stay aligned with its head_pos/tail_pos
    output
  - predict_relations(tokens, labels, threshold, ner=ner, top_k) and the
    {relation, head, head_type, tail, tail_type, score} row mapping, where
    head_type/tail_type come back through GLiREL's exclusive-end
    (ent[0], ent[1] + 1) lookup
  - rows sorted by score, descending

Entities come from the question when it carries them (preferred — the bench
controls the spans): "ner" in GLiREL token format, or "entities" as char
spans. Otherwise the island lazily loads the relex checkpoint as the span
producer (classic gliner NER). Legacy fed it GLiNER 2.5 spans, which this
island does not serve — pass entities explicitly for parity with v0 demos.
"""

from __future__ import annotations

import re
from typing import Any

from gliner_runner.models._common import as_text

# GLiREL's own tokenizer shortcut (model.py): the same regex must be used
# here so the token indices we hand it line up with its head_pos/tail_pos.
TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

# the legacy tab's default entity label set (company, person, ...)
GLIREL_ENTITY_LABELS = ["company", "person", "product", "location", "organization"]

RELATION_THRESHOLD = 0.3  # the legacy tab's slider default
DEFAULT_TOP_K = 2  # the legacy tab's predict_relations top_k

# the classic-gliner checkpoint this island falls back to for entity spans
# (GLiNER 2.5, which legacy used, is not served here)
RELEX_SPAN_REPO = "knowledgator/gliner-relex-multi-v1.0"

# Weights live here after the first load; tests stub entries in this dict
# instead of touching the network. Never populated at import time.
_LOADED: dict[str, Any] = {}


def load(repo: str) -> Any:
    """Load the GLiREL checkpoint (first use downloads ~1.7 GB into the HF
    cache) — never at import, never in tests without weights."""
    if repo not in _LOADED:
        from glirel import GLiREL

        _LOADED[repo] = GLiREL.from_pretrained(repo)
    return _LOADED[repo]


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokens plus character offsets, matching GLiREL's internal split."""
    tokens, offsets = [], []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def char_span_to_token_span(start: int, end: int,
                            offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    """First-overlapping..last-overlapping token indices for a char span."""
    idx = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
    return (idx[0], idx[-1]) if idx else None


def char_spans_to_ner(text: str,
                      entities: list[dict]) -> tuple[list[str], list[list]]:
    """Char-span entities ({"start", "end", "label", "text"?}) -> GLiREL's
    ner format ([start, end, type, text], end index inclusive)."""
    tokens, offsets = tokenize(text)
    ner = []
    for item in entities:
        span = char_span_to_token_span(int(item["start"]), int(item["end"]), offsets)
        if span:
            ner.append([span[0], span[1], str(item["label"]),
                        str(item.get("text") or text[int(item["start"]):int(item["end"])])])
    return tokens, ner


def _entities_for(model: Any, text: str, question: dict[str, Any],
                  params: dict[str, Any]) -> tuple[list[str], list[list]]:
    """GLiREL's entity input: handed-in spans win, then the relex
    checkpoint's classic-gliner NER (lazy second model)."""
    handed = question.get("ner") or params.get("ner")
    if handed is not None:
        tokens, _offsets = tokenize(text)
        return tokens, [list(ent) for ent in handed]
    char_spans = question.get("entities") or params.get("entities")
    if char_spans is not None:
        return char_spans_to_ner(text, list(char_spans))
    # fallback: classic gliner NER over the relex checkpoint (never loaded
    # at import; cached in gliner_relex._LOADED like every other model)
    from gliner_runner.models import gliner_relex

    span_model = gliner_relex.load(RELEX_SPAN_REPO)
    labels = question.get("entity_labels") or question.get("labels") or GLIREL_ENTITY_LABELS
    if not isinstance(labels, list):
        raise ValueError("glirel expects `question.entity_labels` as a list of strings")
    threshold = float(params.get("entity_threshold", 0.4))
    entities = span_model.predict_entities(text, list(labels), threshold=threshold)
    return char_spans_to_ner(text, list(entities))


def run_relation(model: Any, input_: Any, question: dict[str, Any],
                 params: dict[str, Any]) -> tuple[list[dict], None]:
    """Score every (head, tail) entity pair against free-text relation
    labels; rows sorted by score, descending (the legacy tab's mapping).

    Question: {"relation_labels": [...]} (or "labels"), entity spans via
    "ner"/"entities"/"entity_labels" (see _entities_for). Params:
    threshold (default 0.3), top_k (default 2), entity_threshold for the
    fallback span producer.
    """
    text = as_text(input_, "glirel relations")
    labels = (question.get("relation_labels") or question.get("labels"))
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("glirel expects `question.relation_labels` as a list of strings")
    tokens, ner = _entities_for(model, text, question, params)
    if len(ner) < 2:
        # GLiREL needs at least two entity spans to form a (head, tail) pair
        return [], None
    threshold = float(params.get("threshold", RELATION_THRESHOLD))
    top_k = int(params.get("top_k", DEFAULT_TOP_K))
    relations = model.predict_relations(tokens, list(labels), threshold=threshold,
                                        ner=ner, top_k=top_k)
    # GLiREL reports head_pos/tail_pos with an EXCLUSIVE end; the ner rows
    # we handed it use the inclusive end, hence the +1 (app.py mapping).
    type_of = {(ent[0], ent[1] + 1): ent[2] for ent in ner}
    rows = [{"relation": rel["label"],
             "head": " ".join(rel["head_text"]),
             "head_type": type_of.get(tuple(rel["head_pos"]), "?"),
             "tail": " ".join(rel["tail_text"]),
             "tail_type": type_of.get(tuple(rel["tail_pos"]), "?"),
             "score": round(rel["score"], 3)}
            for rel in relations]
    return sorted(rows, key=lambda r: r["score"], reverse=True), None
