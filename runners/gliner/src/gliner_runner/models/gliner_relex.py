"""GLiNER-relex + the classic-gliner surface — Knowledgator's gliner line.

Serves the catalog checkpoint (catalog/models/knowledgator/gliner-relex.yaml,
`runner: gliner` deployment):
  - gliner-relex-multi-v1.0.local -> knowledgator/gliner-relex-multi-v1.0 (~319M)

Tracks: ner + relation in ONE pass (no external entity model, unlike GLiREL)
through the checkpoint's `inference` API, plus a census-row classify that
rides the vendor multitask prompt.

This module is also the island's classic-`gliner` home: the vendor multitask
classification mapping and the fp32-CPU load path are ported from
legacy/engines/glinerx_client.py (Knowledgator's GLiNER-X client). The
GLiNER-X checkpoints themselves are culled and stay in their own legacy venv
(`gliner[stanza]` wants transformers 5.x, incompatible with the island's
validated 4.57.6 pin) — only their loading/mapping logic lives on here:
  - NER:      predict_entities / inference(entities), threshold 0.5, the
              (start, end, label) char-offset tuples every extractor returns
  - classify: the gliner.multitask.GLiNERClassifier prompt shape and its
              single-label reduction — census rows only, the checkpoints are
              NER/relations-only and every mapping probed in v0 collapsed
              onto one label (see the glinerx_client docstring)
  - relation: inference(relations=...) with optional post-hoc head/tail
              entity-type constraints (the app.py GLiNER-relex tab's
              parse_relation_specs + type filter, minus its UI wrapper)
"""

from __future__ import annotations

import json
from typing import Any

from gliner_runner.models._common import as_text, label_keys

# the gliner.multitask classification prompt (GLiNERClassifier.prompt),
# the vendor's documented classification surface for classic GLiNER
CLASSIFICATION_PROMPT = "Classify text into the following classes: {} \n {}"

CLASSIFICATION_THRESHOLD = 0.5  # the multitask pipeline's default
NER_THRESHOLD = 0.5  # predict_entities' default, same as the GLiFormer bench
ENTITY_THRESHOLD = 0.4  # the app.py GLiNER-relex tab's entity slider default
RELATION_THRESHOLD = 0.7  # the tab's relation default (card suggests 0.7-0.9)

# Weights live here after the first load; tests stub entries in this dict
# instead of touching the network. Never populated at import time. GLiREL's
# default entity producer reuses this cache (see glirel.py).
_LOADED: dict[str, Any] = {}


def load(repo: str) -> Any:
    """Load one classic-gliner checkpoint on the CPU (fp32, eval mode).

    First use downloads the checkpoint into the HF cache (~1.3 GB for the
    relex checkpoint) — never at import, never in tests without weights.
    """
    if repo not in _LOADED:
        from gliner import GLiNER  # the classic gliner package (>=0.2.29)

        _LOADED[repo] = GLiNER.from_pretrained(repo).to("cpu").eval()
    return _LOADED[repo]


# --------------------------------------------------------------------------
# the vendor multitask classification mapping (ported from glinerx_client.py)
# --------------------------------------------------------------------------


def classification_prompt(labels: list[str], text: str) -> str:
    """The vendor multitask classification prompt: labels listed in the
    header, the text below (GLiNERClassifier.prepare_texts' exact shape)."""
    return CLASSIFICATION_PROMPT.format(", ".join(labels), text)


def reduce_classification(entities: list[dict], labels: list[str]) -> str:
    """GLiNERClassifier.process_predictions' single-label reduction:
    the highest-scoring entity's text becomes the prediction (whatever
    it says — the benches score it against the gold label either way);
    with no entities above the threshold the vendor answers the literal
    'other'."""
    if not entities:
        return "other"
    return max(entities, key=lambda entity: entity["score"])["text"].strip()


def run_classify(model: Any, input_: Any, question: dict[str, Any],
                 params: dict[str, Any]) -> tuple[str, float | None]:
    """Census-row classification through the vendor multitask mapping.

    The relex checkpoints' card claims NER + relations, never
    classification; like the GLiNER-X benches, the answer is whatever the
    top-scoring entity's text says (scored against the gold label either
    way) and rows are census rows, never a competitive classifier.
    """
    labels = label_keys(question)
    text = as_text(input_, "gliner classify")  # validated before any model call
    entities = model.predict_entities(
        classification_prompt(labels, text), labels,
        threshold=CLASSIFICATION_THRESHOLD)
    prob = max((e.get("score") for e in entities), default=None)
    return reduce_classification(entities, labels), prob


# --------------------------------------------------------------------------
# NER + relations through the relex checkpoint's inference API (app.py tab)
# --------------------------------------------------------------------------


def run_ner(model: Any, input_: Any, question: dict[str, Any],
            params: dict[str, Any]) -> tuple[list[tuple[int, int, str]], None]:
    """Entity spans from one inference pass (relations disabled): sorted,
    deduped (start, end, label) char-offset tuples."""
    labels = question.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("gliner NER expects `question.labels` as a list of strings")
    threshold = float(params.get("threshold", NER_THRESHOLD))
    text = as_text(input_, "gliner NER")  # validated before any model call
    result = model.inference(texts=[text], labels=labels,
                             relations=[], threshold=threshold,
                             return_relations=False)
    spans = sorted({(int(e["start"]), int(e["end"]), str(e["label"])) for e in result[0]})
    return spans, None


def parse_relation_specs(
    specs: Any,
) -> tuple[list[str], dict[str, tuple[str | None, str | None]]]:
    """Parse relation specs — a JSON list of {"name", "head"?, "tail"?}
    dicts or plain strings (a JSON-encoded string is accepted too, the
    legacy tab's textbox format) — into ((names, constraints), error).

    The spec's optional head/tail entity-type constraints apply post-hoc
    (run_relation); None names mean "any type".
    """
    if isinstance(specs, str):
        try:
            specs = json.loads(specs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"relation spec is not valid JSON: {exc}") from exc
    if isinstance(specs, dict):
        specs = [specs]
    if not isinstance(specs, list):
        raise ValueError("relation spec must be a JSON list of dicts or strings")
    names, constraints = [], {}
    for i, spec in enumerate(specs, 1):
        if isinstance(spec, str):
            spec = {"name": spec}
        if not isinstance(spec, dict) or not str(spec.get("name", "")).strip():
            raise ValueError(f"relation {i}: expected a non-empty 'name'")
        name = str(spec["name"]).strip()
        names.append(name)
        head, tail = spec.get("head"), spec.get("tail")
        if head or tail:
            constraints[name] = (head or None, tail or None)
    if not names:
        raise ValueError("provide at least one relation")
    return names, constraints


def run_relation(model: Any, input_: Any, question: dict[str, Any],
                 params: dict[str, Any]) -> tuple[list[dict], None]:
    """Joint NER + relations from ONE pass, rows sorted by score.

    Question: {"relations": [spec...]} (see parse_relation_specs) plus
    "entity_labels" for the entity pass. Params: entity_threshold,
    relation_threshold (defaults are the legacy tab's slider defaults).
    Head/tail entity-type constraints are applied post-hoc; only rows that
    survive are returned (the legacy tab's {"kept", "dropped_by_constraints"}
    wrapper was UI bookkeeping, not a prediction).
    """
    names, constraints = parse_relation_specs(question.get("relations") or
                                              question.get("relation_labels"))
    labels = question.get("entity_labels") or question.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("relex relations expect `question.entity_labels` as a list of strings")
    text = as_text(input_, "gliner relations")
    entity_threshold = float(params.get("entity_threshold", ENTITY_THRESHOLD))
    relation_threshold = float(params.get("relation_threshold", RELATION_THRESHOLD))
    _, relations = model.inference(texts=[text], labels=labels, relations=names,
                                   threshold=entity_threshold,
                                   relation_threshold=relation_threshold,
                                   flat_ner=False)
    rows = [{"relation": rel["relation"],
             "head": rel["head"]["text"],
             "head_type": rel["head"]["type"],
             "tail": rel["tail"]["text"],
             "tail_type": rel["tail"]["type"],
             "score": round(rel["score"], 3)}
            for rel in relations[0]]

    def type_ok(row: dict) -> bool:
        head_c, tail_c = constraints.get(row["relation"], (None, None))
        return ((head_c is None or row["head_type"] == head_c)
                and (tail_c is None or row["tail_type"] == tail_c))

    return sorted((row for row in rows if type_ok(row)),
                  key=lambda r: r["score"], reverse=True), None
