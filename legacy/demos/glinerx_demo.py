"""GLiNER-X tour + one-shot runner — executes inside .venv-glinerx.

Knowledgator's GLiNER-X family (mT5 encoder, classic `gliner` package
with the stanza extra) needs its own dependency set — a fresh
gliner[stanza] resolve wants transformers 5.x — so the app's main venv
keeps its pinned stack and the GLiNER-X tab spawns this script in
`.venv-glinerx`, talking JSON over stdin/stdout (same pattern as the
ReLiK tab / demos/relik_demo.py).

Without stdin JSON (`--tour`) it runs a short scripted tour: entities in
English, Spanish and Chinese, then the classification caveat — these
checkpoints are NER-only, so the tour shows what the vendor multitask
prompt actually answers (a census-row mapping, see
engines/glinerx_client.py).

stdin:  {"texts": [str, ...], "labels": [str, ...],
         "model": "knowledgator/gliner-x-small" (optional,
                 default small; bench name like "GLiNER-X-base" works too)}
stdout: {"results": [{"text": str,
                      "spans": [{"start", "end", "label", "text",
                                 "score"}, ...],
                      "classification": [{"label", "score"}, ...]},
                     ...]}
        or {"error": "Type: message"}
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import sys

from engines.glinerx_client import (MODELS, classification_prompt,
                                    load_glinerx, reduce_classification,
                                    warm_splitter)

DEFAULT_LABELS = ["person", "company", "product", "location"]


def resolve_model_id(model: str | None) -> str:
    if not model:
        return MODELS["GLiNER-X-small"]
    return MODELS.get(model, model)  # bench name or raw HF id


def classify(model, text: str, labels: list[str]) -> list[dict]:
    """The vendor multitask mapping (census rows — NER-only checkpoints;
    see engines/glinerx_client.py for the probe notes)."""
    entities = model.predict_entities(
        classification_prompt(labels, text), labels, threshold=0.5)
    top = sorted(entities, key=lambda e: e["score"], reverse=True)
    return [{"label": reduce_classification(entities, labels),
             "score": round(top[0]["score"], 4)} if top else
            {"label": "other", "score": 0.0}]


def run_texts(model, texts: list[str], labels: list[str]) -> list[dict]:
    results = []
    for text in texts:
        spans = model.predict_entities(text, labels, threshold=0.5)
        results.append({
            "text": text,
            "spans": [{"start": e["start"], "end": e["end"],
                       "label": e["label"], "text": e["text"],
                       "score": round(e["score"], 4)} for e in spans],
            "classification": classify(model, text, labels),
        })
    return results


def tour() -> None:
    model = load_glinerx(MODELS["GLiNER-X-small"])
    print("loaded knowledgator/gliner-x-small (mT5 encoder, stanza word "
          "splitter)\n", flush=True)
    samples = [
        ("Ana Silva works at Microsoft in Lisbon and uses Azure every day.",
         DEFAULT_LABELS),
        ("María García trabaja en Telefónica en Madrid y utiliza Movistar.",
         DEFAULT_LABELS),
        ("在北京的腾讯公司发布了微信新版本。",
         ["person", "company", "product", "location"]),
    ]
    for text, labels in samples:
        print("text:", text)
        for e in model.predict_entities(text, labels, threshold=0.5):
            print(f"  {e['text']!r} -> {e['label']}  ({e['score']:.3f})")
    print("\nclassification (vendor multitask prompt — these checkpoints "
          "are NER-only, expect a collapsed census-row answer):")
    for text, gold in [("The food was cold and the waiter was rude.",
                        "negative"),
                       ("I absolutely love this coffee shop.", "positive")]:
        pred = classify(model, text, ["positive", "negative", "neutral"])
        print(f"  {text!r} -> {pred[0]['label']}  (gold {gold})")


def main() -> None:
    if sys.stdin.isatty() or "--tour" in sys.argv:
        tour()
        return
    payload = json.loads(sys.stdin.read())
    try:
        import os

        model_id = resolve_model_id(payload.get("model"))
        texts = [t for t in payload.get("texts", []) if t.strip()]
        labels = payload.get("labels") or DEFAULT_LABELS
        if not texts:
            raise ValueError("provide at least one text line")
        # model load and the lazy stanza pipeline builds log to stderr,
        # but move fd 1 onto stderr anyway so this script's stdout stays
        # pure JSON even if a dependency prints to stdout (same guard as
        # demos/relik_demo.py)
        saved_stdout = os.dup(1)
        try:
            os.dup2(2, 1)
            model = load_glinerx(model_id)
            warm_splitter(model, texts)
            results = run_texts(model, texts, labels)
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
