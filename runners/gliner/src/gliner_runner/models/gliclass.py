"""GLiClass — Knowledgator's zero-shot classification line.

Serves the three benched checkpoints (catalog/models/knowledgator/gliclass-*.yaml,
all `runner: gliner` deployments):
  - gliclass-base-v3.0.local         -> knowledgator/gliclass-base-v3.0 (187M)
  - gliclass-modern-base-v3.0.local  -> knowledgator/gliclass-modern-base-v3.0 (151M)
  - gliclass-large-v3.0.local        -> knowledgator/gliclass-large-v3.0 (439M)

Track: classify only — GLiClass cannot extract spans. The call path is
legacy/bench_spectrum.py's classify_gliclass verbatim: the gliclass
ZeroShotClassificationPipeline (multi-label) over the bare label keys,
threshold 0.0 so every label scores, then argmax — the pipeline's own
thresholding is too aggressive for the pools. The v3 checkpoints run fine
on the island's transformers 4.57.6 pin even though GLiClass 0.1.20
declares transformers>=5 (legacy/overrides.txt; the override stays tied to
that exact release in pyproject.toml).
"""

from __future__ import annotations

from typing import Any

from gliner_runner.models._common import as_text, label_keys

CLASSIFICATION_THRESHOLD = 0.0  # the bench's pipe threshold: score all, argmax

# Weights live here after the first load; tests stub entries in this dict
# instead of touching the network. Never populated at import time.
_LOADED: dict[str, Any] = {}


def load(repo: str) -> Any:
    """Build the GLiClass pipeline for one checkpoint (model + tokenizer,
    multi-label, CPU). First use downloads the checkpoint into the HF
    cache — never at import, never in tests without weights."""
    if repo not in _LOADED:
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        model = GLiClassModel.from_pretrained(repo)
        _LOADED[repo] = ZeroShotClassificationPipeline(
            model, AutoTokenizer.from_pretrained(repo),
            classification_type="multi-label", device="cpu")
    return _LOADED[repo]


def run_classify(model: Any, input_: Any, question: dict[str, Any],
                 params: dict[str, Any]) -> tuple[str | None, float | None]:
    """One forward pass scores every label; argmax wins, None when the
    pipe returns nothing (bench semantics: `max(out, key=score)` over the
    threshold-0.0 output)."""
    labels = label_keys(question)
    threshold = float(params.get("threshold", CLASSIFICATION_THRESHOLD))
    text = as_text(input_, "gliclass classify")  # validated before any model call
    out = model(text, labels, threshold=threshold)[0]
    if not out:
        return None, None
    best = max(out, key=lambda x: x["score"])
    return best["label"], best["score"]
