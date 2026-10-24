"""Shared helpers for the model modules (no model imports here)."""

from __future__ import annotations

from typing import Any


def as_text(value: Any, kind: str) -> str:
    """Coerce a request input to the plain string the encoders want.

    `input` is untyped on the wire (any JSON); every extractor here works
    on raw text, so anything else is a caller bug and gets a clear error.
    """
    if isinstance(value, str):
        return value
    raise ValueError(f"{kind} expects a text string as `input`, got {type(value).__name__}")


def label_keys(question: dict[str, Any]) -> list[str]:
    """The label names a classification call should use.

    The benches pass label DESCRIPTIONS as a dict ({label: description}) and
    the legacy engines scored the bare keys only (bench_spectrum.py builds
    `list(SENTIMENT_LABELS)` — descriptions are never shown to these
    encoders). A plain list passes through unchanged.
    """
    labels = question.get("labels")
    if isinstance(labels, dict):
        return list(labels)
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        return list(labels)
    raise ValueError("`question.labels` must be a list of strings or a {label: description} dict")
