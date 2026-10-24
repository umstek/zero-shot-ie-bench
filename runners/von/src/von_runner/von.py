"""The von-1.0 engine: loading + predict, ported from legacy/engines/von_client.py.

Same checkpoint pin (wfzyx/von-1.0 @ fbc0196…), same pinned Git SDK
revision (a94aa36…), same call path into the SDK
(evaluate_choice / evaluate_noul / evaluate_score with the calibration's
default temperature) so predictions match the legacy .venv-von runs.

The PyPI SDK's default NLI loader invents a random classifier for this
checkpoint. The pinned Git SDK has the correct architecture, but falls back
to a random scorer if option_marker.pt is absent. Never allow that fallback.

Heavy imports (torch, transformers, the SDK itself) stay inside the load
and wrapper functions so this module imports without them — offline tests
stub the SDK modules exactly like legacy tests/test_von_loading.py did.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

# HF identity of the checkpoint (download + provenance; legacy MODEL_ID).
MODEL_ID = "wfzyx/von-1.0"
MODEL_REVISION = "fbc0196efa4fa1825557a8d5adad464fa1e872bb"
SDK_REVISION = "a94aa368ff4dcc58cdfd569cf0b4b1ae7d592d54"
CHECKPOINT_FILES = [
    "config.json",
    "model.safetensors",
    "option_marker.pt",
    "tokenizer.json",
    "tokenizer_config.json",
    "marker_calibration.json",
]

# Wire identity served on /info and accepted as TrackRequest.model_id —
# the catalog model id (catalog/models/wfzyx/von.yaml), whose version
# "von-1.0@<MODEL_REVISION>" resolves to the deployment von-1.0.local.
SERVED_MODEL_ID = "wfzyx/von"

DEFAULT_CHOICE_INSTRUCTIONS = "Choose the best option."

_BACKEND: Any | None = None
_BACKEND_LOCK = threading.Lock()


def provenance() -> dict[str, str]:
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sdk_revision": SDK_REVISION,
        "backend": "option-marker",
    }


def cached_snapshot() -> Path | None:
    """The pinned checkpoint if it is already in the local HF cache.

    local_files_only=True — this never downloads. Returns None when the
    snapshot (or option_marker.pt inside it) is absent; tests gate on it
    so CI never pulls weights.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover — huggingface-hub is a hard dep
        return None
    try:
        checkpoint = Path(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=CHECKPOINT_FILES,
                local_files_only=True,
            )
        )
    except Exception:  # noqa: BLE001 — any cache miss means "not cached"
        return None
    if not (checkpoint / "option_marker.pt").is_file():
        return None
    return checkpoint


def load_von_backend() -> Any:
    """Download the pinned checkpoint and strictly load the backend once;
    the decide/judge/rate wrappers below share it. Heavy imports stay
    inside so the module imports without torch (offline tests)."""
    from huggingface_hub import snapshot_download

    try:
        from von.backends.option_marker_backend import OptionMarkerBackend
    except ImportError as exc:
        raise RuntimeError(
            "The von-sdk option-marker backend is missing; the island "
            "pins the Git SDK revision in pyproject.toml (the PyPI SDK "
            "does not include the trained option-marker loader)."
        ) from exc

    checkpoint = Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=CHECKPOINT_FILES)
    )
    if not (checkpoint / "option_marker.pt").is_file():
        raise RuntimeError("Refusing von inference: trained option_marker.pt weights are missing.")
    backend = OptionMarkerBackend(checkpoint_dir=str(checkpoint), device="cpu")
    # Pinned SDK uses weights_only=True and strict load_state_dict here.
    # Eager loading both validates the trained head and excludes setup
    # from the first predict's timing.
    backend._get_model()
    return backend


def get_backend() -> Any:
    """Process-wide lazy singleton: the backend loads on the first predict
    (HTTP request), never at import. Lock-guarded because FastAPI runs sync
    handlers on a threadpool."""
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = load_von_backend()
        return _BACKEND


def reset_backend() -> None:
    """Drop the cached backend (tests swap in stubs between cases)."""
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None


def load_von_decider(backend: Any = None) -> Any:
    """Download and strictly load all weights before returning decide(),
    a thin wrapper over the SDK's evaluate_choice (choices is the
    {label: description-or-null} criteria dict)."""
    backend = backend if backend is not None else load_von_backend()
    from von.types import Choice

    def decide(*, state: str, choices: dict[str, Any], instructions: str) -> Any:
        return backend.evaluate_choice(
            "decision",
            state,
            Choice(instructions=instructions, criteria=choices),
            temperature=backend._default_temp,
        )

    return decide


def load_von_judge(backend: Any = None) -> Any:
    """Yes/no judgments: download and strictly load all weights before
    returning judge(state=..., instructions=...), a thin wrapper over the
    SDK's evaluate_noul. The answer's `noul` is P(condition true) —
    0.0-1.0, no separate argmax pick."""
    backend = backend if backend is not None else load_von_backend()
    from von.types import Noul

    def judge(*, state: str, instructions: str) -> Any:
        return backend.evaluate_noul(
            "judgment", state, Noul(instructions=instructions), temperature=backend._default_temp
        )

    return judge


def load_von_rate(backend: Any = None) -> Any:
    """Ordinal score rubrics: download and strictly load all weights
    before returning rate(state=..., rubric=[...], instructions=...), a
    thin wrapper over the SDK's evaluate_score. `rubric` is the ordered
    level list (lowest to highest, str or {what, examples} dicts); the
    answer's `score` is the probability-weighted expectation over the
    levels and `probabilities` is keyed by level index."""
    backend = backend if backend is not None else load_von_backend()
    from von.types import Score

    def rate(*, state: str, rubric: list[Any], instructions: str) -> Any:
        return backend.evaluate_score(
            "rating",
            state,
            Score(instructions=instructions, criteria=rubric),
            temperature=backend._default_temp,
        )

    return rate


# ------------------------------------------------------------ answer rows
# Ported from legacy demos/von_demo.py — the SDK answer -> row mappings the
# legacy UI and tests relied on.


def row(answer: Any) -> dict[str, Any]:
    """ChoiceAnswer -> row."""
    return {
        "choice": answer.choice,
        "probabilities": answer.probabilities,
        "confidence": answer.confidence,
    }


def judgment_row(answer: Any) -> dict[str, Any]:
    """NoulAnswer -> row: the yes-probability plus the derived yes/no
    verdict at the conventional 0.5 threshold."""
    return {"noul": answer.noul, "verdict": "yes" if answer.noul >= 0.5 else "no"}


def rating_row(answer: Any) -> dict[str, Any]:
    """ScoreAnswer -> row."""
    return {
        "score": answer.score,
        "confidence": answer.confidence,
        "legend": answer.legend,
        "probabilities": answer.probabilities,
    }


# ------------------------------------------------- payload -> SDK question
# The wire payload is the item's track-typed content (runner-protocol):
# {"input": {"text": ...} or str, "question": {"type": choice|noul|score,
# ...}, "params": {...} or None}. The suite item's question is a loose
# object (packages/schema/src/suite.ts), so instructions/rubric may ride on
# the question, on params, or be absent — each question type below says
# what it accepts and what it defaults to.


def _text_of(payload: dict[str, Any]) -> str:
    """The item's state text: input.text for suite-shaped items, the
    string itself for bare-string inputs. Verbatim — the legacy bench
    never stripped question texts."""
    source = payload.get("input")
    if isinstance(source, dict):
        source = source.get("text")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("payload needs a non-empty 'input' (or 'input.text')")
    return source


def _first_str(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _question_of(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, dict):
        raise ValueError("payload needs a 'question' object")
    return question


def _params_of(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params")
    return params if isinstance(params, dict) else {}


def _choices(payload: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """choice question -> the {label: description-or-null} criteria dict.
    Accepts options as a label list (descriptions default to null, the
    shape the suites ship) or as/through a {label: description} dict on
    question.choices / params.choices."""
    params = _params_of(payload)
    raw = (
        question.get("choices")
        or params.get("choices")
        or question.get("options")
        or params.get("options")
    )
    if isinstance(raw, dict):
        choices = dict(raw)
    elif isinstance(raw, list):
        choices = {label: None for label in raw}
    else:
        choices = {}
    if not choices:
        raise ValueError(
            "choice question needs a non-empty 'question.options' list (or a 'choices' object)"
        )
    return choices


def _rubric(payload: dict[str, Any], question: dict[str, Any]) -> list[Any]:
    """score question -> the ordered rubric levels, lowest to highest,
    passed through verbatim (str or {what, examples} dicts)."""
    raw = question.get("rubric") or _params_of(payload).get("rubric")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(
            "score question needs a 'rubric' of at least two levels, lowest to highest"
        )
    return raw


def _predict_choice(payload: dict[str, Any]) -> dict[str, Any]:
    question = _question_of(payload)
    state = _text_of(payload)
    instructions = _first_str(question.get("instructions"), _params_of(payload).get("instructions"))
    choices = _choices(payload, question)
    answer = load_von_decider(get_backend())(
        state=state, choices=choices, instructions=instructions or DEFAULT_CHOICE_INSTRUCTIONS
    )
    chosen = row(answer)
    # prediction = the label (the exact-choice scorer compares against the
    # gold label, as the legacy bench stored res.choice); prob = the
    # engine's calibrated confidence for that answer.
    return {"prediction": chosen["choice"], "prob": chosen["confidence"], "error": None}


def _predict_noul(payload: dict[str, Any]) -> dict[str, Any]:
    question = _question_of(payload)
    state = _text_of(payload)
    instructions = _first_str(question.get("instructions"), _params_of(payload).get("instructions"))
    if instructions is None:
        raise ValueError("noul question needs 'question.instructions'")
    answer = load_von_judge(get_backend())(state=state, instructions=instructions)
    judged = judgment_row(answer)
    # prediction = the derived verdict at the 0.5 threshold (legacy
    # judgment_row); prob = the raw P(condition true) for brier/ece.
    return {"prediction": judged["verdict"], "prob": judged["noul"], "error": None}


def _predict_score(payload: dict[str, Any]) -> dict[str, Any]:
    question = _question_of(payload)
    state = _text_of(payload)
    instructions = _first_str(question.get("instructions"), _params_of(payload).get("instructions"))
    if instructions is None:
        raise ValueError("score question needs 'question.instructions'")
    rubric = _rubric(payload, question)
    answer = load_von_rate(get_backend())(state=state, rubric=rubric, instructions=instructions)
    rated = rating_row(answer)
    # prediction = the probability-weighted expectation over the levels
    # (the SDK ScoreAnswer.score); prob = the engine's confidence.
    return {"prediction": rated["score"], "prob": rated["confidence"], "error": None}


_HANDLERS = {"choice": _predict_choice, "noul": _predict_noul, "score": _predict_score}


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """runner-protocol predict(model_id, payload) -> {prediction, prob, error}.

    Dispatches on question.type (choice -> evaluate_choice, noul ->
    evaluate_noul, score -> evaluate_score — the three System One question
    types the legacy demo mapped). Never raises: every failure becomes an
    {"error": "Type: message"} row so one bad item cannot kill a run,
    exactly like the legacy stdin runner's error JSON."""
    try:
        if model_id != SERVED_MODEL_ID:
            raise ValueError(
                f"unknown model_id: {model_id!r} (this runner serves {SERVED_MODEL_ID!r})"
            )
        question = _question_of(payload)
        qtype = str(question.get("type") or "").strip().lower()
        handler = _HANDLERS.get(qtype)
        if handler is None:
            raise ValueError(
                f"unsupported question type: {qtype!r} (von serves choice, noul and score)"
            )
        return handler(payload)
    except Exception as exc:  # noqa: BLE001 — per-item isolation by contract
        return {"prediction": None, "prob": None, "error": f"{type(exc).__name__}: {exc}"}
