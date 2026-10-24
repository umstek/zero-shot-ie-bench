"""The nanodiff runner's FastAPI app: the diffusion decision engine on the
shared protocol.

`GET /healthz`, `GET /info` and `POST /v1/decision` (tracks=["decision"] —
a single-letter choice interface has no span extraction or rubric, so the
engine claims nothing else). Two served models, the two typed-decisions
checkpoints the legacy benches ran:

- pngwn/nanodiff-350m-typed-decisions-lam1  (the v1 lam1 release)
- pngwn/nanodiff-350m-typed-decisions-v2    (the v2 retrain, same arch)

There are no catalog yamls for the pair (trial-level systems, culled
before v0 — TRIALS.md records the numbers), so the wire identity is the
HF repo id and the version tag names the step-3000 artifacts the
checkpoints carry.

No Gradio UI: the legacy nanodiff tour is covered by apps/playground
against this API. Weights load lazily on the first predict — importing
this module never downloads or loads a checkpoint.

Run: `uv run uvicorn nanodiff_runner.server:app --host 127.0.0.1 --port 8032`
or `uv run python -m nanodiff_runner.server [--host H] [--port P]`.
"""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI
from runner_protocol import create_app as make_runner_app

from . import engine

# Transcribed from the checkpoints the legacy bench loaded (the pair has
# no catalog yaml to transcribe — see module docstring). runtime echoes
# how the engine executes: torch, fp32 weights under a bfloat16 autocast
# forward (Config(dtype="bfloat16"), torch.amp.autocast in predict), no
# quantization, CPU.
_ND_RUNTIME: dict[str, Any] = {
    "engine": "torch",
    "dtype": "bfloat16",
    "quant": None,
    "device": "cpu",
}

NANODIFF_MODELS: list[dict[str, Any]] = [
    {
        "model_id": engine.repo_of(engine.CKPT),
        "version": "typed-decisions-lam1@step-3000",
        "runtime": dict(_ND_RUNTIME),
    },
    {
        "model_id": engine.repo_of(engine.CKPT_V2),
        "version": "typed-decisions-v2@step-3000",
        "runtime": dict(_ND_RUNTIME),
    },
]

NANODIFF_TRACKS = ["decision"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8032

# the generic instruction for bare choice questions (the suites' decision
# items carry options only); a known params.task upgrades it to the exact
# bench phrasing in engine.QUESTION.
DEFAULT_QUESTION = "Choose the best option."


def create_app() -> FastAPI:
    return make_runner_app(
        runner="nanodiff",
        models=NANODIFF_MODELS,
        predict=predict,
        tracks=NANODIFF_TRACKS,
        ui=None,
    )


# ------------------------------------------------- payload -> engine call
# The wire payload is the item's track-typed content (runner-protocol):
# {"input": {"text": ...} or str, "question": {...}, "params": {...} or
# None}. The suites' decision items carry question.options; the nanodiff
# tab's items additionally carried a task word whose fixed phrasing the
# legacy bench took from engine.QUESTION.


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


def _question_of(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, dict):
        raise ValueError("payload needs a 'question' object")
    return question


def _params_of(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params")
    return params if isinstance(params, dict) else {}


def _first_str(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _options(payload: dict[str, Any], question: dict[str, Any]) -> list[str]:
    """The ordered option texts (letters A..J map onto them by position).
    Accepts an options list or a {label: description} choices object on
    the question or the params."""
    params = _params_of(payload)
    raw = (
        question.get("options")
        or params.get("options")
        or question.get("choices")
        or params.get("choices")
    )
    if isinstance(raw, dict):
        options = [str(label) for label in raw]
    elif isinstance(raw, list):
        options = [str(option) for option in raw]
    else:
        options = []
    if not options:
        raise ValueError(
            "choice question needs a non-empty 'question.options' list (or a 'choices' object)"
        )
    return options


def _question_text(payload: dict[str, Any], question: dict[str, Any]) -> str:
    """The question string: explicit on the item, else the fixed phrasing
    the legacy bench used for a known task word (params.task ->
    engine.QUESTION), else the generic instruction (the suites' decision
    items carry options only, like the von island's default)."""
    text = _first_str(
        question.get("text"),
        question.get("instructions"),
        question.get("question"),
    )
    if text is not None:
        return text
    task = _params_of(payload).get("task")
    if isinstance(task, str) and task in engine.QUESTION:
        return engine.QUESTION[task]
    return DEFAULT_QUESTION


def _predict_choice(payload: dict[str, Any], question: dict[str, Any], ckpt: str) -> dict[str, Any]:
    state = _text_of(payload)
    options = _options(payload, question)
    text = _question_text(payload, question)
    pick, probs = engine.predict(engine.get_model(ckpt), state, text, options, engine.DEVICE)
    # prediction = the option text (the exact-match scorer compares against
    # the gold label, as the legacy bench stored res.choice); prob = the
    # answer-slot softmax mass for that option.
    return {"prediction": pick, "prob": probs[pick], "error": None}


def _entry_options(entry: dict[str, Any]) -> list[str]:
    """One multi sub-question's ordered option texts — strictly the
    entry's own options/choices (no fallback onto the parent payload:
    each slot's softmax is restricted to its own question's letters)."""
    raw = entry.get("options") or entry.get("choices")
    if isinstance(raw, dict):
        options = [str(label) for label in raw]
    elif isinstance(raw, list):
        options = [str(option) for option in raw]
    else:
        options = []
    if not options:
        raise ValueError(
            "each multi question needs a non-empty 'options' list (or a 'choices' object)"
        )
    return options


def _multi_questions(
    payload: dict[str, Any], question: dict[str, Any]
) -> list[tuple[str, list[str]]]:
    """multi question -> [(question, options), ...] for predict_multi
    (k questions over ONE state, one bidirectional forward)."""
    raw = question.get("questions") or _params_of(payload).get("questions")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "multi question needs a non-empty 'question.questions' list of {text, options} objects"
        )
    questions: list[tuple[str, list[str]]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"questions[{i}] must be a {{text, options}} object")
        text = _first_str(entry.get("text"), entry.get("question"), entry.get("instructions"))
        if text is None:
            raise ValueError(f"questions[{i}] needs a 'text'")
        questions.append((text, _entry_options(entry)))
    return questions


def _predict_multi(payload: dict[str, Any], question: dict[str, Any], ckpt: str) -> dict[str, Any]:
    state = _text_of(payload)
    questions = _multi_questions(payload, question)
    results = engine.predict_multi(engine.get_model(ckpt), state, questions, engine.DEVICE)
    # prediction = one pick per question, in order; prob stays None (the
    # wire's prob is a single float — the per-slot distributions are a
    # predict_multi return value, not a protocol field).
    return {
        "prediction": [pick for pick, _ in results],
        "prob": None,
        "error": None,
    }


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """runner-protocol predict(model_id, payload) -> {prediction, prob,
    error}.

    Dispatches on model_id (v1 lam1 vs v2 checkpoint) and question.type
    (choice -> one masked slot; multi -> the release's multi-decision
    form). Never raises: every failure becomes an {"error": "Type:
    message"} row so one bad item cannot kill a run, exactly like the
    legacy stdin runner's error JSON."""
    try:
        for ckpt in engine.CHECKPOINTS:
            if engine.repo_of(ckpt) == model_id:
                break
        else:
            raise ValueError(
                f"unknown model_id: {model_id!r} (this runner serves "
                f"{[engine.repo_of(c) for c in engine.CHECKPOINTS]})"
            )
        question = _question_of(payload)
        qtype = str(question.get("type") or "").strip().lower()
        if qtype == "choice":
            return _predict_choice(payload, question, ckpt)
        if qtype == "multi":
            return _predict_multi(payload, question, ckpt)
        raise ValueError(f"unsupported question type: {qtype!r} (nanodiff serves choice and multi)")
    except Exception as exc:  # noqa: BLE001 — per-item isolation by contract
        return {"prediction": None, "prob": None, "error": f"{type(exc).__name__}: {exc}"}


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the nanodiff 350M typed-decisions checkpoints on the runner protocol."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
