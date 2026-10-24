"""Shared System One helpers for the island's decision engines.

decision2, d3, k2type, intern-decision and every Ollaya proxy answer the
TypeSafe System One wire format that legacy/engines/jev_client.py speaks
(choice / score / noul questions over a shared state). Everything they
share lives here, ported from the systemone-local island's
models/_systemone.py (runners never import each other):

- state_text / normalize_question: the runner decision-track payload
  (suites/decision items: input {"text": ...}, question {"type": "choice",
  "options": [...]}) -> the System One state text + question dict. A
  System One-native question dict passes through unchanged.
- systemone_result: an engine's System One answer row -> the runner result
  {"prediction", "prob"}, with prob = the probability the answer puts on
  its own prediction (what Brier/ECE need). In-place error rows
  ({"error": "invalid_question"} — decision2/d3 never raise) map to the
  runner's {"prediction": None, "error": ...} shape.

No heavy imports: this module is stdlib-only so the island imports (and
its tests run) without torch/transformers ever loading.
"""

from __future__ import annotations

from typing import Any

# A choice question whose options carry no question text gets this — one
# neutral default keeps all engines identically prompted.
DEFAULT_INSTRUCTIONS = "Which answer fits the context?"


def state_text(input_: Any) -> str:
    """The runner payload's `input` -> the state string the engines judge.

    suites/decision items carry {"text": ..., "images": [...]}; a bare
    string passes through. Images are out of scope for these engines on
    the decision track (the benches exercise the text path).
    """
    if isinstance(input_, str):
        return input_
    if isinstance(input_, dict):
        text = input_.get("text")
        if isinstance(text, str):
            return text
    raise ValueError(
        "decision items expect input as a string or {'text': str}, got "
        f"{type(input_).__name__}"
    )


def normalize_question(question: Any) -> dict[str, Any]:
    """Any accepted question shape -> the System One question dict.

    Accepts the suites' decision shape ({"type": "choice", "options":
    [label, ...]}) and the System One-native shape ({"type", "instructions",
    "criteria"}). Returns {"type", "instructions", "criteria"} with
    criteria a dict for choice/noul and an ordered list for score.
    """
    if not isinstance(question, dict):
        raise ValueError(f"question must be an object, got {type(question).__name__}")
    qtype = str(question.get("type", "choice"))
    if qtype == "bool":  # decider's alias; System One spells it noul
        qtype = "noul"
    instructions = question.get("instructions", question.get("question"))
    if instructions is None or (
        isinstance(instructions, str) and not instructions.strip()
    ):
        instructions = DEFAULT_INSTRUCTIONS

    if qtype == "choice":
        criteria = question.get("criteria", question.get("options"))
        if isinstance(criteria, dict):
            normalized = dict(criteria)
        elif isinstance(criteria, (list, tuple)):
            normalized = {str(option): None for option in criteria}
        else:
            raise ValueError(
                "choice question needs `options` (a list of labels) or "
                f"`criteria` (a {{label: description}} dict), got {criteria!r:.120}"
            )
        if not 2 <= len(normalized) <= 255:
            raise ValueError(f"choice needs 2..255 options, got {len(normalized)}")
        return {"type": "choice", "instructions": instructions, "criteria": normalized}

    if qtype == "score":
        criteria = question.get("criteria", question.get("options", question.get("levels")))
        if isinstance(criteria, dict):
            criteria = [criteria[key] for key in sorted(criteria, key=float)]
        if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"score needs an ordered list of 2..10 levels, got {criteria!r:.120}")
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    if qtype == "noul":
        criteria = question.get("criteria")
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"noul criteria must be a {{'true'/'false': description}} map, got {criteria!r:.120}")
        return {"type": "noul", "instructions": instructions, "criteria": criteria}

    raise ValueError(f"unknown question type: {qtype!r}")


def choice_row(labels: list[str], probs: list[float]) -> dict[str, Any]:
    """An unrounded System One choice answer row from per-label probabilities."""
    if len(probs) != len(labels) or not probs:
        raise ValueError(f"choice probabilities ({len(probs)}) do not match labels ({len(labels)})")
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "type": "choice",
        "choice": labels[best],
        "probabilities": {label: float(p) for label, p in zip(labels, probs, strict=False)},
    }


def answer_to_result(question_type: str, row: dict[str, Any]) -> dict[str, Any]:
    """A System One answer row -> the runner result {"prediction", "prob"}.

    prob is the probability the row puts on its own prediction: the chosen
    label's probability for choice, p(predicted side) for noul, and the
    modal level's probability for score (whose prediction is the expected
    level index — a graded value, so its prob is the modal confidence).
    """
    if row.get("type") == "choice" or "choice" in row:
        prediction = row["choice"]
        probs = row.get("probabilities") or {}
        prob = probs.get(prediction)
        if prob is None and probs:
            prob = max(probs.values())
        return {"prediction": prediction, "prob": None if prob is None else float(prob)}
    if row.get("type") == "noul" or "noul" in row:
        p_true = float(row["noul"])
        prediction = "true" if p_true >= 0.5 else "false"
        return {"prediction": prediction, "prob": p_true if prediction == "true" else 1.0 - p_true}
    if row.get("type") == "score" or "score" in row:
        probs = row.get("probabilities") or {}
        return {
            "prediction": float(row["score"]),
            "prob": max((float(p) for p in probs.values()), default=None),
        }
    raise ValueError(f"unrecognized answer row: {row!r:.200}")


def systemone_result(question_type: str, row: dict[str, Any]) -> dict[str, Any]:
    """answer_to_result, with decision2/d3's in-place error rows mapped to
    the runner's error shape (malformed or over-budget questions answer
    {"error": "invalid_question" | "max_length_exceeded"} instead of
    raising — the item errors, the run goes on)."""
    if "error" in row:
        return {"prediction": None, "prob": None, "error": f"engine answered error={row['error']!r}"}
    return answer_to_result(question_type, row)
