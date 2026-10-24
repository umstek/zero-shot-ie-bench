"""Shared System One helpers for the island's four engines.

kev, decider and OpenThai answer the TypeSafe System One wire format that
legacy/engines/jev_client.py speaks (choice / score / noul questions over a
state); AgentJev speaks its own POST /api/evaluate rows, so its module
converts from this shape instead. Everything the four share lives here:

- state_text / normalize_question: the runner decision-track payload
  (suites/decision items: input {"text": ...}, question {"type": "choice",
  "options": [...]}) -> the System One state text + question dict. A
  System One-native question dict passes through unchanged.
- choice_row / noul_row / score_row: build an unrounded System One answer
  row from raw per-option probabilities, so every engine feeds the same
  answer extractor.
- answer_to_result: a System One answer row -> the runner result
  {"prediction", "prob"}, with prob = the probability the answer puts on
  its own prediction (what Brier/ECE need).

No heavy imports: this module is stdlib-only so the island imports (and
its tests run) without torch/transformers ever loading.
"""

from __future__ import annotations

from typing import Any

# A choice question whose options carry no question text gets this — the
# same fixed sentence decider/systemone.py and openthai fall back to for
# a noul without instructions ("the question id is never shown to the
# model", so one neutral default keeps all engines identically prompted).
DEFAULT_INSTRUCTIONS = "Which answer fits the context?"

# The AgentJev contract's question kinds (jev_service.contract): boolean,
# choice, score. "noul" is the System One spelling of "boolean".
AGENTJEV_TYPES = ("boolean", "choice", "score")


def state_text(input_: Any) -> str:
    """The runner payload's `input` -> the state string the engines judge.

    suites/decision items carry {"text": ..., "images": [...]}; a bare
    string passes through. Images are out of scope for these engines
    (none claims the image modality on the decision track).
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
    criteria a dict for choice/noul and an ordered list for score — exactly
    what kev.api.SystemOneRequest, decider and openthai parse_question take.
    """
    if not isinstance(question, dict):
        raise ValueError(f"question must be an object, got {type(question).__name__}")
    qtype = str(question.get("type", "choice"))
    if qtype == "bool":  # decider's alias; System One spells it noul
        qtype = "noul"
    # "question" is AgentJev's spelling of "instructions" (both engines
    # accept either key; see decider.systemone.render_question).
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
            # 10 = the AgentJev contract's ceiling; the System One engines
            # allow up to 255 but the suites never send more than 10.
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


def noul_row(p_true: float) -> dict[str, Any]:
    """A System One noul answer row (row['noul'] = p(true)); false rides first."""
    return {"type": "noul", "noul": float(p_true)}


def score_row(probs: list[float]) -> dict[str, Any]:
    """A System One score answer row: score = the expected level index."""
    score = sum(index * p for index, p in enumerate(probs))
    return {
        "type": "score",
        "score": score,
        "probabilities": {str(index): float(p) for index, p in enumerate(probs)},
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


def agentjev_row(question: dict[str, Any], key: str = "q") -> dict[str, Any]:
    """A System One question dict -> one AgentJev /api/evaluate row.

    Follows the parts of jev_service.contract.prepare() that would reject a
    row: choice needs 2..255 options with nonempty descriptions (a None
    description becomes the bare label — semantic() rejects None); score
    needs 2..10 distinct level descriptions; boolean criteria drop
    undescribed sides (the server defaults them to TRUE/FALSE).
    """
    qtype = question["type"]
    instructions = question["instructions"]
    row: dict[str, Any] = {"id": key, "type": AGENTJEV_TYPES[1], "question": instructions}
    if qtype == "choice":
        options = {
            label: (label if description in (None, "") else description)
            for label, description in question["criteria"].items()
        }
        row["options"] = options
    elif qtype == "score":
        row["type"] = AGENTJEV_TYPES[2]
        row["levels"] = [str(level) for level in question["criteria"]]
    elif qtype == "noul":
        row["type"] = AGENTJEV_TYPES[0]
        criteria = question["criteria"] or {}
        row["criteria"] = {
            side: criteria[side]
            for side in ("true", "false")
            if criteria.get(side) not in (None, "")
        }
    else:  # normalize_question only produces the three kinds above
        raise ValueError(f"unknown question type: {qtype!r}")
    return row


def agentjev_answer_row(row: dict[str, Any]) -> dict[str, Any]:
    """An AgentJev answer row -> the equivalent System One answer row.

    boolean rows carry {"probability" (= p true), "value"}; choice rows
    {"value", "distribution"}; score rows {"score", "distribution"} — all
    three map onto the shapes answer_to_result extracts.
    """
    kind = row.get("type")
    if kind == "boolean":
        return noul_row(row["probability"])
    if kind == "choice":
        return choice_row(list(row["distribution"]), [row["distribution"][k] for k in row["distribution"]])
    if kind == "score":
        return score_row([row["distribution"][k] for k in sorted(row["distribution"], key=float)])
    raise ValueError(f"unrecognized AgentJev answer row: {row!r:.200}")
