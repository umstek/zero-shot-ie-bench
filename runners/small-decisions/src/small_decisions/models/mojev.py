"""MoLeMo-Lab's MoJev (0.85B), loaded in-process from the HF hub.

Ported from legacy/engines/mojev_engine/ (revision-pinned upstream): the
scorer class is NOT vendored — it loads from the checkpoint via
trust_remote_code (the Qwen3.5 encoder is why this island pins transformers
5.17); the request path is adapted from the mojev GitHub runtime's serve.py
Engine.answer (option_texts/build_answer here, the packed-collate branch
vendored verbatim in models/_vendored/mojev_packer.py, MIT). One packed
forward answers k typed questions about one state: every question's
candidates are sorted (permutation invariance), packed with the state into
ONE sequence, and each field's softmax is restricted to its real
candidates.

Loads from the hub (MoLeMo-Lab/mojev @ 0c8695b6252f4205907433d4e196a94f032e60c3,
the catalog version id) — the HF cache satisfies later loads. Heavy imports
stay inside load(); the pure question->candidates/answer mappings here are
stdlib-only so offline tests exercise them without torch/transformers.

Catalog: molemo-lab/mojev, version
mojev@0c8695b6252f4205907433d4e196a94f032e60c3, deployment mojev.local
(runtime torch/cpu).
"""

from __future__ import annotations

import json
import threading
from typing import Any

from . import _systemone

MODEL_ID = "molemo-lab/mojev"
VERSION = "mojev@0c8695b6252f4205907433d4e196a94f032e60c3"  # catalog/models/molemo-lab/mojev.yaml
RUNTIME = {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"}
ENTRY: dict[str, Any] = {"model_id": MODEL_ID, "version": VERSION, "runtime": dict(RUNTIME)}

REPO = "MoLeMo-Lab/mojev"
REVISION = "0c8695b6252f4205907433d4e196a94f032e60c3"

_ENGINE: dict[str, Any] = {}
_LOCK = threading.Lock()


def _render(value: Any) -> str:
    """serve.py _render: strings pass through, everything else is JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def option_texts(name: str, question: dict[str, Any]) -> tuple[list[str], str]:
    """(candidate strings, kind) for one typed question, or ValueError with
    the problem named (serve.py option_texts, minus the 422 wrapper). A
    candidate's text is what the model scores, so a criterion's description
    is included when present — "angry: An upset message" scores better than
    the bare key. The prompt always says "kind: choice" (the serving path
    declares every field a choice); the question type only changes the
    candidate texts and the answer readout."""
    kind = question.get("type")
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"{name}: question type must be 'noul', 'choice' or 'score', not {kind!r}")
    criteria = question.get("criteria")
    if kind == "noul":
        # A two-option decision: candidates are ordered [no, yes].
        criteria = criteria or {}
        if not isinstance(criteria, dict):
            raise ValueError(
                f"{name}: noul criteria must be a dict with optional 'false'/'true' "
                f"description{'s'}, not {type(criteria).__name__}"
            )
        no = criteria.get("false") or "no"
        yes = criteria.get("true") or "yes"
        options = [_render(no), _render(yes)]
        if options[0] == options[1]:
            raise ValueError(f"{name}: noul 'false' and 'true' texts must differ")
        return options, kind
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"{name}: choice criteria must be a non-empty dict of label -> description-or-None")
        options = [key if value in (None, "") else f"{key}: {_render(value)}" for key, value in criteria.items()]
        if len(set(options)) != len(options):
            raise ValueError(f"{name}: choice candidates must be distinct after rendering")
        return options, kind
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{name}: score criteria must be a non-empty list of ordered level descriptions")
    return [_render(level) for level in criteria], kind


def build_answer(name: str, question: dict[str, Any], kind: str, probabilities: list[float]) -> dict[str, Any]:
    """serve.py build_answer: the typed answer for one field, probabilities
    in the caller's criteria order. noul -> the single P(yes) scalar
    (candidates are ordered [no, yes]); choice -> argmax label, confidence
    and the probability map; score -> the probability-weighted expected
    level (it can fall between integers) plus the legend mapping indexes
    back to the rubric text."""
    if kind == "noul":
        return {"type": "noul", "noul": float(probabilities[1])}
    if kind == "choice":
        keys = list(question["criteria"])
        mapped = {key: float(p) for key, p in zip(keys, probabilities)}
        best = max(mapped, key=mapped.__getitem__)
        return {"type": "choice", "choice": best, "confidence": mapped[best], "probabilities": mapped}
    levels = question["criteria"]
    mapped = {str(index): float(p) for index, p in enumerate(probabilities)}
    expected = sum(index * p for index, p in enumerate(probabilities))
    return {
        "type": "score",
        "score": float(expected),
        "confidence": float(max(probabilities)),
        "legend": {str(index): _render(level) for index, level in enumerate(levels)},
        "probabilities": mapped,
    }


def available() -> bool:
    """The checkpoint is in the local HF cache (a miss downloads at load
    time, which available() does not count as available)."""
    from . import _base

    return _base.hf_snapshot(REPO) is not None


def _load_typed_engine(device: str = "cpu") -> Any:
    """(score, answer_typed, tokenizer) sharing one loaded checkpoint.
    answer_typed(state, questions) -> (answers keyed by question name,
    usage), mirroring serve.py Engine.answer. Heavy imports stay inside."""
    import torch  # noqa: PLC0415 — the heavy stack, lazy

    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    from ._vendored import mojev_packer

    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    context_tokens = int(model.config.context_tokens)

    @torch.no_grad()
    def answer_typed(state: str, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """k typed questions (choice/score/noul) about one state in ONE
        packed forward. Every field is declared "choice" — the question
        type only changes the candidate texts and the readout — and the
        model holds no per-field parameters, so criteria it has never
        seen are simply text it encodes."""
        names, kinds, menus, orders, fields = [], [], [], [], []
        for name, question in questions.items():
            options, kind = option_texts(name, question)
            # Candidates are sorted before scoring and the answer is mapped
            # back; see mojev_packer.sort_candidates for why the invariance
            # depends on it.
            order = mojev_packer.sort_candidates(options)
            names.append(name)
            kinds.append(kind)
            menus.append([options[i] for i in order])
            orders.append(order)
            instructions = question.get("instructions")
            fields.append(
                (
                    mojev_packer.field_prompt(
                        name, _render(instructions) if instructions is not None else "", menus[-1]
                    ),
                    menus[-1],
                )
            )

        batch = mojev_packer.pack_fields(tokenizer, _render(state), fields, context_tokens)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)[0]  # (n_fields, max_cardinality)

        answers = {}
        for index, (name, kind, menu, order) in enumerate(zip(names, kinds, menus, orders, strict=False)):
            row = logits[index, : len(menu)]
            # Undo the sort so probabilities line up with the caller's criteria.
            probs = mojev_packer.unsort(order, row.softmax(-1).tolist())
            answers[name] = build_answer(name, questions[name], kind, probs)
        usage = {
            "input_tokens": int(batch["packed_mask"].sum()),
            "output_tokens": 0,  # the decision is the softmax; nothing is generated
        }
        return answers, usage

    return answer_typed


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}."""
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    with _LOCK:
        answer_typed = _ENGINE.get("answer_typed")
        if answer_typed is None:
            answer_typed = _load_typed_engine()
            _ENGINE["answer_typed"] = answer_typed
    answers, _usage = answer_typed(state, {"q": question})
    return _systemone.answer_to_result(question["type"], answers["q"])
