# See packer.py for the vendored-code attribution (mojev GitHub runtime,
# MIT). This file holds the benchmark-facing engine: checkpoint loading via
# trust_remote_code (the scorer class ships inside the HF repo) and the
# serve.py Engine.answer decision path -- option_texts/build_answer vendored
# from serve.py, score() for one choice question (the benchmark path) and
# answer_typed() for k typed questions about one state in one packed
# forward. Revision a74d58cd19ec573e83e8e27f9fecd837b8d830fb (2026-09-24).
"""MoJev engine: calibrated one-pass decision scoring over runtime options."""
from __future__ import annotations

import json

import torch
from transformers import AutoModel, AutoTokenizer

REPO = "MoLeMo-Lab/mojev"
REVISION = "0c8695b6252f4205907433d4e196a94f032e60c3"

from .packer import (field_prompt, pack_batch, pack_fields,  # noqa: E402
                     sort_candidates, unsort)


def _render(value) -> str:
    """serve.py _render: strings pass through, everything else is JSON."""
    return value if isinstance(value, str) else json.dumps(value,
                                                           ensure_ascii=False)


def option_texts(name: str, question: dict) -> tuple[list[str], str]:
    """(candidate strings, kind) for one typed question, or ValueError with
    the problem named (serve.py option_texts, minus the 422 wrapper). A
    candidate's text is what the model scores, so a criterion's description
    is included when present -- "angry: An upset message" scores better than
    the bare key. The prompt always says "kind: choice" (the serving path
    declares every field a choice); the question type only changes the
    candidate texts and the answer readout."""
    kind = question.get("type")
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"{name}: question type must be 'noul', 'choice' "
                         f"or 'score', not {kind!r}")
    criteria = question.get("criteria")
    if kind == "noul":
        # A two-option decision: candidates are ordered [no, yes].
        criteria = criteria or {}
        if not isinstance(criteria, dict):
            raise ValueError(f"{name}: noul criteria must be a dict with "
                             "optional 'false'/'true' descriptions, not "
                             f"{type(criteria).__name__}")
        no = criteria.get("false") or "no"
        yes = criteria.get("true") or "yes"
        return [_render(no), _render(yes)], kind
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"{name}: choice criteria must be a non-empty "
                             "dict of label -> description-or-None")
        return [key if value in (None, "") else f"{key}: {_render(value)}"
                for key, value in criteria.items()], kind
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{name}: score criteria must be a non-empty list "
                         "of ordered level descriptions")
    return [_render(level) for level in criteria], kind


def build_answer(name: str, question: dict, kind: str,
                 probabilities: list[float]) -> dict:
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
        return {"type": "choice", "choice": best,
                "confidence": mapped[best], "probabilities": mapped}
    levels = question["criteria"]
    mapped = {str(index): float(p) for index, p in enumerate(probabilities)}
    expected = sum(index * p for index, p in enumerate(probabilities))
    return {
        "type": "score",
        "score": float(expected),
        "confidence": float(max(probabilities)),
        "legend": {str(index): _render(level)
                   for index, level in enumerate(levels)},
        "probabilities": mapped,
    }


def load_typed_engine(device: str = "cpu"):
    """(score, answer_typed, tokenizer) sharing one loaded checkpoint.
    answer_typed(state, questions) -> (answers keyed by question name,
    usage), mirroring serve.py Engine.answer: every question's candidates
    are sorted, packed with the state and every other question into ONE
    sequence, one forward pass, and each field's softmax is restricted to
    its real candidates."""
    model = AutoModel.from_pretrained(REPO, revision=REVISION,
                                      trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    context_tokens = int(model.config.context_tokens)

    @torch.no_grad()
    def score(state: str, name: str, question: str, options: list[str]):
        order = sort_candidates(options)
        menu = tuple(options[i] for i in order)
        prompt = field_prompt(name, question, menu)
        batch = pack_batch(tokenizer, state, prompt, list(menu),
                           context_tokens)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)[0]
        probs_sorted = logits[0, :len(menu)].softmax(-1).tolist()
        probs = unsort(order, probs_sorted)
        return options[probs.index(max(probs))], probs

    @torch.no_grad()
    def answer_typed(state: str, questions: dict[str, dict]):
        """k typed questions (choice/score/noul) about one state in ONE
        packed forward. Every field is declared "choice" -- the question
        type only changes the candidate texts and the readout -- and the
        model holds no per-field parameters, so criteria it has never seen
        are simply text it encodes."""
        names, kinds, menus, orders, fields = [], [], [], [], []
        for name, question in questions.items():
            options, kind = option_texts(name, question)
            # Candidates are sorted before scoring and the answer is mapped
            # back; see sort_candidates for why the invariance depends on it.
            order = sort_candidates(options)
            names.append(name)
            kinds.append(kind)
            menus.append([options[i] for i in order])
            orders.append(order)
            instructions = question.get("instructions")
            fields.append((
                field_prompt(name,
                             _render(instructions)
                             if instructions is not None else "",
                             menus[-1]),
                menus[-1]))

        batch = pack_fields(tokenizer, _render(state), fields,
                            context_tokens)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)[0]           # (n_fields, max_cardinality)

        answers = {}
        for index, (name, kind, menu, order) in enumerate(
                zip(names, kinds, menus, orders)):
            row = logits[index, :len(menu)]
            # Undo the sort so probabilities line up with the caller's criteria.
            probs = unsort(order, row.softmax(-1).tolist())
            answers[name] = build_answer(name, questions[name], kind, probs)
        usage = {
            "input_tokens": int(batch["packed_mask"].sum()),
            "output_tokens": 0,    # the decision is the softmax; nothing is generated
        }
        return answers, usage

    return score, answer_typed, tokenizer


def load_engine(device: str = "cpu"):
    """(score function, tokenizer). score(state, name, question, options) ->
    (chosen option text, probabilities in caller order), mirroring
    serve.py Engine.answer for a single choice question (name is the schema
    field name read by Field.prompt). The benchmark path and both benches
    unpack exactly this pair; answer_typed rides along in
    load_typed_engine, which loads the same checkpoint once."""
    score, _, tokenizer = load_typed_engine(device)
    return score, tokenizer
