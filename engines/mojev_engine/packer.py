# Minimal text-only inference packing for MoLeMo-Lab/mojev, adapted from the
# mojev GitHub runtime (github.com/MoLeMo-Lab/mojev): the request path of
# serve.py Engine.answer (schema Field.prompt + candidate sorting), the
# single-field text-only branch of full.py packed_collate, and the softmax
# readout. Revision a74d58cd19ec573e83e8e27f9fecd837b8d830fb (2026-09-24).
# MIT licensed (package LICENSE; the model card repeats "Code is MIT
# licensed"). Checkpoint: huggingface.co/MoLeMo-Lab/mojev, revision
# 0c8695b6252f4205907433d4e196a94f032e60c3 (card license tag: mit; the Qwen
# base-model license applies to the encoder weights). The scorer class itself
# is NOT vendored: it loads from the checkpoint via trust_remote_code.
"""MoJev: packed one-pass typed-decision scoring (state | question | options
under a tree attention mask)."""
from __future__ import annotations

import torch

DEFAULT_FIELD_TOKENS_MAX = 32  # full.py packed_collate default


def field_prompt(name: str, description: str, options: tuple[str, ...]) -> str:
    """schema.Field.prompt for a single-choice field (schema.py)."""
    parts = [name.replace("_", " ")]
    if description:
        parts.append(description)
    parts.append("kind: choice")
    if len(options) <= 8:
        parts.append("options: " + ", ".join(options[:8]))
    return " | ".join(parts)


def sort_candidates(options: list[str]) -> list[int]:
    """full.py sort_candidates: scoring a sorted menu keeps the packed
    sequence a function of the candidate set (permutation invariance)."""
    return sorted(range(len(options)), key=lambda index: options[index])


def unsort(order: list[int], values: list) -> list:
    """full.py unsort: map per-candidate results back to caller order."""
    out = [None] * len(order)
    for position, original in enumerate(order):
        out[original] = values[position]
    return out


def pack_batch(tokenizer, context: str, prompt: str, options: list[str],
               context_tokens: int,
               field_tokens_max: int = DEFAULT_FIELD_TOKENS_MAX) -> dict:
    """One packed row [context][field prompt][option 0]...[option N-1] with
    span indicators, mirroring full.py packed_collate for a single choice
    field (text-only; no images, so no processor)."""
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad = tokenizer.pad_token_id
    ctx = tokenizer([context], truncation=True,
                    max_length=context_tokens)["input_ids"][0]
    question = tokenizer([prompt], truncation=True,
                         max_length=field_tokens_max)["input_ids"][0]
    pieces = tokenizer(list(options), truncation=False)["input_ids"]

    tokens = list(ctx) + list(question)
    opt_spans = []
    for piece in pieces:
        opt_spans.append((len(tokens), len(tokens) + len(piece)))
        tokens.extend(piece)
    total = len(tokens)

    packed = torch.full((1, total), pad, dtype=torch.long)
    packed[0, :len(tokens)] = torch.tensor(tokens)
    packed_mask = torch.zeros(1, total, dtype=torch.bool)
    packed_mask[0, :len(tokens)] = True
    context_span = torch.zeros(1, total)
    context_span[0, :len(ctx)] = 1.0
    field_span = torch.zeros(1, 1, total)
    field_span[0, 0, len(ctx):len(ctx) + len(question)] = 1.0
    option_span = torch.zeros(1, 1, len(options), total)
    option_mask = torch.zeros(1, 1, len(options), dtype=torch.bool)
    for i, (start, end) in enumerate(opt_spans):
        if end > start:
            option_span[0, 0, i, start:end] = 1.0
            option_mask[0, 0, i] = True
    return {"packed_ids": packed, "packed_mask": packed_mask,
            "context_span": context_span, "field_span": field_span,
            "option_span": option_span, "option_mask": option_mask}
