# See packer.py for the vendored-code attribution (mojev GitHub runtime,
# MIT). This file holds the benchmark-facing engine: checkpoint loading via
# trust_remote_code (the scorer class ships inside the HF repo) and the
# serve.py Engine.answer decision path reduced to one choice question.
"""MoJev engine: calibrated one-pass decision scoring over runtime options."""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

REPO = "MoLeMo-Lab/mojev"
REVISION = "0c8695b6252f4205907433d4e196a94f032e60c3"

from .packer import field_prompt, pack_batch, sort_candidates, unsort  # noqa: E402


def load_engine(device: str = "cpu"):
    """(score function, tokenizer). score(state, name, question, options) ->
    (chosen option text, probabilities in caller order), mirroring
    serve.py Engine.answer for a single choice question (name is the schema
    field name read by Field.prompt)."""
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(REPO)
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

    return score, tokenizer
