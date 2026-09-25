"""Checkpoint loading + single-choice scoring for nanodiff 350M.

Adapted from the pngwn/nanodiff-350m-typed-decisions release's
code/eval_calibration.py scoring path: the answer-letter position is the
only [MASK]ed token, one bidirectional forward, softmax restricted to the
option-letter token ids (' A'..' J', GPT-2 BPE), argmax = decision.
"""

from __future__ import annotations

import torch

from .nanodiff import Config, NanoDiff
from .decision_format import (MASK, RESPONSE_LEN, build_single_prompt,
                              build_single_response, encode_example,
                              option_token_ids)

CKPT = ("pngwn/nanodiff-350m-typed-decisions-lam1::"
        "nanodiff-350m-typed-decisions-lam1.pt")
ARCH = dict(n_layer=16, n_head=20, n_embd=1280, block_size=512)
QUESTION = {"sentiment": "What is the overall sentiment of this text?",
            "topic": "Which topic category does this text belong to?"}
LETTERS = "ABCDEFGHIJ"


def load_model(device):
    from huggingface_hub import hf_hub_download
    repo, _, fn = CKPT.partition("::")
    ckpt = hf_hub_download(repo_id=repo, filename=fn)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = NanoDiff(Config(**ARCH, dtype="bfloat16"))
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), blob.get("iter_num")


@torch.no_grad()
def predict(model, state, question, options, device):
    prompt_str = build_single_prompt(state, question, options)
    response_str, letter_offsets = build_single_response([0])  # " A"
    prompt_ids, response_ids, answer_tokens = encode_example(
        prompt_str, response_str, letter_offsets)
    letter_ids = [option_token_ids()[c] for c in LETTERS[:len(options)]]

    p = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
    x = torch.tensor([response_ids], dtype=torch.int64, device=device)
    for j in answer_tokens:                      # mask ONLY the answer slot
        x[0, j] = MASK
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(torch.cat([p, x], dim=1))
    slot = logits[0, -RESPONSE_LEN + answer_tokens[0],
                  letter_ids].float()
    probs = torch.softmax(slot, dim=-1)
    idx = int(probs.argmax())
    return options[idx], {options[i]: round(float(p_), 4)
                          for i, p_ in enumerate(probs)}
