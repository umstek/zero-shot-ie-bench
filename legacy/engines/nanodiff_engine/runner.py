"""Checkpoint loading + single-choice scoring for nanodiff 350M.

Adapted from the pngwn/nanodiff-350m-typed-decisions release's
code/eval_calibration.py scoring path: the answer-letter position is the
only [MASK]ed token, one bidirectional forward, softmax restricted to the
option-letter token ids (' A'..' J', GPT-2 BPE), argmax = decision.
"""

from __future__ import annotations

import torch

from .nanodiff import Config, NanoDiff
from .decision_format import (MASK, PROMPT_LEN, RESPONSE_LEN,
                              build_multi_prompt, build_multi_response,
                              build_single_prompt, build_single_response,
                              encode_example, n_tokens, option_token_ids,
                              truncate_tokens)

CKPT = ("pngwn/nanodiff-350m-typed-decisions-lam1::"
        "nanodiff-350m-typed-decisions-lam1.pt")
# the v2 retrain of the same typed-decision recipe (step 3000; its embedded
# run args say lam=1.0, seed 1337 -- the same run the empty
# pngwn/nanodiff-350m-typed-decisions-v2-lam1 repo was meant to publish).
# Same architecture and state-dict layout as v1, loadable unchanged.
CKPT_V2 = ("pngwn/nanodiff-350m-typed-decisions-v2::"
           "nanodiff-350m-typed-decisions-v2.pt")
ARCH = dict(n_layer=16, n_head=20, n_embd=1280, block_size=512)
QUESTION = {"sentiment": "What is the overall sentiment of this text?",
            "topic": "Which topic category does this text belong to?"}
LETTERS = "ABCDEFGHIJ"


def load_model(device, ckpt=CKPT):
    from huggingface_hub import hf_hub_download
    repo, _, fn = ckpt.partition("::")
    ckpt = hf_hub_download(repo_id=repo, filename=fn)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = NanoDiff(Config(**ARCH, dtype="bfloat16"))
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), blob.get("iter_num")


@torch.no_grad()
def predict(model, state, question, options, device):
    if not 1 <= len(options) <= len(LETTERS):
        raise ValueError(f"a question needs 1..{len(LETTERS)} options "
                         "(single-token answer letters A-J)")
    # budget the state so the fully assembled prompt fits PROMPT_LEN;
    # encode_example hard-rejects longer prompts (the release's eval path
    # never sees them, a long user state in the demo/app would)
    budget = PROMPT_LEN - n_tokens(build_single_prompt("", question, options))
    state = truncate_tokens(state, max(0, budget))
    prompt_str = build_single_prompt(state, question, options)
    while n_tokens(prompt_str) > PROMPT_LEN and budget > 0:
        budget -= 8
        state = truncate_tokens(state, budget)
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


@torch.no_grad()
def predict_multi(model, state, questions, device):
    """k questions over ONE state in one bidirectional forward (the
    release's multi-decision form: one numbered answer line, k masked
    letter slots, each slot's softmax restricted to that question's
    option letters). questions: [(question, options), ...]; returns one
    (pick, {option: prob}) pair per question, in order."""
    if any(not 1 <= len(options) <= len(LETTERS)
           for _, options in questions):
        raise ValueError(f"each question needs 1..{len(LETTERS)} options "
                         "(single-token answer letters A-J)")
    budget = PROMPT_LEN - n_tokens(build_multi_prompt("", questions))
    state = truncate_tokens(state, max(0, budget))
    prompt_str = build_multi_prompt(state, questions)
    while n_tokens(prompt_str) > PROMPT_LEN and budget > 0:
        budget -= 8
        state = truncate_tokens(state, budget)
        prompt_str = build_multi_prompt(state, questions)
    response_str, letter_offsets = build_multi_response(
        [0] * len(questions))
    prompt_ids, response_ids, answer_tokens = encode_example(
        prompt_str, response_str, letter_offsets)
    letters = option_token_ids()

    p = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
    x = torch.tensor([response_ids], dtype=torch.int64, device=device)
    for j in answer_tokens:          # mask every answer slot, one per question
        x[0, j] = MASK
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(torch.cat([p, x], dim=1))
    results = []
    for q_idx, (_, options) in enumerate(questions):
        ids = [letters[LETTERS[i]] for i in range(len(options))]
        slot = logits[0, -RESPONSE_LEN + answer_tokens[q_idx],
                      ids].float()
        probs = torch.softmax(slot, dim=-1)
        idx = int(probs.argmax())
        results.append((options[idx],
                        {options[i]: round(float(p_), 4)
                         for i, p_ in enumerate(probs)}))
    return results
