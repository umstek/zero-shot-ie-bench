"""Typed-decision format for nanoDiff: exactly one masked token per decision.

Uniform interface, modelled on the TypeSafe 'choice' primitive -- a state plus a
supplied list of options, answered with one option letter:

    ### State:
    <unstructured state text>

    ### Question:
    <question>

    ### Options:
    A) yes
    B) no

    ### Answer:
     A

The multi-decision form asks k questions about ONE state and answers them on a
single line, so k typed decisions are read out of ONE bidirectional forward
pass:

    ### State:
    ...

    ### Questions:
    1) ... [A) yes, B) no]
    2) ... [A) 1, B) 2, C) 3, D) 4, E) 5]

    ### Answer:
    1) A 2) C

The primitive (noul / score / choice) is carried entirely by the option TEXT, so
one model serves all three. Because the answer is always a single capital letter,
the softmax at the answer position, restricted to the option letters, IS the
calibrated distribution over the supplied list.
"""
from __future__ import annotations

import tiktoken

EOT = 50256           # <|endoftext|> -- padding + answer end-marker
MASK = 50257          # nanoDiff absorbing state
LETTERS = "ABCDEFGHIJ"

PROMPT_LEN = 480
RESPONSE_LEN = 32
BLOCK_SIZE = PROMPT_LEN + RESPONSE_LEN      # 512 == the 350M base's block_size

_enc = None


def encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


def n_tokens(text):
    return len(encoder().encode(text))


def truncate_tokens(text, max_tokens):
    """Keep the head of `text`, at most `max_tokens` gpt2 tokens."""
    enc = encoder()
    ids = enc.encode(text)
    return text if len(ids) <= max_tokens else enc.decode(ids[:max_tokens])


def option_lines(options):
    return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))


def build_single_prompt(state, question, options):
    return (f"### State:\n{state}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Options:\n{option_lines(options)}\n\n"
            f"### Answer:")


def build_multi_prompt(state, questions):
    lines = []
    for i, (q, opts) in enumerate(questions):
        inline = ", ".join(f"{LETTERS[j]}) {o}" for j, o in enumerate(opts))
        lines.append(f"{i+1}) {q} [{inline}]")
    return (f"### State:\n{state}\n\n"
            f"### Questions:\n" + "\n".join(lines) + "\n\n### Answer:")


def build_single_response(answers):
    """One option index -> (response string, char offsets of the answer letter)."""
    return " " + LETTERS[answers[0]], [1]


def build_multi_response(answers):
    """k option indices -> (response string, char offsets of the k answers)."""
    s = " ".join(f"{i + 1}) {LETTERS[a]}" for i, a in enumerate(answers))
    offsets, pos = [], 0
    for i in range(len(answers)):
        if i:
            pos += 1                 # the space separator
        pos += len(f"{i + 1}) ")     # skip the "N) " marker
        offsets.append(pos)
        pos += 1                     # the letter itself
    return s, offsets


def decode_with_offsets(text):
    """Token ids of `text` plus the char span each token covers.

    NOTE: the span must be measured by decoding the token ids, NOT by asking for
    the encoding of a character prefix -- gpt2 BPE is not prefix-stable there, so
    `len(encode(text[:cp]))` is not the token index of char `cp`. For the multi
    response ('1) A 2) B' -> ['1', ')', ' A', ' 2', ')', ' B']) it silently
    returns the OLD token count and lands on the wrong token (' 2' instead of
    ' A'), which is what raised "answer token 362 is not one of the option
    letters".
    """
    enc = encoder()
    ids = enc.encode(text)
    spans, c = [], 0
    for t in ids:
        n = len(enc.decode([t]))
        spans.append((c, c + n))
        c += n
    assert c == len(text), (c, len(text), text)
    return ids, spans


def char_to_token(spans, pos):
    """Index of the token whose char span contains character offset `pos`."""
    for i, (a, b) in enumerate(spans):
        if a <= pos < b:
            return i
    raise ValueError(f"char offset {pos} is not inside any token (spans={spans})")


OPTION_TOKEN_IDS = None


def option_token_ids():
    """Token id of ' A'..' J'. Every answer letter is preceded by a space in both
    response forms, so the option set is a fixed, known set of token ids."""
    global OPTION_TOKEN_IDS
    if OPTION_TOKEN_IDS is None:
        enc = encoder()
        ids = {}
        for L in LETTERS:
            t = enc.encode(" " + L)
            if len(t) != 1:
                raise ValueError(f"option letter {L!r} is not a single token: {t}")
            ids[L] = t[0]
        OPTION_TOKEN_IDS = ids
    return OPTION_TOKEN_IDS


def encode_example(prompt_str, response_str, letter_offsets):
    """Encode one decision example into fixed-length (prompt, response) token ids.

    Mirrors nanodiff.sft.encode_sft_example: the prompt is RIGHT-aligned (truncated
    from the left, left-padded with EOT) so the '### Answer:' cue always lands at
    position PROMPT_LEN, and the response is LEFT-aligned at that position (EOT end
    marker, right-padded with EOT -- which is what teaches the model to stop).

    Returns:
        prompt_ids      (PROMPT_LEN,) ints
        response_ids    (RESPONSE_LEN,) ints
        answer_tokens   list of token indices within the response that carry the
                        decision(s) -- these are the positions we read probabilities at
    """
    enc = encoder()
    option_token_ids()                          # validates the option alphabet

    prompt = enc.encode(prompt_str)
    if len(prompt) > PROMPT_LEN:
        raise ValueError(f"prompt is {len(prompt)} tokens, PROMPT_LEN={PROMPT_LEN}")
    prompt = [EOT] * (PROMPT_LEN - len(prompt)) + prompt

    ids, spans = decode_with_offsets(response_str)
    answer_tokens = []
    for cp in letter_offsets:
        idx = char_to_token(spans, cp)
        if idx >= len(ids):
            raise ValueError(f"letter offset {cp} past end of response")
        tok = ids[idx]
        if enc.encode(enc.decode([tok])) != [tok]:
            raise ValueError(f"answer letter at char {cp} is not a single token")
        if tok not in set(option_token_ids().values()):
            raise ValueError(f"answer token {tok} is not one of the option letters")
        answer_tokens.append(idx)

    response = ids + [EOT]
    if len(response) > RESPONSE_LEN:
        raise ValueError(f"response is {len(response)} tokens, RESPONSE_LEN={RESPONSE_LEN}")
    response = response + [EOT] * (RESPONSE_LEN - len(response))
    return prompt, response, answer_tokens
