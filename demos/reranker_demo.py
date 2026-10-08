"""Reranker-as-decision-engine demo - runs in the MAIN venv.

Five local cross-encoder rerankers (mixedbread-ai/mxbai-rerank-base-v2 and
its 1.54B large sibling, BAAI/bge-reranker-v2-m3, Alibaba-NLP/
gte-reranker-modernbert-base, nvidia/llama-nemotron-rerank-1b-v2) score
(instruction, label) pairs and the argmax becomes the decision - a
zero-shot classifier built out of a reranker, no NER, no generation. The
four sentence-transformers checkpoints ride CrossEncoder with the house
(instruction, bare-label) shape; nemotron's remote-code head needs its
card-verbatim "question:{text} \n \n passage:{description}" template
(bare labels collapse it - see the NEMOTRON_RERANKERS notes in
bench_spectrum.py). Needs sentence-transformers==5.7.0 (see README).

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv/Scripts/python demos/reranker_demo.py
"""

from __future__ import annotations

import sys
import time

# repo id -> short name, in bench order
RERANKERS = {
    "mixedbread-ai/mxbai-rerank-base-v2": "mxbai-rerank-base-v2",
    "mixedbread-ai/mxbai-rerank-large-v2": "mxbai-rerank-large-v2",
    "BAAI/bge-reranker-v2-m3": "bge-reranker-v2-m3",
    "Alibaba-NLP/gte-reranker-modernbert-base": "GTE-rerank-ModernBERT-base",
}
NEMOTRON_REPO = "nvidia/llama-nemotron-rerank-1b-v2"

SHARED_TEXTS = [
    "The food was cold and the waiter was rude.",
    "This is the best laptop I have ever owned.",
    "The meeting is scheduled for 3 PM in the main conference room.",
    "The flight was delayed for six hours with no explanation.",
    "She was thrilled with her exam results.",
    "Water boils at 100 degrees Celsius at sea level.",
]
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
TOPIC_LABELS = ["technology", "business", "sports", "politics"]
# same wording as bench_spectrum.py classify_reranker so demo and bench
# scores compare directly; free-form task words get the generic template
INSTRUCTIONS = {
    "sentiment": 'What is the overall sentiment of this text: "{text}"',
    "topic": 'Which topic category does this text belong to: "{text}"',
}
GENERIC_INSTRUCTION = 'What is the overall {task} of this text: "{text}"'
# the house label descriptions, nemotron's measured passage texts (the
# same mapping the benches run)
NEMOTRON_DESCRIPTIONS = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual or mixed text without a clear attitude",
    "technology": "Software, hardware, AI, gadgets, engineering",
    "business": "Companies, markets, revenue, deals, management",
    "sports": "Athletes, matches, teams, tournaments",
    "politics": "Government, elections, policy, legislation",
}


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    import json

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def score_pairs(model, text: str, task: str, labels: list[str]):
    """One (instruction, label) pair per label; returns label -> score."""
    instr = INSTRUCTIONS.get(task, GENERIC_INSTRUCTION)
    pairs = [(instr.format(task=task, text=text), label)
             for label in labels]
    scores = model.predict(pairs)
    return dict(zip(labels, (round(float(s), 4) for s in scores)))


def load_nemotron_scorer(repo: str):
    """Card-verbatim load (trust_remote_code remote-code head, left
    padding, pad token falling back to eos); returns a scorer with the
    same (text, task, labels) surface as score_pairs, one batched forward
    over the card's "question:{text} \n \n passage:{description}" prompts.
    fp32 on CPU (the card's bf16 example targets GPU)."""
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, trust_remote_code=True).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    def score(text: str, task: str, labels: list[str]) -> dict[str, float]:
        prompts = [f"question:{text} \n \n "
                   f"passage:{NEMOTRON_DESCRIPTIONS.get(label, label)}"
                   for label in labels]
        batch = tokenizer(prompts, padding=True, truncation=True,
                          return_tensors="pt", max_length=512)
        with torch.inference_mode():
            logits = model(**batch).logits
        return dict(zip(labels, (round(float(s), 4)
                                 for s in logits.view(-1).tolist())))

    return score


def decide(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from sentence_transformers import CrossEncoder

    scorers = {}  # short name -> (text, task, labels) -> {label: score}
    for repo, name in RERANKERS.items():
        print(f"Loading {repo} (first run downloads the checkpoint)...")
        t0 = time.perf_counter()
        model = CrossEncoder(repo, device="cpu")
        print(f"  {name} ready in {time.perf_counter() - t0:.1f}s")
        scorers[name] = (lambda m: lambda *a, **k: score_pairs(m, *a, **k))(
            model)
    print(f"Loading {NEMOTRON_REPO} (remote-code bidirectional head)...")
    t0 = time.perf_counter()
    scorers["nemotron-rerank-1b-v2"] = load_nemotron_scorer(NEMOTRON_REPO)
    print(f"  nemotron-rerank-1b-v2 ready in "
          f"{time.perf_counter() - t0:.1f}s")

    # ---------------------------------------------------------------- 1
    banner("1. What a reranker does - raw pair scores, one text")
    text = "Oh great, my package finally arrived - only two weeks late."
    print(f'  text: "{text}"')
    scores = scorers["mxbai-rerank-base-v2"](text, "sentiment",
                                             SENTIMENT_LABELS)
    print("  one (instruction, label) pair per label:")
    show(scores)
    print(f"  argmax -> {decide(scores)}")

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, all five checkpoints")
    for name, scorer in scorers.items():
        print(f"  {name}:")
        for text in SHARED_TEXTS:
            scores = timed(scorer, text, "sentiment", SENTIMENT_LABELS)
            top = decide(scores)
            print(f"    {top:<8} {scores}  {text}")

    # ---------------------------------------------------------------- 3
    banner("3. Topic - one question per checkpoint")
    text = "The senate passed the budget bill after a late-night session."
    print(f'  text: "{text}"')
    for name, scorer in scorers.items():
        scores = timed(scorer, text, "topic", TOPIC_LABELS)
        print(f"  {name:<28} -> {decide(scores)}  {scores}")

    print("\nDone. Same texts through the decision engines:  "
          ".venv/Scripts/python demos/certo_demo.py / "
          ".venv-von/Scripts/python demos/mojev_demo.py\n")


if __name__ == "__main__":
    main()
