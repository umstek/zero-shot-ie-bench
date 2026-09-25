"""Reranker-as-decision-engine demo - runs in the MAIN venv.

Three cross-encoder rerankers (mixedbread-ai/mxbai-rerank-base-v2,
BAAI/bge-reranker-v2-m3, Alibaba-NLP/gte-reranker-modernbert-base) score
(instruction, label) pairs with sentence-transformers' CrossEncoder and the
argmax becomes the decision - a zero-shot classifier built out of a reranker,
no NER, no generation. Needs sentence-transformers==5.7.0 (see README).

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv/Scripts/python reranker_demo.py
"""

from __future__ import annotations

import sys
import time

# repo id -> short name, in bench order
RERANKERS = {
    "mixedbread-ai/mxbai-rerank-base-v2": "mxbai-rerank-base-v2",
    "BAAI/bge-reranker-v2-m3": "bge-reranker-v2-m3",
    "Alibaba-NLP/gte-reranker-modernbert-base": "GTE-rerank-ModernBERT-base",
}

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
INSTRUCTION = 'What is the overall {task} of this text: "{text}"'


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
    pairs = [(INSTRUCTION.format(task=task, text=text), label)
             for label in labels]
    scores = model.predict(pairs)
    return dict(zip(labels, (round(float(s), 4) for s in scores)))


def decide(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from sentence_transformers import CrossEncoder

    models = {}
    for repo, name in RERANKERS.items():
        print(f"Loading {repo} (first run downloads the checkpoint)...")
        t0 = time.perf_counter()
        models[name] = CrossEncoder(repo, device="cpu")
        print(f"  {name} ready in {time.perf_counter() - t0:.1f}s")

    # ---------------------------------------------------------------- 1
    banner("1. What a reranker does - raw pair scores, one text")
    text = "Oh great, my package finally arrived - only two weeks late."
    print(f'  text: "{text}"')
    scores = score_pairs(models["mxbai-rerank-base-v2"], text, "sentiment",
                         SENTIMENT_LABELS)
    print("  one (instruction, label) pair per label:")
    show(scores)
    print(f"  argmax -> {decide(scores)}")

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, all three checkpoints")
    for name, model in models.items():
        print(f"  {name}:")
        for text in SHARED_TEXTS:
            scores = timed(score_pairs, model, text, "sentiment",
                           SENTIMENT_LABELS)
            top = decide(scores)
            print(f"    {top:<8} {scores}  {text}")

    # ---------------------------------------------------------------- 3
    banner("3. Topic - one question per checkpoint")
    text = "The senate passed the budget bill after a late-night session."
    print(f'  text: "{text}"')
    for name, model in models.items():
        scores = timed(score_pairs, model, text, "topic", TOPIC_LABELS)
        print(f"  {name:<28} -> {decide(scores)}  {scores}")

    print("\nDone. Same texts through the decision engines:  "
          ".venv/Scripts/python certo_demo.py / "
          ".venv-von/Scripts/python mojev_demo.py\n")


if __name__ == "__main__":
    main()
