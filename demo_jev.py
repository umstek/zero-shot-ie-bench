"""Jev demo - TypeSafe AI's cloud classifier (System One / jev-latest).

Three question primitives, one request:
    choice(question, {label: description}) -> picks a label
    score(question, [rubric...])           -> 0..N-1 continuous score
    noul(question)                         -> yes/no probability

Everything is batched: N texts cost one request, not N.

Run:
    python demo_jev.py      # 2 live API requests, needs TYPESAFE_API_KEY
"""

from __future__ import annotations

import json
import sys
import time

from engines.jev_client import JevClient, choice, noul, score


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    client = JevClient()
    print(f"Jev client ready (model alias jev-latest, key loaded locally).")

    # ---------------------------------------------------------------- 1
    banner("1. The three primitives - excuses")
    excuses = [
        "my cat walked on the keyboard",
        "I was abducted by aliens",
        "I misread the calendar",
    ]
    result = client.ask(
        {"excuses": excuses},
        {
            "best": choice(
                "Which entry in `excuses` is the most believable?",
                {
                    "cat": "A pet disrupted the work - mundane and plausible",
                    "aliens": "Extraterrestrial interference",
                    "calendar": "A scheduling mistake",
                },
            ),
            "plausibility_cat": score(
                "How plausible is `excuses[0]` as a real reason?",
                ["Obviously fake", "Stretch, but happens", "Completely believable"],
            ),
            "funny": noul("Is `excuses[1]` intentionally humorous?"),
        },
    )
    answers = result["answers"]
    print(f"  most believable : {answers['best']['choice']} "
          f"(conf {answers['best']['confidence']:.2f})")
    print(f"  cat plausibility: {answers['plausibility_cat']['score']:.2f} / 2.00 "
          f"({answers['plausibility_cat']['legend']['1']})")
    print(f"  aliens funny?   : {answers['funny']['noul']:.2f}")
    print(f"  usage: {result['usage']['input_tokens']} in / "
          f"{result['usage']['output_tokens']} out, "
          f"latency {result['_latency_s']}s")

    # ---------------------------------------------------------------- 2
    banner("2. Batched classification - 6 texts, ONE request")
    texts = [
        "The food was cold and the waiter was rude.",
        "This is the best laptop I have ever owned.",
        "The meeting is scheduled for 3 PM in the main conference room.",
        "The flight was delayed for six hours with no explanation.",
        "She was thrilled with her exam results.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    labels = {
        "positive": "Text expresses a clearly positive attitude",
        "negative": "Text expresses a clearly negative attitude",
        "neutral": "Factual or mixed text without a clear attitude",
    }
    t0 = time.perf_counter()
    picks = client.classify(texts, labels, task="sentiment classification")
    dt = time.perf_counter() - t0
    for text, pick in zip(texts, picks):
        print(f"  {pick:<8} {text}")
    print(f"  one request for 6 texts, {dt:.2f}s total")

    # ---------------------------------------------------------------- 3
    banner("3. Raw answer shape (for the curious)")
    print(json.dumps(answers["best"], indent=2))

    print("\nDone. Benchmarks across all three systems:  python bench.py\n")


if __name__ == "__main__":
    main()
