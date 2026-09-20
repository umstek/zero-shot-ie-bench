"""Laya demo - open-source local answer to Jev (Convai Innovations).

Same typed-question interface as Jev (choice / score / noul over a state),
but a local non-autoregressive ModernBERT-large (421M) trained with RLCD for
calibrated probabilities. All questions in one call are answered in a single
forward pass. Apache 2.0, runs offline.

Run:
    python demo_laya.py       # loads English checkpoint (~808 MB once)
"""

from __future__ import annotations

import json
import sys
import time

from jev_client import choice, noul, score  # same question dict shape


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import laya

    print("Loading convaiinnovations/laya (English, 421M)...")
    t0 = time.perf_counter()
    agent = laya.load("convaiinnovations/laya")
    print(f"Loaded in {time.perf_counter() - t0:.0f}s")

    # ---------------------------------------------------------------- 1
    banner("1. The three primitives - excuses (same questions as demo_jev)")
    result = agent.predict(
        {"excuses": ["my cat walked on the keyboard",
                     "I was abducted by aliens"]},
        {
            "best": choice(
                "Which entry in `excuses` is the most believable?",
                {"cat": "A pet disrupted the work - mundane and plausible",
                 "aliens": "Extraterrestrial interference"},
            ),
            "plausibility_cat": score(
                "How plausible is `excuses[0]` as a real reason?",
                ["Obviously fake", "Stretch, but happens",
                 "Completely believable"],
            ),
            "funny": noul("Is `excuses[1]` intentionally humorous?"),
        },
    )
    answers = result["answers"]
    print(f"  most believable : {answers['best']['choice']} "
          f"(conf {answers['best']['confidence']:.2f})")
    print(f"  cat plausibility: {answers['plausibility_cat']['score']:.2f} / 2.00")
    print(f"  aliens funny?   : {answers['funny']['noul']:.2f}")
    print("  note: Jev answered cat/1.26/0.95 on the same questions - "
          "the base Laya checkpoint is\n  blunter on judgment calls "
          "(its card says so: strong on classification, weak zero-shot\n"
          "  on nuanced decisions; probabilities ship over-confident).")

    # ---------------------------------------------------------------- 2
    banner("2. Batched classification - 6 texts, ONE forward pass")
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
    questions = {
        f"t{i}": choice(
            f'What is the overall sentiment of this text: "{text}"',
            labels)
        for i, text in enumerate(texts)
    }
    t0 = time.perf_counter()
    out = agent.predict({"task": "sentiment classification"}, questions)
    dt = time.perf_counter() - t0
    for i, text in enumerate(texts):
        pick = out["answers"][f"t{i}"].get("choice")
        print(f"  {pick:<8} {text}")
    print(f"  one forward pass for 6 texts, {dt:.2f}s on CPU")

    # ---------------------------------------------------------------- 3
    banner("3. Raw answer shape (adds action/act_probability + usage)")
    print(json.dumps(answers["best"], indent=2))

    # ---------------------------------------------------------------- 4
    banner("4. Multilingual Router - script detection before the forward pass")
    print("  (downloads the mmBERT checkpoint once, ~647 MB)")
    try:
        from laya import Router

        router = Router()
        questions = {
            "department": choice(
                "Which department should handle this request?",
                {"billing": "invoices, payments, refunds",
                 "technical": "bugs, outages, system errors",
                 "other": "everything else"},
            ),
            "churn_risk": noul("Does the user threaten to cancel or leave?"),
        }
        for state in (
            {"body": "Hi, we were billed twice for March. Please refund "
                     "the duplicate today or we will cancel our plan."},
            {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
        ):
            res = router.predict(state, questions)
            a = res["answers"]
            print(f"  [{res['routing']['model']}] "
                  f"dept={a['department']['choice']} "
                  f"churn={a['churn_risk']['noul']:.2f} "
                  f"| {res['routing']['reason'][:60]}")
    except Exception as exc:
        print(f"  [skipped] router demo: {type(exc).__name__}: {exc}")

    print("\nDone. Full benchmark incl. Laya:  python bench.py\n")


if __name__ == "__main__":
    main()
