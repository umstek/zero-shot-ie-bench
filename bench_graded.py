"""Graded benchmark: easy / medium / hard tiers across subjects.

Classification: sentiment and topic, 8 texts per tier per subject.
NER: strict span+label match, 6 texts per tier (local extractors only).

Tier definitions
  easy   - one strong signal, standard vocabulary (what bench.py measures)
  medium - requires weighing mixed signals or cross-domain vocabulary,
           but one label clearly dominates
  hard   - sarcasm, negation flips, lowercase brands, context-dependent
           ambiguity (e.g. Apple the company vs apple the fruit); gold is
           still unambiguous to a careful human

Single run per case: run-to-run determinism was established in bench.py
(all systems 100% stable over 5 repeats).

Systems: GLiNER2.5 small/base/multi, GLiFormer base/large (classification
+ NER), Laya and Jev (classification only).

Output: bench_graded_results.json (app.py "Benchmarks v2" tab).
"""

from __future__ import annotations

import json
import sys
import time

from bench import (NER_LABELS, SENTIMENT_LABELS, TOPIC_LABELS, run_gliformer,
                   run_gliner25, run_jev, run_laya, spans_of, strict_prf)

TIERS = ("easy", "medium", "hard")

SENTIMENT: dict[str, list[tuple[str, str]]] = {
    "easy": [
        ("I absolutely love this playlist; every song is perfect.", "positive"),
        ("The hotel exceeded all our expectations.", "positive"),
        ("Worst customer service I have ever experienced.", "negative"),
        ("The delivery arrived broken and two weeks late.", "negative"),
        ("The shop opens at nine and closes at six.", "neutral"),
        ("The lecture covered three chapters of the textbook.", "neutral"),
        ("What a wonderful surprise, thank you so much!", "positive"),
        ("The room was dirty and the air conditioning was broken.", "negative"),
    ],
    "medium": [
        ("The camera is great, but the battery, screen, and support all disappoint.", "negative"),
        ("Service was slow and the place was noisy, though the dessert was good.", "negative"),
        ("The course was demanding, but I learned a lot and enjoyed it.", "positive"),
        ("Not the cheapest option, but easily the most reliable laptop I've owned.", "positive"),
        ("The film runs for three hours and won two awards.", "neutral"),
        ("The update is 400 MB and installs in about five minutes.", "neutral"),
        ("My order arrived on time, but two items were missing.", "negative"),
        ("The hotel is clean, quiet, and close to the station.", "positive"),
    ],
    "hard": [
        ("Oh great, another meeting that could have been an email.", "negative"),
        ("Wow, the app crashed exactly when I needed it. Perfect timing.", "negative"),
        ("I don't hate the new design; it's just worse in every way.", "negative"),
        ("Sure, ignore my emails for two weeks, that's fine, I love being ignored.", "negative"),
        ("Well, that presentation wasn't a total disaster - only most of it.", "negative"),
        ("I'm thrilled to pay a restocking fee for a product that never worked.", "negative"),
        ("It's not a bad book at all, honestly one of the better ones this year.", "positive"),
        ("The instructions were so clear that I assembled it wrong twice.", "negative"),
    ],
}

TOPIC: dict[str, list[tuple[str, str]]] = {
    "easy": [
        ("The new GPU architecture doubles throughput per watt.", "technology"),
        ("The striker scored twice in the final minutes of the match.", "sports"),
        ("The senate passed the budget bill after a late-night session.", "politics"),
        ("The company reported record quarterly revenue driven by overseas sales.", "business"),
        ("Developers reported memory leaks in the latest framework release.", "technology"),
        ("She won the marathon with a personal best of 2:19.", "sports"),
        ("Voters head to the polls in the regional elections next month.", "politics"),
        ("Shares fell three percent after the merger announcement.", "business"),
    ],
    "medium": [
        ("The football club's shares rose five percent after the transfer deal.", "business"),
        ("The midfielder's agent negotiated a record signing bonus for the striker.", "sports"),
        ("Lawmakers questioned the social media CEO about the data breach.", "politics"),
        ("The new graphics card sells out within minutes of every restock.", "technology"),
        ("The central bank raised interest rates before the housing report.", "business"),
        ("The startup's AI pitch won the national innovation award.", "technology"),
        ("The striker donated his match bonus to a local hospital.", "sports"),
        ("The senate debated new rules for autonomous vehicle testing.", "politics"),
    ],
    "hard": [
        ("The government launched a national AI strategy to boost chip manufacturing.", "politics"),
        ("The esports team's IPO was oversubscribed on its first trading day.", "business"),
        ("Biotech stocks fell after the regulator rejected the new therapy.", "business"),
        ("The Olympic committee partnered with a crypto exchange for the games.", "sports"),
        ("The prime minister's speech went viral on the new video app.", "politics"),
        ("The gaming console outsold every streaming device this holiday season.", "technology"),
        ("The chess world championship was streamed live to millions online.", "sports"),
        ("Venture funding for climate-tech startups doubled after the summit.", "business"),
    ],
}

NER: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {
    "easy": [
        ("Serena Williams won the match in Melbourne on Saturday.",
         [("Serena Williams", "person"), ("Melbourne", "location")]),
        ("Amazon opened a new office in Berlin last month.",
         [("Amazon", "company"), ("Berlin", "location")]),
        ("The iPhone 15 Pro costs 999 dollars in the United States.",
         [("iPhone 15 Pro", "product"), ("United States", "location")]),
        ("Elon Musk visited the factory in Texas with engineers from Tesla.",
         [("Elon Musk", "person"), ("Texas", "location"), ("Tesla", "company")]),
        ("Toyota recalled the Corolla in Canada after reports of brake faults.",
         [("Toyota", "company"), ("Corolla", "product"), ("Canada", "location")]),
        ("Angela Merkel spoke at the conference in Munich.",
         [("Angela Merkel", "person"), ("Munich", "location")]),
    ],
    "medium": [
        ("He transferred the money via revolut to pay for the spotify subscription.",
         [("revolut", "company"), ("spotify", "company")]),
        ("The galaxy note series competes directly with the iphone in most markets.",
         [("galaxy note series", "product"), ("iphone", "product")]),
        ("musk's spacex launched another batch of starlink satellites from florida.",
         [("musk", "person"), ("spacex", "company"), ("starlink", "product"),
          ("florida", "location")]),
        ("She ordered a big mac and a coke at the mcdonald's near times square.",
         [("big mac", "product"), ("coke", "product"), ("mcdonald's", "company"),
          ("times square", "location")]),
        ("The windows 11 update broke outlook for thousands of excel users at ford.",
         [("windows 11", "product"), ("outlook", "product"), ("excel", "product"),
          ("ford", "company")]),
        ("Zhang yiming founded bytedance in beijing before creating tiktok.",
         [("Zhang yiming", "person"), ("bytedance", "company"),
          ("beijing", "location"), ("tiktok", "product")]),
    ],
    "hard": [
        ("Tim Cook said Apple would open stores in India while eating an "
         "apple at a farm near Delhi.",
         [("Tim Cook", "person"), ("Apple", "company"), ("India", "location"),
          ("Delhi", "location")]),
        ("While visiting Amazon in Seattle, Jeff Bezos talked about Alexa, "
         "Kindle, and the rainforest tour he took near Manaus.",
         [("Amazon", "company"), ("Seattle", "location"), ("Jeff Bezos", "person"),
          ("Alexa", "product"), ("Kindle", "product"), ("Manaus", "location")]),
        ("The London office of Goldman Sachs hired Marie Curie's "
         "granddaughter, a physicist from Paris.",
         [("London", "location"), ("Goldman Sachs", "company"),
          ("Marie Curie", "person"), ("Paris", "location")]),
        ("Microsoft's Satya Nadella and Google's Sundar Pichai met at the "
         "White House to discuss AI.",
         [("Microsoft", "company"), ("Satya Nadella", "person"),
          ("Google", "company"), ("Sundar Pichai", "person"),
          ("White House", "location")]),
        ("After leaving DeepMind, the researcher joined OpenAI's London "
         "team to work on ChatGPT.",
         [("DeepMind", "company"), ("OpenAI", "company"),
          ("London", "location"), ("ChatGPT", "product")]),
        ("Nike signed a deal with fc barcelona to make jerseys for the "
         "2026 season.",
         [("Nike", "company"), ("fc barcelona", "company")]),
    ],
}

EXTRACTORS = [
    ("GLiNER2.5-small", "fastino/gliner2.5-small-v1"),
    ("GLiNER2.5-base", "fastino/gliner2.5-base-v1"),
    ("GLiNER2.5-multi", "fastino/gliner2.5-multi-v1"),
    ("GLiFormer-base", "knowledgator/gliformer-base-v1"),
    ("GLiFormer-large", "knowledgator/gliformer-large-v1"),
]


def cls_tasks(tier: str) -> dict:
    return {
        "sentiment": ([t for t, _ in SENTIMENT[tier]],
                      [g for _, g in SENTIMENT[tier]], SENTIMENT_LABELS),
        "topic": ([t for t, _ in TOPIC[tier]],
                  [g for _, g in TOPIC[tier]], TOPIC_LABELS),
    }


def ner_texts(tier: str) -> list:
    out = []
    for text, truth in NER[tier]:
        for span, _ in truth:
            if text.count(span) != 1:
                raise ValueError(f"gold span not unique in text: {span!r}")
        out.append((text, spans_of(text, truth)))
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M"),
                    "device": "CPU (no CUDA on this machine)",
                    "tiers": {"easy": "one strong signal",
                              "medium": "mixed signals, cross-domain vocabulary",
                              "hard": "sarcasm, negation flips, lowercase "
                                      "brands, context-dependent ambiguity"},
                    "notes": [
                        "Single run per case; determinism established in "
                        "bench.py (100% stable over 5 repeats).",
                        "Hard-tier gold labels are unambiguous to a careful "
                        "human but adversarial for lexicon-style models.",
                    ]},
          "classification": {"sentiment": {}, "topic": {}},
          "ner": {tier: {} for tier in TIERS}}

    for name, model_id in EXTRACTORS:
        print(f"{name} ...")
        t0 = time.perf_counter()
        runner = run_gliformer if name.startswith("GLiFormer") else run_gliner25
        for tier in TIERS:
            res = runner(model_id, cls_tasks(tier), ner_texts(tier), 1)
            for task in ("sentiment", "topic"):
                out["classification"][task].setdefault(tier, {})[name] = {
                    "accuracy": round(res[task]["correct"] / res[task]["n"], 4),
                    "mean_latency_s": res[task]["mean_latency_s"]}
            gold_tier = set().union(*[s for _, s in ner_texts(tier)])
            p, r, f1 = strict_prf(res.pop("_ner_pred"), gold_tier)
            out["ner"][tier][name] = {"precision": p, "recall": r, "f1": f1}
        print(f"  done in {time.perf_counter() - t0:.0f}s")

    for name, runner in (("Laya (local)", run_laya), ("Jev", run_jev)):
        print(f"{name} ...")
        t0 = time.perf_counter()
        for tier in TIERS:
            res = runner(cls_tasks(tier), 1)
            for task in ("sentiment", "topic"):
                out["classification"][task].setdefault(tier, {})[name] = {
                    "accuracy": round(res[task]["correct"] / res[task]["n"], 4),
                    "mean_latency_s": res[task]["mean_latency_s"]}
        print(f"  done in {time.perf_counter() - t0:.0f}s")

    with open("bench_graded_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("\n=== Accuracy by tier (sentiment / topic / NER-F1) ===")
    systems = [n for n, _ in EXTRACTORS] + ["Laya (local)", "Jev"]
    for tier in TIERS:
        print(f"  --- {tier} ---")
        for system in systems:
            s = out["classification"]["sentiment"][tier].get(system, {})
            t = out["classification"]["topic"][tier].get(system, {})
            n = out["ner"][tier].get(system, {})
            print(f"    {system:<18} "
                  f"{s.get('accuracy', 0) * 100:5.1f}%  "
                  f"{t.get('accuracy', 0) * 100:5.1f}%  "
                  f"{f'F1={n[chr(102) + chr(49)]:.2f}' if n else 'n/a':>8}")
    print("\nWrote bench_graded_results.json")


if __name__ == "__main__":
    main()
