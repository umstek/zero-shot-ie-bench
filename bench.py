"""Benchmark: GLiNER 2.5 family vs GLiFormer (base+large) vs Laya vs Jev.

Every case runs --repeats times (default 5) to measure determinism:
how often the predicted output is identical across repeats, plus latency
mean/std. Accuracy is scored on the first repeat; stability is reported
separately.

Systems
  - fastino/gliner2.5-small-v1      (74M, boundary, English)
  - fastino/gliner2.5-base-v1       (194M, boundary, English)
  - fastino/gliner2.5-multi-v1      (287M, boundary, multilingual)
  - knowledgator/gliformer-base-v1  (~190M, layout-aware)
  - knowledgator/gliformer-large-v1 (575.6M, layout-aware)
  - Laya  (convaiinnovations/laya, local, English checkpoint)
  - Jev   (TypeSafe AI cloud, jev-latest) - classification only

Tasks
  - classification, sentiment (24 texts, 8 pos / 8 neg / 8 neutral)
  - classification, topic     (12 texts, 3 per topic)
  - NER, strict span+label F1 (10 texts) - local extractors only

Fairness notes
  - Local extractors run one call per text per repeat (their native API).
  - Laya and Jev run one batched call per task per repeat (their native
    mode); per-text latency = batch latency / n.
  - All systems are zero-shot; identical label sets and label descriptions.
  - Defaults everywhere (thresholds, decoding) - out-of-the-box behaviour.
  - Laya needs string instructions (dict instructions collapse it onto one
    label - verified before benchmarking).

Output: bench_results.json (consumed by app.py's Benchmark tab).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

SENTIMENT_LABELS = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual or mixed text without a clear attitude",
}

SENTIMENT = [
    ("I absolutely love this coffee shop, the staff are wonderful.", "positive"),
    ("The concert last night was fantastic from start to finish.", "positive"),
    ("This is the best laptop I have ever owned.", "positive"),
    ("She was thrilled with her exam results.", "positive"),
    ("The support team resolved my issue in minutes. Impressive.", "positive"),
    ("What a beautiful day for a hike.", "positive"),
    ("The new update makes the app a joy to use.", "positive"),
    ("Our vacation in Bali exceeded every expectation.", "positive"),
    ("The food was cold and the waiter was rude.", "negative"),
    ("This is the worst purchase I have ever made.", "negative"),
    ("The app crashes every single time I open it.", "negative"),
    ("I am deeply disappointed with the service quality.", "negative"),
    ("The flight was delayed for six hours with no explanation.", "negative"),
    ("My screen arrived cracked and support ignored my emails.", "negative"),
    ("The hotel room was filthy and smelled of smoke.", "negative"),
    ("Terrible value for the price, avoid at all costs.", "negative"),
    ("The meeting is scheduled for 3 PM in the main conference room.", "neutral"),
    ("Water boils at 100 degrees Celsius at sea level.", "neutral"),
    ("The report contains twelve chapters and three appendices.", "neutral"),
    ("He takes the bus to work every morning.", "neutral"),
    ("The library opens at nine on weekdays.", "neutral"),
    ("This model was released in 2024 by a research lab.", "neutral"),
    ("The invoice total comes to 342 dollars.", "neutral"),
    ("She was born in Lyon and studied economics.", "neutral"),
]

TOPIC_LABELS = {
    "technology": "Software, hardware, AI, gadgets, engineering",
    "business": "Companies, markets, revenue, deals, management",
    "sports": "Athletes, matches, teams, tournaments",
    "politics": "Government, elections, policy, legislation",
}

TOPIC = [
    ("The new GPU architecture doubles throughput per watt.", "technology"),
    ("Developers reported memory leaks in the latest framework release.", "technology"),
    ("The startup shipped an app that transcribes meetings locally on device.", "technology"),
    ("The company reported record quarterly revenue driven by overseas sales.", "business"),
    ("Shares fell three percent after the merger announcement.", "business"),
    ("The retailer will close fifty stores and cut operating costs.", "business"),
    ("The striker scored twice in the final minutes of the match.", "sports"),
    ("She won the marathon with a personal best of 2:19.", "sports"),
    ("The team traded their star pitcher for two prospects.", "sports"),
    ("The senate passed the budget bill after a late-night session.", "politics"),
    ("Voters head to the polls in the regional elections next month.", "politics"),
    ("The minister announced new tariffs on imported steel.", "politics"),
]

NER_LABELS = ["person", "company", "product", "location"]

NER = [
    ("Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday.",
     [("Apple", "company"), ("Tim Cook", "person"), ("iPhone 15", "product"),
      ("Cupertino", "location")]),
    ("Microsoft hired Jane Doe to lead Azure in London.",
     [("Microsoft", "company"), ("Jane Doe", "person"), ("Azure", "product"),
      ("London", "location")]),
    ("Satya Nadella spoke in Redmond about Microsoft.",
     [("Satya Nadella", "person"), ("Redmond", "location"),
      ("Microsoft", "company")]),
    ("Tesla launched the Model 3 in California last week.",
     [("Tesla", "company"), ("Model 3", "product"), ("California", "location")]),
    ("Angela Merkel met Emmanuel Macron in Berlin to discuss trade.",
     [("Angela Merkel", "person"), ("Emmanuel Macron", "person"),
      ("Berlin", "location")]),
    ("Netflix cancelled the series after two seasons, disappointing fans in Brazil.",
     [("Netflix", "company"), ("Brazil", "location")]),
    ("The Galaxy S24 outsold the Pixel 8 in Japan this quarter.",
     [("Galaxy S24", "product"), ("Pixel 8", "product"), ("Japan", "location")]),
    ("Novak Djokovic won the tournament in Melbourne on Sunday.",
     [("Novak Djokovic", "person"), ("Melbourne", "location")]),
    ("Boeing delivered the new 787 Dreamliner to United Airlines in Chicago.",
     [("Boeing", "company"), ("787 Dreamliner", "product"),
      ("United Airlines", "company"), ("Chicago", "location")]),
    ("Serena Williams invested in the startup after leaving Paris.",
     [("Serena Williams", "person"), ("Paris", "location")]),
]


def spans_of(text: str, truth: list[tuple[str, str]]) -> set[tuple[int, int, str]]:
    return {(text.index(span), text.index(span) + len(span), label)
            for span, label in truth}


def lat_stats(latencies: list[float]) -> dict:
    return {"mean_latency_s": round(statistics.mean(latencies), 3),
            "latency_std_s": round(statistics.pstdev(latencies), 3)}


def stability_of(runs: list[list]) -> float:
    """runs: one prediction list per repeat. Share of items whose
    prediction (by value) was identical across all repeats."""
    per_item = list(zip(*runs))
    stable = sum(1 for preds in per_item if len(set(map(repr, preds))) == 1)
    return round(stable / len(per_item), 4) if per_item else 1.0


def cls_summary(per_repeat, texts, gold):
    correct = sum(1 for pred, truth in zip(per_repeat[0], gold)
                  if pred == truth)
    misses = [(text, truth, pred) for text, truth, pred
              in zip(texts, gold, per_repeat[0]) if pred != truth]
    return correct, misses, stability_of(per_repeat)


def ner_repeat_loop(predict_once, ner_texts, repeats):
    """predict_once(text) -> frozenset of (start, end, label)."""
    predicted: set = set()
    latencies: list[float] = []
    stable_texts = 0
    for text, _ in ner_texts:
        run_sets = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_sets.append(predict_once(text))
            latencies.append(time.perf_counter() - t0)
        predicted |= set(run_sets[0])
        if len(set(run_sets)) == 1:
            stable_texts += 1
    stats = lat_stats(latencies)
    stats["stability"] = round(stable_texts / len(ner_texts), 4)
    return predicted, stats


def run_gliner25(model_id: str, cls_tasks, ner_texts, repeats: int):
    """One GLiNER 2.5 boundary checkpoint (small/base/multi share the API)."""
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        per_repeat, latencies = [], []
        for _ in range(repeats):
            run_preds = []
            for text in texts:
                t0 = time.perf_counter()
                run_preds.append(model.classify_text(
                    text, {"task": list(labels)})["task"])
                latencies.append(time.perf_counter() - t0)
            per_repeat.append(run_preds)
        correct, misses, stability = cls_summary(per_repeat, texts, gold)
        results[name] = dict(correct=correct, n=len(texts),
                             **lat_stats(latencies), stability=stability,
                             misses=misses)

    def predict_once(text):
        out = model.extract_entities(text, NER_LABELS, include_spans=True,
                                     include_confidence=False)
        return frozenset(
            (item["start"], item["end"], label)
            for label, items in out.get("entities", {}).items()
            for item in items)

    predicted, ner_stats = ner_repeat_loop(predict_once, ner_texts, repeats)
    results["_ner_pred"] = predicted
    results["ner"] = ner_stats
    return results


def run_gliformer(model_id: str, cls_tasks, ner_texts, repeats: int):
    from gliformer import GLiFormer

    model = GLiFormer.from_pretrained(model_id, load_tokenizer=True)
    model = model.to("cpu").eval()
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        per_repeat, latencies = [], []
        for _ in range(repeats):
            run_preds = []
            for text in texts:
                t0 = time.perf_counter()
                out = model.classify(text, list(labels), threshold=0.5)
                latencies.append(time.perf_counter() - t0)
                run_preds.append(out[0]["class_name"] if out else None)
            per_repeat.append(run_preds)
        correct, misses, stability = cls_summary(per_repeat, texts, gold)
        results[name] = dict(correct=correct, n=len(texts),
                             **lat_stats(latencies), stability=stability,
                             misses=misses)

    def predict_once(text):
        return frozenset(
            (e["start"], e["end"], e["label"])
            for e in model.predict_entities(text, NER_LABELS, threshold=0.5))

    predicted, ner_stats = ner_repeat_loop(predict_once, ner_texts, repeats)
    results["_ner_pred"] = predicted
    results["ner"] = ner_stats
    return results


LAYA_INSTRUCTIONS = {
    "sentiment": 'What is the overall sentiment of this text: "{text}"',
    "topic": 'Which topic category does this text belong to: "{text}"',
}


def run_laya(cls_tasks, repeats: int):
    """Laya (local, 421M): one batched forward pass per task per repeat -
    its native batched mode, symmetric with Jev's batching."""
    import laya

    from jev_client import choice

    agent = laya.load("convaiinnovations/laya")
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        bare = {label: None for label in labels}
        questions = {
            f"t{i}": choice(
                LAYA_INSTRUCTIONS.get(name, name + ' of this text: "{text}"')
                .format(text=text), bare)
            for i, text in enumerate(texts)
        }
        per_repeat, batch_lat = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = agent.predict({"task": name}, questions)
            batch_lat.append(time.perf_counter() - t0)
            per_repeat.append([out["answers"][f"t{i}"].get("choice")
                               for i in range(len(texts))])
        correct, misses, stability = cls_summary(per_repeat, texts, gold)
        results[name] = dict(
            correct=correct, n=len(texts),
            mean_latency_s=round(statistics.mean(batch_lat) / len(texts), 3),
            latency_std_s=round(
                statistics.pstdev([b / len(texts) for b in batch_lat]), 3),
            batch_latency_s=round(statistics.mean(batch_lat), 2),
            stability=stability, misses=misses)
    return results


def run_jev(cls_tasks, repeats: int):
    from jev_client import JevClient

    client = JevClient()
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        per_repeat, batch_lat = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            per_repeat.append(client.classify(list(texts), dict(labels),
                                              task=name))
            batch_lat.append(time.perf_counter() - t0)
        correct, misses, stability = cls_summary(per_repeat, texts, gold)
        results[name] = dict(
            correct=correct, n=len(texts),
            mean_latency_s=round(statistics.mean(batch_lat) / len(texts), 3),
            latency_std_s=round(
                statistics.pstdev([b / len(texts) for b in batch_lat]), 3),
            batch_latency_s=round(statistics.mean(batch_lat), 2),
            stability=stability, misses=misses)
    return results


# ------------------------------------------------------- ability result files
# Benchmarks are organized by ABILITY, not by when a system was added:
#   bench_classification_results.json - every system that can classify
#   bench_extraction_results.json     - systems that produce spans (NER)

CLASSIFICATION_FILE = "bench_classification_results.json"
EXTRACTION_FILE = "bench_extraction_results.json"

_CLASSIFICATION_META = {
    "suite": "graded sentiment + topic, 8 texts per tier per task",
    "tiers": {"easy": "one strong signal",
              "medium": "mixed signals, cross-domain vocabulary",
              "hard": "sarcasm, negation flips, lowercase brands, "
                      "context-dependent ambiguity"},
    "notes": [
        "All classification-capable systems run this same suite.",
        "Systems: extractors (GLiNER2.5, GLiFormer), decision engines "
        "(Laya, Jev, von, so1), purpose-built classifiers (GLiClass).",
        "von runs in .venv-von (needs transformers 5); so1 uses "
        "Qwen2.5-0.5B; single run per case (determinism established in "
        "bench_results.json: 100% stable over 5 repeats).",
    ],
}


def merge_classification(system: str, entry: dict) -> None:
    """entry: {"sentiment": {tier: {accuracy, mean_latency_s}}, ...}"""
    try:
        with open(CLASSIFICATION_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError:
        data = {"meta": _CLASSIFICATION_META, "systems": {}}
    data["systems"][system] = entry
    data["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M")
    with open(CLASSIFICATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def write_extraction(systems: dict) -> None:
    """systems: {name: {tier: {precision, recall, f1}}}"""
    with open(EXTRACTION_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {
            "suite": "graded NER, strict span+label match, 6 texts per tier",
            "notes": [
                "Only span-producing systems (extractors) run this suite.",
                "Decision engines and classifiers have no span output.",
            ]},
            "systems": systems}, fh, indent=2, ensure_ascii=False)


def strict_prf(predicted: set, gold: set):
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


LOCAL_SYSTEMS = [
    ("GLiNER2.5-small", "fastino/gliner2.5-small-v1"),
    ("GLiNER2.5-base", "fastino/gliner2.5-base-v1"),
    ("GLiNER2.5-multi", "fastino/gliner2.5-multi-v1"),
    ("GLiFormer-base", "knowledgator/gliformer-base-v1"),
    ("GLiFormer-large", "knowledgator/gliformer-large-v1"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="cross-system benchmark")
    parser.add_argument("--repeats", type=int, default=5,
                        help="runs per case for determinism (default 5)")
    args = parser.parse_args()

    cls_tasks = {
        "sentiment": ([t for t, _ in SENTIMENT], [g for _, g in SENTIMENT],
                      SENTIMENT_LABELS),
        "topic": ([t for t, _ in TOPIC], [g for _, g in TOPIC], TOPIC_LABELS),
    }
    ner_texts = [(text, spans_of(text, truth)) for text, truth in NER]
    gold_ner = set().union(*[spans for _, spans in ner_texts])

    all_results = {}
    for name, model_id in LOCAL_SYSTEMS:
        print(f"Running {name} ({model_id}), {args.repeats}x per case ...")
        t0 = time.perf_counter()
        runner = run_gliformer if name.startswith("GLiFormer") else run_gliner25
        all_results[name] = runner(model_id, cls_tasks, ner_texts, args.repeats)
        print(f"  done in {time.perf_counter() - t0:.0f}s")

    print(f"Running Laya, {args.repeats}x per task ...")
    t0 = time.perf_counter()
    all_results["Laya (local)"] = run_laya(cls_tasks, args.repeats)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    print(f"Asking Jev, {args.repeats}x per task "
          f"({2 * args.repeats} paid requests) ...")
    t0 = time.perf_counter()
    all_results["Jev"] = run_jev(cls_tasks, args.repeats)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    out = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "device": "CPU (no CUDA on this machine)",
            "repeats_per_case": args.repeats,
            "systems": {name: mid for name, mid in LOCAL_SYSTEMS},
            "notes": [
                "Zero-shot, out-of-the-box defaults for every system.",
                "GLiNER2.5 small/base/multi share one architecture; multi "
                "is the 287M multilingual checkpoint run on English.",
                f"Determinism: every case ran {args.repeats} times; "
                "accuracy is scored on the first repeat; stability = share "
                "of cases whose prediction (label, or full NER span set) "
                "was identical across all repeats; latency_std is the "
                "spread of per-call latencies (per-text for extractors, "
                "batch/n for Laya and Jev).",
                "Laya and Jev: one batched call per task per repeat. "
                "Local extractors: one call per text per repeat.",
                "NER: strict span+label match; Laya and Jev excluded "
                "(no span output).",
                "Laya: base English checkpoint; its card notes base "
                "checkpoints are classification-shaped and probabilities "
                "ship over-confident before temperature fitting. String "
                "instructions required (dict instructions collapse it).",
                "No gliner2.5-large exists; gliner2-large-v1 and "
                "gliner-community v2.5 checkpoints use the legacy span "
                "loader and were not run.",
            ],
        },
        "classification": {},
        "ner": {},
    }

    for task in cls_tasks:
        out["classification"][task] = {}
        for name, res in all_results.items():
            entry = {
                "accuracy": round(res[task]["correct"] / res[task]["n"], 4),
                "stability": res[task]["stability"],
                "mean_latency_s": res[task]["mean_latency_s"],
                "latency_std_s": res[task].get("latency_std_s", 0),
                "misses": res[task]["misses"],
            }
            if "batch_latency_s" in res[task]:
                entry["batch_latency_s"] = res[task]["batch_latency_s"]
            out["classification"][task][name] = entry

    for name, res in all_results.items():
        if "_ner_pred" not in res:
            continue
        p, r, f1 = strict_prf(res["_ner_pred"], gold_ner)
        out["ner"][name] = {"precision": p, "recall": r, "f1": f1,
                            **res["ner"]}

    with open("bench_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n=== Classification accuracy | stability over {args.repeats} "
          "runs ===")
    for task in cls_tasks:
        row = out["classification"][task]
        print(f"  {task:<10} " + "  ".join(
            f"{n}: {row[n]['accuracy'] * 100:.1f}%/"
            f"{row[n]['stability'] * 100:.0f}%" for n in all_results))
    print("\n=== NER strict F1 | span-set stability ===")
    for name, m in out["ner"].items():
        print(f"  {name:<18} F1={m['f1']:.2f}  stable={m['stability'] * 100:.0f}%"
              f"  lat={m['mean_latency_s']}±{m['latency_std_s']}s")
    print("\nWrote bench_results.json")


if __name__ == "__main__":
    main()
