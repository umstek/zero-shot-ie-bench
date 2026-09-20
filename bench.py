"""Benchmark: GLiNER 2.5 family vs GLiFormer vs Jev on this machine.

Systems
  - fastino/gliner2.5-small-v1  (74M, boundary, English)
  - fastino/gliner2.5-base-v1   (194M, boundary, English)
  - fastino/gliner2.5-multi-v1  (287M, boundary, multilingual)
  - knowledgator/gliformer-large-v1 (575.6M, layout-aware)
  - Jev (TypeSafe AI cloud, jev-latest) - classification only

Tasks
  - classification, sentiment (24 texts, 8 pos / 8 neg / 8 neutral)
  - classification, topic     (12 texts, 3 per topic)
  - NER, strict span+label F1 (10 texts) - local models only; Jev has no
    span output, it is a question-answering classifier, so it is excluded.

Fairness notes
  - Local models run one call per text (their native API) on CPU.
  - Jev runs ONE batched request per classification task (its native mode);
    per-text latency is reported as total/n for comparability.
  - All systems are zero-shot; identical label sets and label descriptions.
  - Defaults everywhere (thresholds, decoding) - out-of-the-box behaviour.

Output: bench_results.json (consumed by app.py's Benchmark tab).
"""

from __future__ import annotations

import json
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


def run_gliner25(model_id: str, cls_tasks, ner_texts):
    """One GLiNER 2.5 boundary checkpoint (small/base/multi share the API)."""
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        correct, latencies, misses = 0, [], []
        for text, truth in zip(texts, gold):
            t0 = time.perf_counter()
            pred = model.classify_text(text, {"task": list(labels)})["task"]
            latencies.append(time.perf_counter() - t0)
            if pred == truth:
                correct += 1
            else:
                misses.append((text, truth, pred))
        results[name] = dict(correct=correct, n=len(texts),
                             mean_latency_s=sum(latencies) / len(latencies),
                             misses=misses)

    predicted, ner_lat = set(), []
    for text, truth in ner_texts:
        t1 = time.perf_counter()
        out = model.extract_entities(text, NER_LABELS,
                                     include_spans=True, include_confidence=False)
        ner_lat.append(time.perf_counter() - t1)
        for label, items in out.get("entities", {}).items():
            for item in items:
                predicted.add((item["start"], item["end"], label))
    results["_ner_pred"] = predicted
    results["ner_mean_latency_s"] = sum(ner_lat) / len(ner_lat)
    return results


def run_gliformer(model_id: str, cls_tasks, ner_texts):
    from gliformer import GLiFormer

    model = GLiFormer.from_pretrained(model_id, load_tokenizer=True)
    model = model.to("cpu").eval()
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        correct, latencies, misses = 0, [], []
        for text, truth in zip(texts, gold):
            t0 = time.perf_counter()
            preds = model.classify(text, list(labels), threshold=0.5)
            latencies.append(time.perf_counter() - t0)
            pred = preds[0]["class_name"] if preds else None
            if pred == truth:
                correct += 1
            else:
                misses.append((text, truth, pred))
        results[name] = dict(correct=correct, n=len(texts),
                             mean_latency_s=sum(latencies) / len(latencies),
                             misses=misses)

    predicted, ner_lat = set(), []
    for text, truth in ner_texts:
        t1 = time.perf_counter()
        for ent in model.predict_entities(text, NER_LABELS, threshold=0.5):
            predicted.add((ent["start"], ent["end"], ent["label"]))
        ner_lat.append(time.perf_counter() - t1)
    results["_ner_pred"] = predicted
    results["ner_mean_latency_s"] = sum(ner_lat) / len(ner_lat)
    return results


LAYA_INSTRUCTIONS = {
    "sentiment": 'What is the overall sentiment of this text: "{text}"',
    "topic": 'Which topic category does this text belong to: "{text}"',
}


def run_laya(cls_tasks):
    """Laya (local, 421M): one forward pass per task with all texts as
    questions - its native batched mode, symmetric with Jev's batching.
    String instructions + bare labels: dict-shaped instructions (fine for
    Jev) collapse Laya onto one label - verified before benchmarking."""
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
        t0 = time.perf_counter()
        out = agent.predict({"task": name}, questions)
        total = time.perf_counter() - t0
        picks = [out["answers"][f"t{i}"].get("choice")
                 for i in range(len(texts))]
        correct, misses = 0, []
        for text, truth, pred in zip(texts, gold, picks):
            if pred == truth:
                correct += 1
            else:
                misses.append((text, truth, pred))
        results[name] = dict(
            correct=correct, n=len(texts),
            mean_latency_s=total / len(texts),
            batch_latency_s=round(total, 2), batched_requests=1,
            misses=misses)
    return results


def run_jev(cls_tasks):
    from jev_client import JevClient

    client = JevClient()
    results = {}
    for name, (texts, gold, labels) in cls_tasks.items():
        t0 = time.perf_counter()
        picks = client.classify(list(texts), dict(labels), task=name)
        total = time.perf_counter() - t0
        correct, misses = 0, []
        for text, truth, pred in zip(texts, gold, picks):
            if pred == truth:
                correct += 1
            else:
                misses.append((text, truth, pred))
        results[name] = dict(
            correct=correct, n=len(texts),
            mean_latency_s=total / len(texts),  # one batched request / n
            batch_latency_s=round(total, 2), batched_requests=1,
            misses=misses)
    return results


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
    ("GLiFormer-large", "knowledgator/gliformer-large-v1"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cls_tasks = {
        "sentiment": ([t for t, _ in SENTIMENT], [g for _, g in SENTIMENT],
                      SENTIMENT_LABELS),
        "topic": ([t for t, _ in TOPIC], [g for _, g in TOPIC], TOPIC_LABELS),
    }
    ner_texts = [(text, spans_of(text, truth)) for text, truth in NER]
    gold_ner = set().union(*[spans for _, spans in ner_texts])

    local_results = {}
    for name, model_id in LOCAL_SYSTEMS:
        print(f"Running {name} ({model_id}) ...")
        t0 = time.perf_counter()
        runner = run_gliformer if name == "GLiFormer-large" else run_gliner25
        local_results[name] = runner(model_id, cls_tasks, ner_texts)
        print(f"  done in {time.perf_counter() - t0:.0f}s")

    print("Running Laya (convaiinnovations/laya, local) ...")
    t0 = time.perf_counter()
    local_results["Laya (local)"] = run_laya(cls_tasks)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    print("Asking Jev (2 batched requests) ...")
    t0 = time.perf_counter()
    local_results["Jev"] = run_jev(cls_tasks)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    out = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "device": "CPU (no CUDA on this machine)",
            "systems": {name: mid for name, mid in LOCAL_SYSTEMS},
            "notes": [
                "Zero-shot, out-of-the-box defaults for every system.",
                "GLiNER2.5-small/base/multi share one architecture and API; "
                "multi is the 287M multilingual checkpoint run on English.",
                "Jev: one batched request per classification task; per-text "
                "latency = batch latency / n. Local models: one call per text.",
                "NER: strict span+label match; Jev and Laya excluded "
                "(no span output).",
                "Laya: base English checkpoint, one batched forward pass "
                "per task (same shape as Jev's batching); its card notes "
                "base checkpoints are classification-shaped, not general "
                "zero-shot decision engines, and probabilities ship "
                "over-confident before temperature fitting.",
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
        for name, res in local_results.items():
            entry = {
                "accuracy": round(res[task]["correct"] / res[task]["n"], 4),
                "mean_latency_s": round(res[task]["mean_latency_s"], 3),
                "misses": res[task]["misses"],
            }
            if "batch_latency_s" in res[task]:
                entry["batch_latency_s"] = res[task]["batch_latency_s"]
            out["classification"][task][name] = entry

    for name, res in local_results.items():
        if "_ner_pred" not in res:
            continue
        p, r, f1 = strict_prf(res.pop("_ner_pred"), gold_ner)
        out["ner"][name] = {"precision": p, "recall": r, "f1": f1,
                            "mean_latency_s": round(
                                res["ner_mean_latency_s"], 3)}

    with open("bench_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("\n=== Classification accuracy ===")
    for task in cls_tasks:
        row = out["classification"][task]
        print(f"  {task:<10} " + "  ".join(
            f"{n}: {row[n]['accuracy'] * 100:.1f}%"
            for n in local_results))
    print("\n=== NER strict F1 (Jev: n/a) ===")
    for name, m in out["ner"].items():
        print(f"  {name:<18} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"F1={m['f1']:.2f}")
    print("\nWrote bench_results.json")


if __name__ == "__main__":
    main()
