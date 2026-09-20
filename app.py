"""Interactive demo: GLiNER 2.5 vs GLiFormer vs Jev, plus a benchmark tab.

Run:
    python app.py            # loads both local checkpoints (~1 min), then
                             # serves http://127.0.0.1:7860
    python bench.py          # refreshes bench_results.json first if you want
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import statistics

SYSTEMS = {
    "GLiNER 2.5": "fastino/gliner2.5-base-v1",
    "GLiFormer": "knowledgator/gliformer-large-v1",
}
JEV = "Jev (cloud, jev-latest)"
SAMPLE_TEXT = (
    "Apple CEO Tim Cook announced the iPhone 15 in Cupertino yesterday. "
    "Alice works for Acme in Paris."
)
BENCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "bench_results.json")


def parse_labels(csv: str) -> list[str]:
    return [label.strip() for label in csv.split(",") if label.strip()]


def tile_highlights(text: str, spans: list[tuple[int, int, str]]):
    """Turn (start, end, label) spans into gr.HighlightedText segments."""
    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for start, end, label in sorted(spans, key=lambda s: s[0]):
        if start < cursor or start >= end or end > len(text):
            continue
        if start > cursor:
            segments.append((text[cursor:start], None))
        segments.append((text[start:end], label))
        cursor = end
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments


# --------------------------------------------------------------- Jev pieces
def get_jev_client():
    from jev_client import JevClient

    return JevClient()


# ------------------------------------------------- classification bench tab
def build_classification_tab():
    import gradio as gr

    path = os.path.join(os.path.dirname(BENCH_FILE),
                        "bench_classification_results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            bench = json.load(fh)
    except OSError:
        gr.Markdown("### Run `bench_graded.py` / `bench_newcomers.py` first.")
        return

    systems = bench["systems"]
    gr.Markdown("## Classification benchmark\n"
                f"One suite for **every classification-capable system** "
                f"({len(systems)}): extractors, decision engines, and "
                "purpose-built classifiers. Graded tiers: easy / medium / "
                "hard, 8 texts per tier per task.")
    for task in ("sentiment", "topic"):
        rows = []
        for system, entry in systems.items():
            tiers = entry[task]
            rows.append({"System": system,
                         **{tier: f"{tiers[tier]['accuracy'] * 100:.1f}%"
                            for tier in ("easy", "medium", "hard")},
                         "s/text": min(tiers[t]["mean_latency_s"]
                                       for t in tiers)})
        gr.DataFrame(pd.DataFrame(rows),
                     label=f"{task.capitalize()} accuracy (%) by tier")

    for task in ("sentiment", "topic"):
        long_rows = [{"System": system, "Tier": tier,
                      "Accuracy %": round(entry[task][tier]["accuracy"] * 100, 1)}
                     for system, entry in systems.items()
                     for tier in ("easy", "medium", "hard")]
        gr.BarPlot(pd.DataFrame(long_rows), x="System", y="Accuracy %",
                   color="Tier", title=f"{task.capitalize()} accuracy by "
                   "difficulty tier (%)", y_lim=(0, 105), height=300)
    speed_rows = [{"System": system,
                   "s per text": round(sum(
                       entry[t][tier]["mean_latency_s"]
                       for t in entry for tier in entry[t])
                       / sum(len(entry[t]) for t in entry), 3)}
                  for system, entry in systems.items()]
    gr.BarPlot(pd.DataFrame(speed_rows), x="System", y="s per text",
               color="System", title="Mean latency per text, averaged over "
               "tiers (CPU; Jev/Laya = batch latency / n)", height=300)

    gr.Markdown("#### Determinism\nEvery system above was measured 5x on "
                "the flat suite in `bench.py`: **100% output-stable** "
                "(identical predictions and NER span sets on every repeat; "
                "only latency jittered). Details: bench_results.json.")

    ml_path = os.path.join(os.path.dirname(BENCH_FILE),
                           "bench_multilingual_results.json")
    try:
        with open(ml_path, encoding="utf-8") as fh:
            ml = json.load(fh)
    except OSError:
        ml = None
    if ml:
        gr.Markdown("### Multilingual sub-suite — only multilingual-capable "
                    "systems\n9 languages, no English (popular: es/fr/zh · "
                    "medium: vi/tr/uk · rare: si/is/cy). GLiFormer-large is "
                    "an English-only **control**.")
        langs = list(next(iter(ml["by_language"].values())).keys())
        rows = [{"Language": lang,
                 **{sys_: f"{accs[lang] * 100:.0f}%"
                    for sys_, accs in ml["by_language"].items()}}
                for lang in langs]
        gr.DataFrame(pd.DataFrame(rows), label="Accuracy by language")
        long_rows = [{"Language": lang, "System": sys_,
                      "Accuracy %": round(accs[lang] * 100, 1)}
                     for sys_, accs in ml["by_language"].items()
                     for lang in langs]
        gr.BarPlot(pd.DataFrame(long_rows), x="Language", y="Accuracy %",
                   color="System", title="Multilingual accuracy by language (%)",
                   y_lim=(0, 105), height=300)
        rows = [{"System": sys_,
                 **{tier: f"{v * 100:.1f}%" for tier, v in tiers.items()}}
                for sys_, tiers in ml["by_tier"].items()]
        gr.DataFrame(pd.DataFrame(rows), label="Accuracy by popularity tier")


# ----------------------------------------------------- extraction bench tab
def build_extraction_tab():
    import gradio as gr

    path = os.path.join(os.path.dirname(BENCH_FILE),
                        "bench_extraction_results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            graded = json.load(fh)
    except OSError:
        graded = None

    gr.Markdown("## Extraction benchmark (NER)\n"
                "Span-producing systems only — that is the ability this "
                "benchmark measures. Decision engines (Laya, Jev, von, so1) "
                "and pure classifiers (GLiClass) have no span output and do "
                "not appear here; they compete in the Classification "
                "benchmark tab instead.")
    if graded:
        rows = []
        for system, tiers in graded["systems"].items():
            rows.append({"System": system,
                         **{tier: round(tiers[tier]["f1"], 2)
                            for tier in ("easy", "medium", "hard")}})
        gr.DataFrame(pd.DataFrame(rows),
                     label="Strict span+label F1 by tier (6 texts per tier)")
        long_rows = [{"System": system, "Tier": tier,
                      "Strict F1": round(tiers[tier]["f1"], 2)}
                     for system, tiers in graded["systems"].items()
                     for tier in ("easy", "medium", "hard")]
        gr.BarPlot(pd.DataFrame(long_rows), x="System", y="Strict F1",
                   color="Tier", title="NER strict F1 by difficulty tier",
                   y_lim=(0, 1.05), height=300)

    try:
        with open(BENCH_FILE, encoding="utf-8") as fh:
            flat = json.load(fh)
    except OSError:
        flat = None
    if flat:
        rows = [{"System": name,
                 "Precision": round(m["precision"], 2),
                 "Recall": round(m["recall"], 2),
                 "Strict F1": round(m["f1"], 2),
                 "Span stability %": round(m.get("stability", 0) * 100, 0),
                 "Mean s/text": m["mean_latency_s"]}
                for name, m in flat["ner"].items()]
        gr.DataFrame(pd.DataFrame(rows),
                     label="Flat suite (10 texts) — P/R/F1 + 5x determinism")
        gr.Markdown("#### Notes\n" + "\n".join(
            f"- {note}" for note in flat["meta"]["notes"][:4]))

# --------------------------------------------------------------- laya tab
def build_laya_tab():
    import gradio as gr

    with gr.Tab("Laya (local)"):
        gr.Markdown("### Laya — local open-source System-1 decision model "
                    "(421M, ModernBERT-large, RLCD)\n"
                    "Same choice/score/noul questions as Jev, but local, "
                    "free, Apache 2.0. Use **string instructions** — "
                    "dict-shaped instructions collapse it onto one label.")
        laya_text = gr.Textbox(
            label="Texts (one per line)",
            value="The food was cold and the waiter was rude.\n"
                  "This is the best laptop I have ever owned.\n"
                  "The meeting is scheduled for 3 PM.",
            lines=5)
        laya_labels = gr.Textbox(
            label="Labels (comma-separated)",
            value="positive, negative, neutral")
        laya_task = gr.Textbox(label="Task word for the instruction",
                               value="sentiment")
        laya_button = gr.Button("Ask Laya (one forward pass)",
                                variant="primary")
        laya_out = gr.JSON(label="Answers (label + probabilities per line)")

        def run_laya(texts_block, labels_csv, task_word):
            import laya

            from jev_client import choice

            texts = [line.strip() for line in texts_block.splitlines()
                     if line.strip()]
            bare = {label: None for label in parse_labels(labels_csv)}
            if not texts or not bare:
                return {"error": "provide text lines and labels"}
            questions = {
                f"t{i}": choice(
                    f'Which label applies to this text: "{text}" '
                    f'(task: {task_word})', bare)
                for i, text in enumerate(texts)
            }
            try:
                agent = get_laya_agent()
                payload = agent.predict({"task": task_word}, questions)
            except Exception as exc:
                return {"error": str(exc)}
            return {f'"{text[:40]}…"': {
                "label": payload["answers"][f"t{i}"].get("choice"),
                "probabilities": payload["answers"][f"t{i}"].get(
                    "probabilities"),
            } for i, text in enumerate(texts)}

        laya_button.click(run_laya,
                          [laya_text, laya_labels, laya_task], laya_out)


_LAYA_AGENT = None


def get_laya_agent():
    global _LAYA_AGENT
    if _LAYA_AGENT is None:
        import laya

        _LAYA_AGENT = laya.load("convaiinnovations/laya")
    return _LAYA_AGENT


def build_compare_tab():
    import gradio as gr

    gr.Markdown("""
## Feature comparison — all seven families

GLiNER 2.5 = small/base/multi checkpoints · GLiClass = edge/modern-base/
base/large — per-size scores live in the benchmark tabs.

| | GLiNER 2.5 | GLiFormer | GLiClass | Laya | von | so1 | Jev |
|---|---|---|---|---|---|---|---|
| **Ability group** | Extractor | Extractor | Classifier | Decision engine | Decision engine | Decision engine (BYO LLM) | Decision engine (cloud) |
| Zero-shot NER spans | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Text classification | ✅ | ✅ | ✅ | ✅ choice | ✅ choice | ✅ choice | ✅ choice |
| All labels scored in one pass | ✅ | ✅ | ✅ (its core design) | ✅ | ✅ | ✅ packed | ✅ one request |
| Relations | ✅ + JointIE graph | ✅ joint head | ❌ | ❌ | ❌ | ❌ | ❌ |
| Span attributes (per-entity sentiment) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Structured records | ✅ flat, anchor-based | ✅ nested Pydantic | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ordinal score rubrics | ❌ | ❌ | ❌ | ✅ score | ✅ rate | ✅ | ✅ score |
| Yes/no judgments | ❌ | ❌ | ❌ | ✅ noul | ✅ judge | ✅ yes_no | ✅ noul |
| Text embeddings | ❌ | ✅ 1024-d | ❌ (reranker-capable) | ❌ | ❌ | ❌ | ❌ |
| Multilingual | ✅ multi ckpt (100% on 6 langs here) | ❌ English | multilang-mini variant | ✅ Router, 100+ langs | not established | = base LLM's languages | ✅ 100% incl. Sinhala here |
| Runs offline / data local | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Cost | free | free | free | free | free | free | $0.042/1M input |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT (lib) | proprietary API |
| Notable | boundary architecture | layout-aware + embeddings | purpose-built classifier, 16 ms/text at edge size | RLCD calibration, script-detecting Router | TypeSafe /v1/systemone protocol-compatible | turns any ChatML LLM into a decision engine via logprobs | 255-choice cap, ECE 0.246 (3rd-party measured) |

Two benchmark groups follow from this table: everything classifies
(**Classification benchmark** tab); only the two extractor families produce
spans (**Extraction benchmark** tab).
""")

def build_gliner_tab(model):
    import gradio as gr

    with gr.Tab("GLiNER 2.5"):
        gr.Markdown("### GLiNER 2.5 base (194M, local, CPU) — "
                    "schema-driven extraction")
        with gr.Tab("Entities"):
            with gr.Row():
                ent_text = gr.Textbox(label="Text", value=SAMPLE_TEXT, lines=5)
                ent_labels = gr.Textbox(
                    label="Labels (comma-separated)",
                    value="company, person, product, location, organization",
                    lines=5)
            ent_button = gr.Button("Extract", variant="primary")
            ent_highlight = gr.HighlightedText(label="Highlights",
                                               show_legend=True)
            ent_json = gr.JSON(label="Raw output")

            def run_entities(text, labels_csv):
                labels = parse_labels(labels_csv)
                if not text or not labels:
                    return [], {"error": "provide text and at least one label"}
                result = model.extract_entities(
                    text, labels, include_confidence=True, include_spans=True)
                spans = [(item["start"], item["end"], label)
                         for label, items in result.get("entities", {}).items()
                         for item in items]
                return tile_highlights(text, spans), result.get("entities")

            ent_button.click(run_entities,
                             [ent_text, ent_labels], [ent_highlight, ent_json])

        with gr.Tab("Classification"):
            cls_text = gr.Textbox(
                label="Text",
                value="This laptop has amazing performance but terrible "
                      "battery life!", lines=3)
            cls_labels = gr.Textbox(
                label="Labels (comma-separated)",
                value="positive, negative, neutral")
            cls_multi = gr.Checkbox(label="Multi-label", value=False)
            cls_button = gr.Button("Classify", variant="primary")
            cls_out = gr.JSON()

            def run_classification(text, labels_csv, multi):
                labels = parse_labels(labels_csv)
                if not text or not labels:
                    return {"error": "provide text and at least one label"}
                spec = ({"labels": labels, "multi_label": multi,
                         "cls_threshold": 0.4} if multi else labels)
                return model.classify_text(text, {"task": spec})

            cls_button.click(run_classification,
                             [cls_text, cls_labels, cls_multi], cls_out)

        with gr.Tab("Relations"):
            rel_text = gr.Textbox(label="Text",
                                  value="Alice works for Acme in Paris.",
                                  lines=3)
            rel_labels = gr.Textbox(
                label="Relation labels (comma-separated)",
                value="works_for, located_in")
            rel_button = gr.Button("Extract relations", variant="primary")
            rel_out = gr.JSON()

            def run_relations(text, labels_csv):
                labels = parse_labels(labels_csv)
                if not text or not labels:
                    return {"error": "provide text and at least one label"}
                result = model.extract_relations(
                    text, labels, include_spans=True, include_confidence=True)
                return result.get("relation_extraction")

            rel_button.click(run_relations, [rel_text, rel_labels], rel_out)


def build_gliformer_tab(model):
    import gradio as gr

    with gr.Tab("GLiFormer"):
        gr.Markdown("### GLiFormer large (575.6M, local, CPU) — layout-aware "
                    "multi-task encoder with embeddings")
        with gr.Tab("Entities"):
            with gr.Row():
                gf_text = gr.Textbox(label="Text", value=SAMPLE_TEXT, lines=5)
                gf_labels = gr.Textbox(
                    label="Labels (comma-separated)",
                    value="company, person, product, location", lines=5)
            gf_button = gr.Button("Extract", variant="primary")
            gf_highlight = gr.HighlightedText(label="Highlights",
                                              show_legend=True)
            gf_json = gr.JSON(label="Raw output")

            def run_entities(text, labels_csv):
                labels = parse_labels(labels_csv)
                if not text or not labels:
                    return [], {"error": "provide text and labels"}
                ents = model.predict_entities(text, labels, threshold=0.5)
                spans = [(e["start"], e["end"], e["label"]) for e in ents]
                return tile_highlights(text, spans), ents

            gf_button.click(run_entities,
                            [gf_text, gf_labels], [gf_highlight, gf_json])

        with gr.Tab("Classification"):
            gfc_text = gr.Textbox(
                label="Text",
                value="The new search feature is fast and easy to use.",
                lines=3)
            gfc_labels = gr.Textbox(label="Labels (comma-separated)",
                                    value="positive, negative, neutral")
            gfc_button = gr.Button("Classify", variant="primary")
            gfc_out = gr.JSON()

            def run_classification(text, labels_csv):
                labels = parse_labels(labels_csv)
                if not text or not labels:
                    return {"error": "provide text and labels"}
                return model.classify(text, labels, threshold=0.5)

            gfc_button.click(run_classification,
                             [gfc_text, gfc_labels], gfc_out)


def build_jev_tab():
    import gradio as gr

    with gr.Tab("Jev (cloud)"):
        gr.Markdown("### Jev — TypeSafe AI System One (`jev-latest`)\n"
                    "One **batched** request classifies every line. Needs "
                    "`TYPESAFE_API_KEY` (env var or `.env` in the repo "
                    "root — see README). "
                    "Paid API — each click is a request.")
        jev_text = gr.Textbox(
            label="Texts (one per line)",
            value="The food was cold and the waiter was rude.\n"
                  "This is the best laptop I have ever owned.\n"
                  "The meeting is scheduled for 3 PM.",
            lines=5)
        jev_labels = gr.Textbox(
            label="Labels (comma-separated; optionally label: description)",
            value="positive, negative, neutral")
        jev_task = gr.Textbox(label="Task name (goes into the instructions)",
                              value="sentiment classification")
        jev_button = gr.Button("Ask Jev (one batched request)",
                               variant="primary")
        jev_table = gr.JSON(label="Answers (per line: label, confidence, "
                                  "probabilities) + usage")

        def run_jev(texts_block, labels_csv, task):
            from jev_client import choice

            texts = [line.strip() for line in texts_block.splitlines()
                     if line.strip()]
            criteria: dict[str, str | None] = {}
            for chunk in parse_labels(labels_csv):
                label, _, desc = chunk.partition(":")
                criteria[label.strip()] = desc.strip() or None
            if not texts or not criteria:
                return {"error": "provide text lines and labels"}
            try:
                client = get_jev_client()
                payload = client.ask(
                    {"task": task, "labels": criteria},
                    {f"t{i}": choice({"task": f"{task} of this text",
                                      "text": text}, criteria)
                     for i, text in enumerate(texts)})
            except Exception as exc:
                return {"error": str(exc)}
            answers = {f'"{text[:40]}…"': {
                "label": payload["answers"][f"t{i}"].get("choice"),
                "confidence": payload["answers"][f"t{i}"].get("confidence"),
                "probabilities": payload["answers"][f"t{i}"].get(
                    "probabilities"),
            } for i, text in enumerate(texts)}
            answers["_usage"] = payload.get("usage")
            answers["_latency_s"] = payload.get("_latency_s")
            return answers

        jev_button.click(run_jev, [jev_text, jev_labels, jev_task], jev_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-system demo + benchmark")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    import gradio as gr

    print("Loading local checkpoints (cached) ...")
    from gliner2 import AutoExtractor
    import torch
    from gliformer import GLiFormer

    gliner = AutoExtractor.from_pretrained(SYSTEMS["GLiNER 2.5"],
                                           map_location="cpu")
    print("  GLiNER 2.5 ready")
    gliformer = GLiFormer.from_pretrained(SYSTEMS["GLiFormer"],
                                          load_tokenizer=True).to("cpu").eval()
    print("  GLiFormer ready")

    with gr.Blocks(title="GLiNER 2.5 vs GLiFormer vs Jev") as app:
        gr.Markdown("# Zero-shot information extraction & classification\n"
                    "Two local encoders vs one cloud question-answering "
                    "classifier. Benchmark tab has measured numbers.")
        build_gliner_tab(gliner)
        build_gliformer_tab(gliformer)
        build_laya_tab()
        build_jev_tab()
        with gr.Tab("Classification benchmark"):
            build_classification_tab()
        with gr.Tab("Extraction benchmark"):
            build_extraction_tab()
        with gr.Tab("Compare"):
            build_compare_tab()

    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
