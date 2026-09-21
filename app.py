"""Interactive demo + benchmarks for seven zero-shot IE/classification
families: GLiNER 2.5, GLiFormer, GLiClass, Laya, von, so1 and Jev.

Run:
    python app.py            # loads the GLiNER 2.5 + GLiFormer checkpoints
                             # (~1 min), then serves http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import math
import os

import altair as alt
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


# --------------------------------------------------------------- charts
def hbar_chart(df: pd.DataFrame, value: str, title: str,
               value_title: str, descending: bool = True):
    """Horizontal bars, one row per System, sorted by value.

    Gradio's built-in BarPlot stacks series whenever `color` is set and
    offers no grouping/horizontal options, so benchmark charts are
    altair charts rendered through gr.Plot instead.
    """
    order = df.sort_values(value, ascending=not descending)["System"].tolist()
    return (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            y=alt.Y("System:N", sort=order, title=None,
                    axis=alt.Axis(labelFontSize=12)),
            x=alt.X(f"{value}:Q", title=value_title, scale=alt.Scale(
                domain=[0, df[value].max() * 1.12])),
            tooltip=[alt.Tooltip("System:N"), alt.Tooltip(f"{value}:Q",
                     format=".3f")],
        )
        .properties(width=640, height=max(180, 26 * len(order) + 50))
    )


def hbar_chart_labeled(df: pd.DataFrame, value: str, title: str,
                       value_title: str):
    """hbar_chart plus the numeric value printed at each bar's end."""
    base = hbar_chart(df, value, title, value_title)
    order = df.sort_values(value, ascending=False)["System"].tolist()
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=3, fontSize=11)
        .encode(
            y=alt.Y("System:N", sort=order, title=None),
            x=alt.X(f"{value}:Q"),
            text=alt.Text(f"{value}:Q", format=".1f"),
        )
    )
    return (base + labels).properties(
        width=640, height=max(180, 26 * len(order) + 50))


def _scatter_label_layers(df: pd.DataFrame):
    """Split points into (dy, side, sub-frame) label groups so labels of
    neighboring points don't print through each other.

    Packing runs in canvas pixel space via an affine data-to-pixel mapping
    calibrated against actual vl-convert renders of single-point probes
    (residuals < 0.2 px) for the axis config tradeoff_scatter builds:
    width 640, height 380, log x over [min*0.8, max*1.2], y over [0, 100].
    The plot area then starts at (43.4, 9.5) inside the ~690x427 canvas; the
    chart title shifts everything down uniformly and needs no adjustment."""
    dmin = math.log10(df["s per question"].min() * 0.8)
    slope = 640 / (math.log10(df["s per question"].max() * 1.2) - dmin)

    def px_(v):
        return 43.4 + (math.log10(v) - dmin) * slope

    def py_(acc):
        return 9.5 + (100 - acc) / 100 * 380

    pts = sorted(((px_(v), py_(acc), str(name))
                  for v, acc, name in zip(df["s per question"],
                                          df["Accuracy %"], df["System"])),
                 key=lambda p: p[0])
    # every point marker is an obstacle for every label (radius ~5.4 px,
    # padded to 6); canvas is ~690 px wide with the plot ending at ~683
    placed = [(x - 6, x + 6, y - 6, y + 6) for x, y, _ in pts]

    def collisions(box):
        return sum(1 for b in placed
                   if box[0] - 3 < b[1] and box[1] + 3 > b[0]
                   and box[2] - 2 < b[3] and box[3] + 2 > b[2])

    groups: dict[tuple[int, str], list[str]] = {}
    for x, y, name in pts:
        width = 6.5 * len(name)
        # dy=0 on either side outranks any vertical offset: a label centered
        # on its own marker is the clearest association, especially where two
        # markers stack within ~10 px and an offset label reads as the other
        # dot's name
        spots = [(side, x + 11, x + 11 + width, dy)
                 if side == "right" else (side, x - 11 - width, x - 11, dy)
                 for dy in (0, -16, 16, -32, 32, -48, 48, -64, 64)
                 for side in ("right", "left")]
        spots = [s for s in spots if s[1] >= 2 and s[2] <= 686]
        chosen = next((s for s in spots
                       if collisions((s[1], s[2], y + s[3] - 8, y + s[3] + 6)) == 0),
                      None)
        if chosen is None:  # never drop a label: take the least-colliding spot
            chosen = min(spots, key=lambda s: collisions(
                (s[1], s[2], y + s[3] - 8, y + s[3] + 6)))
        side, x0, x1, dy = chosen
        placed.append((x0, x1, y + dy - 8, y + dy + 6))
        groups.setdefault((dy, side), []).append(name)
    return [(dy, side, df[df["System"].isin(names)])
            for (dy, side), names in groups.items()]


def tradeoff_scatter(df: pd.DataFrame, title: str):
    """Accuracy vs latency: every system one labeled point. All layers
    share one explicit x/y scale — per-layer auto domains would place
    subsets' labels on a different coordinate system than the points."""
    xscale = alt.Scale(type="log",
                       domain=[df["s per question"].min() * 0.8,
                               df["s per question"].max() * 1.2])
    yscale = alt.Scale(domain=[0, 100])
    points = (
        alt.Chart(df, title=title)
        .mark_circle(size=90)
        .encode(
            x=alt.X("s per question:Q", scale=xscale,
                    title="Mean latency per question, s (log)"),
            y=alt.Y("Accuracy %:Q", scale=yscale, title="Accuracy %"),
            tooltip=[alt.Tooltip("System:N"),
                     alt.Tooltip("Accuracy %:Q", format=".1f"),
                     alt.Tooltip("s per question:Q", format=".3f")],
        )
    )
    layers = [points]
    for dy, side, sub in _scatter_label_layers(df):
        layers.append(
            alt.Chart(sub)
            .mark_text(align="left" if side == "right" else "right",
                       dx=11 if side == "right" else -11, dy=dy, fontSize=10)
            .encode(x=alt.X("s per question:Q", scale=xscale),
                    y=alt.Y("Accuracy %:Q", scale=yscale),
                    text="System:N"))
    return alt.layer(*layers).interactive().properties(
        width=640, height=380)


def spectrum_line(df: pd.DataFrame, y_title: str, title: str):
    """Accuracy across the measured difficulty range, one line per system."""
    return (
        alt.Chart(df, title=title)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Question difficulty ≤:Q", scale=alt.Scale(domain=[0, 1]),
                    title="Question difficulty ≤ (fraction of systems "
                          "that failed it)"),
            y=alt.Y(f"{y_title}:Q", scale=alt.Scale(domain=[0, 100]),
                    title=y_title),
            color=alt.Color("System:N",
                            legend=alt.Legend(columns=2,
                                              labelFontSize=11)),
            tooltip=["System:N", "Question difficulty ≤:Q",
                     alt.Tooltip(f"{y_title}:Q", format=".1f")],
        )
        .properties(width=640, height=420)
    )


def accuracy_heatmap(df: pd.DataFrame, title: str):
    """System x language accuracy matrix. Color always spans 0-100%."""
    return (
        alt.Chart(df, title=title)
        .mark_rect()
        .encode(
            x=alt.X("System:N", sort=df.groupby("System")["Accuracy %"]
                    .mean().sort_values(ascending=False).index.tolist(),
                    title=None, axis=alt.Axis(labelAngle=-40,
                                              labelFontSize=11)),
            y=alt.Y("Language:N", title=None),
            color=alt.Color("Accuracy %:Q", scale=alt.Scale(
                domain=[0, 100], scheme="redyellowgreen"),
                legend=alt.Legend(format=".0f")),
            tooltip=["System:N", "Language:N",
                     alt.Tooltip("Accuracy %:Q", format=".0f")],
        )
        .properties(width=640, height=280)
    )


# ------------------------------------------------- classification bench tab
def build_classification_tab():
    import gradio as gr

    path = os.path.join(os.path.dirname(BENCH_FILE),
                        "bench_spectrum_results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            bench = json.load(fh)
        if not isinstance(bench, dict) or not isinstance(
                bench.get("systems"), dict):
            raise ValueError
    except (OSError, ValueError):
        gr.Markdown("### Run `bench_spectrum.py --system <name>` first.")
        return
    systems = bench["systems"]
    n_q = len(next(iter(systems.values()))["cls_correct"])

    # difficulty per question: fraction of answering systems that failed it
    difficulty = []
    for i in range(n_q):
        answers = [s["cls_correct"][i] for s in systems.values()]
        difficulty.append(1 - sum(answers) / len(answers))

    n_sys = len(systems)
    gr.Markdown("## Classification benchmark\n"
                f"One mixed pool of {n_q} questions (sentiment + topics, "
                f"varying difficulty) answered by all {n_sys} systems. A "
                "question's difficulty is the fraction of systems that "
                "answered it wrong (0-1).")
    rows = [[sys_, round(s["cls_accuracy"] * 100, 1), s["cls_mean_latency_s"]]
            for sys_, s in systems.items()]
    gr.DataFrame(rows, headers=["System", "Accuracy %", "s per question"],
                 datatype=["str", "number", "number"],
                 label=f"Overall accuracy on the mixed pool ({n_q} "
                       "questions)")

    summary = pd.DataFrame([
        {"System": s, "Accuracy %": round(v["cls_accuracy"] * 100, 1),
         "s per question": v["cls_mean_latency_s"]}
        for s, v in systems.items()])
    gr.Plot(hbar_chart_labeled(summary, "Accuracy %",
                               "Classification accuracy, mixed pool",
                               "Accuracy %"))
    gr.Plot(hbar_chart(summary.sort_values("s per question"), "s per question",
                       "Mean latency per question (CPU; Jev/Laya batch ÷ n)",
                       "seconds"))
    gr.Plot(tradeoff_scatter(summary, "Speed vs accuracy — up and left "
                                      "is better"))

    thresholds = sorted({round(t / 20, 2) for t in range(21)})
    spec_rows = []
    for sys_, s in systems.items():
        for th in thresholds:
            idx = [i for i in range(n_q) if difficulty[i] <= th]
            if len(idx) < 4:
                continue
            acc = sum(s["cls_correct"][i] for i in idx) / len(idx)
            spec_rows.append({"Question difficulty ≤": th,
                              "System": sys_,
                              "Accuracy %": round(acc * 100, 1)})
    if spec_rows:
        gr.Plot(spectrum_line(pd.DataFrame(spec_rows), "Accuracy %",
                              "Accuracy vs question difficulty — each point "
                              "is accuracy on questions at most this hard"))

    gr.Markdown("#### Determinism\nThe seven flat-suite systems were "
                "measured 5x in `bench.py`: **100% output-stable** on "
                "every repeat. Details: bench_results.json.")


# ----------------------------------------------------- extraction bench tab
def build_extraction_tab():
    import gradio as gr

    path = os.path.join(os.path.dirname(BENCH_FILE),
                        "bench_spectrum_results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            bench = json.load(fh)
        if not isinstance(bench, dict) or not isinstance(
                bench.get("systems"), dict):
            raise ValueError
    except (OSError, ValueError):
        gr.Markdown("### Run `bench_spectrum.py --system <name>` first.")
        return
    extractors = {s: v for s, v in bench["systems"].items()
                  if "ner_exact" in v}
    if not extractors:
        gr.Markdown("### No NER results yet.")
        return
    n_q = len(next(iter(extractors.values()))["ner_exact"])
    difficulty = []
    for i in range(n_q):
        answers = [s["ner_exact"][i] for s in extractors.values()]
        difficulty.append(1 - sum(answers) / len(answers))

    gr.Markdown("## Extraction benchmark (NER)\n"
                f"One mixed pool of {n_q} questions, scored as exact "
                "match: a question counts only if the system returned "
                "exactly the gold span set. A question's difficulty is "
                "the fraction of extractors that failed it. Span-less "
                "systems (decision engines, classifiers) are measured in "
                "the Classification benchmark tab.")
    rows = [[s, round(v["ner_exact_rate"] * 100, 1),
             v["ner_mean_latency_s"]] for s, v in extractors.items()]
    gr.DataFrame(rows, headers=["System", "Exact-match %", "s per text"],
                 datatype=["str", "number", "number"],
                 label=f"Exact span-set match on the mixed pool ({n_q} "
                       "questions)")
    summary = pd.DataFrame([
        {"System": s, "Exact match %": round(v["ner_exact_rate"] * 100, 1),
         "s per text": v["ner_mean_latency_s"]}
        for s, v in extractors.items()])
    gr.Plot(hbar_chart_labeled(summary, "Exact match %",
                               "NER exact-match rate, mixed pool",
                               "Exact match %"))
    thresholds = sorted({round(t / 20, 2) for t in range(21)})
    spec_rows = []
    for sys_, s in extractors.items():
        for th in thresholds:
            idx = [i for i in range(n_q) if difficulty[i] <= th]
            if len(idx) < 3:
                continue
            rate = sum(s["ner_exact"][i] for i in idx) / len(idx)
            spec_rows.append({"Question difficulty ≤": th,
                              "System": sys_,
                              "Exact match %": round(rate * 100, 1)})
    if spec_rows:
        gr.Plot(spectrum_line(pd.DataFrame(spec_rows), "Exact match %",
                              "Exact match vs question difficulty"))


# ------------------------------------------------- multilingual bench tab
def build_multilingual_tab():
    import gradio as gr

    path = os.path.join(os.path.dirname(BENCH_FILE),
                        "bench_multilingual_results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            ml = json.load(fh)
        if not isinstance(ml, dict) or not isinstance(
                ml.get("by_language"), dict):
            raise ValueError
    except (OSError, ValueError):
        gr.Markdown("### Run `bench_multilingual.py --system <name>` first.")
        return
    by_language = ml["by_language"]
    langs = list(next(iter(by_language.values())).keys())
    systems = list(by_language.keys())

    gr.Markdown("## Multilingual benchmark\n"
                f"9 languages, no English; 6 texts per language (2 "
                "positive / 2 negative / 2 neutral), labels in English — "
                f"{len(langs) * 6} texts answered by all "
                f"{len(systems)} systems.\n\n"
                "Sampling spread — popular: Spanish, French, Chinese · "
                "medium: Vietnamese, Turkish, Ukrainian · rare: Sinhala, "
                "Icelandic, Welsh. Color in the heatmap always spans "
                "0-100%.")
    order = (pd.DataFrame({s: [v[l] for l in langs]
                           for s, v in by_language.items()},
                          index=langs).mean()
             .sort_values(ascending=False).index.tolist())
    rows = [[s] + [f"{by_language[s][lang] * 100:.0f}"
                   for lang in langs]
            + [f"{sum(by_language[s][l] for l in langs) / len(langs) * 100:.0f}"]
            for s in order]
    gr.DataFrame(rows, headers=["System"] + langs + ["All"],
                 datatype=["str"] * (len(langs) + 2),
                 label="Accuracy by language (%)")
    heat = pd.DataFrame([{"System": s, "Language": lang,
                          "Accuracy %": round(by_language[s][lang] * 100, 1)}
                         for s in systems for lang in langs])
    gr.Plot(accuracy_heatmap(heat, "Accuracy by system and language (%)"))
    overall = pd.DataFrame([
        {"System": s,
         "Accuracy %": round(sum(by_language[s][l] for l in langs)
                             / len(langs) * 100, 1)}
        for s in systems])
    gr.Plot(hbar_chart_labeled(overall, "Accuracy %",
                               "Overall multilingual accuracy (54 texts)",
                               "Accuracy %"))
    if "laya_routing" in ml:
        routes = ml["laya_routing"]
        gr.Markdown("#### Laya Router checkpoint selection\nLaya's Router "
                    "detects the script of each input and routes it to an "
                    "English or multilingual checkpoint before answering — "
                    "per sentence, so a language can use both.")
        rows = [[lang,
                 ", ".join(f"{ckpt} ×{n}" if n > 1 else ckpt
                           for ckpt, n in sorted(counts.items(),
                                                 key=lambda kv: -kv[1]))]
                for lang, counts in routes.items()]
        gr.DataFrame(rows, headers=["Language", "Routed checkpoints"],
                     datatype=["str", "str"],
                     label="Checkpoints used per language (6 texts each)")


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
            return {f"{i + 1}. {text[:40]}…": {
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
| Multilingual | ✅ multi ckpt (89% over 9 langs here) | ❌ English (63%) | ✅ large 81% over 9 langs | ✅ Router, 100+ langs (76%) | option-marker: 48% over 9 langs | = base LLM's languages (37%) | ✅ 100% incl. Sinhala |
| Runs offline / data local | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Cost | free | free | free | free | free | free | $0.042/1M input |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT (lib) | proprietary API |
| Notable | boundary architecture | layout-aware + embeddings | purpose-built classifier, 16 ms/text at edge size | RLCD calibration, script-detecting Router | TypeSafe /v1/systemone protocol-compatible | turns any ChatML LLM into a decision engine via logprobs | 255-choice cap, ECE 0.246 (3rd-party measured) |

Three benchmark tabs follow from this table: **Classification** (every
system, one mixed pool), **Extraction** (the five span-producing systems)
and **Multilingual** (9 languages, no English, every system).
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
            answers = {f"{i + 1}. {text[:40]}…": {
                "label": payload["answers"][f"t{i}"].get("choice"),
                "confidence": payload["answers"][f"t{i}"].get("confidence"),
                "probabilities": payload["answers"][f"t{i}"].get(
                    "probabilities"),
            } for i, text in enumerate(texts)}
            answers["_usage"] = payload.get("usage")
            answers["_latency_s"] = payload.get("_latency_s")
            return answers

        jev_button.click(run_jev, [jev_text, jev_labels, jev_task], jev_table)


# ------------------------------------------------------------ gliclass tab
GLICLASS_MODELS = {
    "edge — 33M, fastest": "knowledgator/gliclass-edge-v3.0",
    "modern-base — DeBERTa-v3": "knowledgator/gliclass-modern-base-v3.0",
    "base": "knowledgator/gliclass-base-v3.0",
    "large — strongest here": "knowledgator/gliclass-large-v3.0",
}
_GLICLASS_PIPES: dict[str, object] = {}


def get_gliclass_pipe(model_id: str):
    if model_id not in _GLICLASS_PIPES:
        from transformers import AutoTokenizer

        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        model = GLiClassModel.from_pretrained(model_id)
        _GLICLASS_PIPES[model_id] = ZeroShotClassificationPipeline(
            model, AutoTokenizer.from_pretrained(model_id),
            classification_type="multi-label", device="cpu")
    return _GLICLASS_PIPES[model_id]


def build_gliclass_tab():
    import gradio as gr

    with gr.Tab("GLiClass"):
        gr.Markdown("### GLiClass v3.0 — purpose-built zero-shot "
                    "classifier\n"
                    "Scores every label against the text in a single "
                    "forward pass (no NLI entailment pairs). First click "
                    "per checkpoint loads it (~10-30 s).")
        gc_model = gr.Dropdown(choices=list(GLICLASS_MODELS.items()),
                               value="knowledgator/gliclass-edge-v3.0",
                               label="Checkpoint")
        gc_text = gr.Textbox(
            label="Text",
            value="Oh great, my package finally arrived — only two weeks "
                  "late and crushed.", lines=3)
        gc_labels = gr.Textbox(label="Labels (comma-separated)",
                               value="positive, negative, neutral")
        gc_button = gr.Button("Classify", variant="primary")
        gc_out = gr.JSON(label="Scores (all labels, one pass)")

        def run_gliclass(model_id, text, labels_csv):
            labels = parse_labels(labels_csv)
            if not text or not labels:
                return {"error": "provide text and at least one label"}
            try:
                pipe = get_gliclass_pipe(model_id)
                out = pipe(text, labels, threshold=0.0)[0]
            except Exception as exc:
                return {"error": str(exc)}
            ranked = sorted(out, key=lambda item: -item["score"])
            return {"top": ranked[0]["label"],
                    "scores": {row["label"]: round(row["score"], 4)
                               for row in ranked}}

        gc_button.click(run_gliclass, [gc_model, gc_text, gc_labels], gc_out)


# ----------------------------------------------------------------- von tab
VON_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      ".venv-von", "Scripts", "python.exe")


def build_von_tab():
    import gradio as gr
    import subprocess

    with gr.Tab("von"):
        gr.Markdown("### von-1.0 — local decision engine on the System One "
                    "protocol\n"
                    "396M ModernBERT with its trained option-marker head, "
                    "Apache 2.0. It needs transformers 5, "
                    "so the app runs it in a separate venv (`.venv-von`): "
                    "each click spawns `von_demo.py`, which loads the model "
                    "once and answers every line in that single process. "
                    "Install `requirements-von.txt` in that environment; "
                    "first use downloads the complete trained checkpoint.")
        von_text = gr.Textbox(
            label="Texts (one per line)",
            value="The food was cold and the waiter was rude.\n"
                  "This is the best laptop I have ever owned.\n"
                  "The meeting is scheduled for 3 PM.", lines=5)
        von_labels = gr.Textbox(
            label="Choices (comma-separated; optionally label: description)",
            value="positive, negative, neutral")
        von_instr = gr.Textbox(
            label="Instructions",
            value="What is the overall sentiment of this text?")
        von_button = gr.Button("Decide", variant="primary")
        von_out = gr.JSON(label="Per line: choice, probabilities, "
                                "confidence")

        def run_von(texts_block, labels_csv, instructions):
            texts = [line.strip() for line in texts_block.splitlines()
                     if line.strip()]
            choices: dict[str, str | None] = {}
            for chunk in parse_labels(labels_csv):
                label, _, desc = chunk.partition(":")
                choices[label.strip()] = desc.strip() or None
            if not texts or not choices:
                return {"error": "provide text lines and choices"}
            helper = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "von_demo.py")
            try:
                proc = subprocess.run(
                    [VON_PY, helper],
                    input=json.dumps({"texts": texts,
                                      "instructions": instructions,
                                      "choices": choices}),
                    capture_output=True, text=True, timeout=180)
                payload = json.loads(proc.stdout)
            except Exception as exc:
                return {"error": str(exc)}
            if "error" in payload:
                return payload
            return {f"{i + 1}. {text[:40]}…": row
                    for i, (text, row)
                    in enumerate(zip(texts, payload["results"]))}

        von_button.click(run_von, [von_text, von_labels, von_instr], von_out)


# ----------------------------------------------------------------- so1 tab
_SO1_DECIDER = None


def get_so1_decider():
    global _SO1_DECIDER
    if _SO1_DECIDER is None:
        from so1 import Decider

        _SO1_DECIDER = Decider.from_pretrained("Qwen/Qwen2.5-0.5B",
                                               backend="hf")
    return _SO1_DECIDER


def build_so1_tab():
    import gradio as gr

    with gr.Tab("so1"):
        gr.Markdown("### so1 — open-alternative-jev\n"
                    "Wraps any ChatML LLM (here Qwen2.5-0.5B on CPU) into "
                    "a choice engine by reading each label's logprob from "
                    "one packed prompt. First click loads the LLM (~20 s). "
                    "Swap in a bigger LLM for better judgments — the "
                    "harness stays the same.")
        so_text = gr.Textbox(
            label="Texts (one per line)",
            value="The food was cold and the waiter was rude.\n"
                  "This is the best laptop I have ever owned.\n"
                  "The meeting is scheduled for 3 PM.", lines=5)
        so_labels = gr.Textbox(label="Labels (comma-separated)",
                               value="positive, negative, neutral")
        so_task = gr.Textbox(label="Question name (asked of the LLM)",
                             value="sentiment")
        so_button = gr.Button("Decide", variant="primary")
        so_out = gr.JSON(label="Per line: choice, probabilities, "
                               "confidence")

        def run_so1(texts_block, labels_csv, question):
            from so1 import Choice

            texts = [line.strip() for line in texts_block.splitlines()
                     if line.strip()]
            labels = parse_labels(labels_csv)
            if not texts or not labels:
                return {"error": "provide text lines and labels"}
            try:
                decider = get_so1_decider()
                rows = []
                for text in texts:
                    row = decider.decide(
                        state=text,
                        questions=[Choice(question, labels)],
                        mode="separate")[0]
                    rows.append({
                        "choice": row.choice,
                        "probabilities": {label: round(prob, 4) for label, prob
                                          in zip(labels,
                                                 row.probabilities or [])},
                        "confidence": round(row.confidence, 4),
                    })
            except Exception as exc:
                return {"error": str(exc)}
            return {f"{i + 1}. {text[:40]}…": row
                    for i, (text, row) in enumerate(zip(texts, rows))}

        so_button.click(run_so1, [so_text, so_labels, so_task], so_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="7-family demo + benchmark")
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

    with gr.Blocks(title="Zero-shot IE & classification bench") as app:
        gr.Markdown("# Zero-shot information extraction & classification\n"
                    "Every family in the comparison has a live tab: three "
                    "local extractors/classifiers (GLiNER 2.5, GLiFormer, "
                    "GLiClass), three local decision engines (Laya, von, "
                    "so1) and the cloud Jev. Benchmark tabs hold the "
                    "measured numbers.")
        build_gliner_tab(gliner)
        build_gliformer_tab(gliformer)
        build_gliclass_tab()
        build_laya_tab()
        build_von_tab()
        build_so1_tab()
        build_jev_tab()
        with gr.Tab("Classification benchmark"):
            build_classification_tab()
        with gr.Tab("Extraction benchmark"):
            build_extraction_tab()
        with gr.Tab("Multilingual benchmark"):
            build_multilingual_tab()
        with gr.Tab("Compare"):
            build_compare_tab()

    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
