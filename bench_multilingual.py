r"""Multilingual zero-shot classification benchmark (no English).

9 languages x 6 texts (2 positive, 2 negative, 2 neutral sentiment), same
sentence meanings across languages, labels in English (standard zero-shot
cross-lingual setup):

  popular : Spanish, French, Chinese (Simplified)
  medium  : Vietnamese, Turkish, Ukrainian
  rare    : Sinhala, Icelandic, Welsh  (Sinhala required by the owner)

Every system in the comparison answers the same 54 texts. Run one system
per invocation (results merge into the shared file):

    python bench_multilingual.py --system GLiNER2.5-base
    .venv-von/Scripts/python bench_multilingual.py --system von
    .venv-von/Scripts/python bench_multilingual.py --system JevK5-Lite
    .venv-von/Scripts/python bench_multilingual.py --system "LFM2.5-RLCD 350M"

Jev is a paid API: it runs all 54 texts as one batched request.
Kev 0.8B needs its local server running first (System One contract):
    cd ../kev && uv run --extra serve python -m kev.serve \
        --run jaredpalmer/kev-0.8b --port 8009
AgentJev 0.6B likewise (own /api/evaluate contract):
    cd ../agent-jev && <python> -m jev_service.server \
        --checkpoint agentjev_v1.pt --model-path <Qwen3-0.6B snapshot> \
        --temperatures temperatures.json --port 8149 --device cpu
decider 0.8B and OpenThai 0.8B serve the System One contract as well
(shared agent-jev venv): DECIDER_MODEL=Mapika/decider-0.8b DECIDER_DEVICE=cpu
<py> -m uvicorn decider.serve:app --port 8018, respectively
OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne <py> -m uvicorn
openthai_systemone.server:app --port 8029. Verdict 151M runs in-process
from the Verdict-open-jev checkout (VERDICT_HOME, default C:\src\verdict)
under the agent-jev venv python.

Ground-truth verification: sentences were blind-translated back to English
by one independent cold-context model instance (separate agent, no labels
in its instructions); it caught 8 errors (2 sentiment-flipping) which were
fixed and re-verified. The full sentence table is in the PR description
for human (native-speaker) review, Sinhala especially.

Output: bench_multilingual_results.json (rendered by the web UI's
Multilingual benchmark tab).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

SENTIMENT_LABELS = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual text without a clear attitude",
}

# English templates (for the PR table only; not benchmarked):
# P1 "The food at this restaurant is delicious, and the staff are very friendly."
# P2 "I am very happy with my new phone; the battery lasts all day."
# N1 "The bus was crowded and slow, and I arrived very late."
# N2 "This movie was long and boring; I regret watching it."
# T1 "The train to the capital leaves at seven in the morning."
# T2 "This book was published in 2019 and has 300 pages."

LANGUAGES: dict[str, list[tuple[str, str]]] = {
    # ---- popular -----------------------------------------------------
    "Spanish": [
        ("La comida de este restaurante está deliciosa y el personal es muy amable.", "positive"),
        ("Estoy muy contento con mi teléfono nuevo; la batería dura todo el día.", "positive"),
        ("El autobús estaba lleno y lento, y llegué muy tarde.", "negative"),
        ("Esta película fue larga y aburrida; me arrepiento de haberla visto.", "negative"),
        ("El tren a la capital sale a las siete de la mañana.", "neutral"),
        ("Este libro fue publicado en 2019 y tiene 300 páginas.", "neutral"),
    ],
    "French": [
        ("La nourriture de ce restaurant est délicieuse et le personnel est très aimable.", "positive"),
        ("Je suis très content de mon nouveau téléphone ; la batterie tient toute la journée.", "positive"),
        ("Le bus était bondé et lent, et je suis arrivé très en retard.", "negative"),
        ("Ce film était long et ennuyeux ; je regrette de l'avoir vu.", "negative"),
        ("Le train pour la capitale part à sept heures du matin.", "neutral"),
        ("Ce livre a été publié en 2019 et compte 300 pages.", "neutral"),
    ],
    "Chinese": [
        ("这家餐厅的菜品很美味，员工也非常友好。", "positive"),
        ("我对新手机非常满意；电池能用一整天。", "positive"),
        ("公交车又挤又慢，我到得很晚。", "negative"),
        ("这部电影又长又无聊；我后悔看了它。", "negative"),
        ("开往首都的火车早上七点发车。", "neutral"),
        ("这本书于2019年出版，共300页。", "neutral"),
    ],
    # ---- medium ------------------------------------------------------
    "Vietnamese": [
        ("Món ăn ở nhà hàng này rất ngon và nhân viên rất thân thiện.", "positive"),
        ("Tôi rất hài lòng với chiếc điện thoại mới; pin dùng được cả ngày.", "positive"),
        ("Xe buýt vừa đông vừa chậm, và tôi đến rất muộn.", "negative"),
        ("Bộ phim này dài và nhàm chán; tôi hối tiếc vì đã xem.", "negative"),
        ("Tàu đi thủ đô khởi hành lúc bảy giờ sáng.", "neutral"),
        ("Cuốn sách này xuất bản năm 2019 và có 300 trang.", "neutral"),
    ],
    "Turkish": [
        ("Bu restoranın yemeği çok lezzetli ve personel çok güler yüzlü.", "positive"),
        ("Yeni telefonumdan çok memnunum; pil bütün gün dayanıyor.", "positive"),
        ("Otobüs kalabalık ve yavaştı, çok geç kaldım.", "negative"),
        ("Bu film uzundu ve sıkıcıydı; izlediğime pişman oldum.", "negative"),
        ("Başkente giden tren sabah yedide kalkıyor.", "neutral"),
        ("Bu kitap 2019'da yayımlandı ve 300 sayfası var.", "neutral"),
    ],
    "Ukrainian": [
        ("Їжа в цьому ресторані смачна, а персонал дуже привітний.", "positive"),
        ("Я дуже задоволений своїм новим телефоном; батареї вистачає на весь день.", "positive"),
        ("Автобус був переповнений і повільний, і я дуже запізнився.", "negative"),
        ("Цей фільм був довгим і нудним; я шкодую, що подивився його.", "negative"),
        ("Потяг до столиці відправляється о сьомій ранку.", "neutral"),
        ("Цю книгу опубліковано у 2019 році, і вона має 300 сторінок.", "neutral"),
    ],
    # ---- rare --------------------------------------------------------
    "Sinhala": [
        ("මෙම අවන්හලේ ආහාර රසවත් ය, සේවකයෝ ද ඉතා හිතවත් ය.", "positive"),
        ("මගේ අලුත් දුරකථනය ගැන මම ඉතා සතුටු ය; බැටරිය දිනය පුරා පවතී.", "positive"),
        ("බස් රථය පිරී සිටි අතර සෙමින් ගමන් කළේ ය; මම ඉතා ප්‍රමාද වීමි.", "negative"),
        ("මෙම චිත්‍රපටය දිගු හා කලකිරීම්බර ය; එය නැරඹූ බවට මම පසුතැවෙමි.", "negative"),
        ("අගනුවරට යන දුම්රිය උදේ හතට පිටත් වේ.", "neutral"),
        ("මෙම පොත 2019 දී ප්‍රකාශයට පත් වූ අතර පිටු 300 ක් ඇත.", "neutral"),
    ],
    "Icelandic": [
        ("Maturinn á þessum veitingastað er ljúffengur og starfsfólkið er mjög vinalegt.", "positive"),
        ("Ég er ánægður með nýja símann minn; rafhlöðan endist allan daginn.", "positive"),
        ("Strætóinn var fullur og hægur, og ég kom mjög seint.", "negative"),
        ("Þessi kvikmynd var löng og leiðinleg; ég iðrast þess að hafa horft á hana.", "negative"),
        ("Lestin til höfuðborgarinnar fer af stað klukkan sjö að morgni.", "neutral"),
        ("Þessi bók var gefin út árið 2019 og er 300 síðna.", "neutral"),
    ],
    "Welsh": [
        ("Mae bwyd y bwyty hwn yn flasus iawn a'r staff yn garedig iawn.", "positive"),
        ("Rwy'n falch iawn o'm ffôn newydd; mae'r batri'n para'r dydd cyfan.", "positive"),
        ("Roedd y bws yn llawn ac yn araf, a chyrhaeddais i'n hwyr iawn.", "negative"),
        ("Roedd y ffilm hon yn hir ac yn ddiflas; mae gennyf edifeirdrwch am ei gweld.", "negative"),
        ("Mae'r trên i'r brifddinas yn gadael am saith y bore.", "neutral"),
        ("Cyhoeddwyd y llyfr hwn yn 2019 ac mae ganddo 300 o dudalennau.", "neutral"),
    ],
}

TIERS = {
    "Spanish": "popular", "French": "popular", "Chinese": "popular",
    "Vietnamese": "medium", "Turkish": "medium", "Ukrainian": "medium",
    "Sinhala": "rare", "Icelandic": "rare", "Welsh": "rare",
}

RESULTS_FILE = "bench_multilingual_results.json"

GLINER = {
    "GLiNER2.5-small": "fastino/gliner2.5-small-v1",
    "GLiNER2.5-base": "fastino/gliner2.5-base-v1",
    "GLiNER2.5-multi": "fastino/gliner2.5-multi-v1",
    "GLiNER2.5-Decide": "fastino/GLiNER2.5-Decide",
}
GLIFORMER = {
    "GLiFormer-base": "knowledgator/gliformer-base-v1",
    "GLiFormer-large": "knowledgator/gliformer-large-v1",
}
GLICLASS = {
    "gliclass-edge": "knowledgator/gliclass-edge-v3.0",
    "gliclass-modern-base": "knowledgator/gliclass-modern-base-v3.0",
    "gliclass-base": "knowledgator/gliclass-base-v3.0",
    "gliclass-large": "knowledgator/gliclass-large-v3.0",
}
ALL_SYSTEMS = (list(GLINER) + list(GLIFORMER) + list(GLICLASS)
               + ["Laya Router", "Laya typed-decisions", "von",
                  "JevK5-Lite", "LFM2.5-RLCD 350M",
                  "so1 (Qwen2.5-0.5B)", "Jev",
                  "Kev 0.8B (local)", "AgentJev 0.6B (local)",
                  "decider 0.8B (local)", "OpenThai 0.8B (local)",
                  "Verdict 151M (local)"])


def make_classifier(name: str):
    """Per-text sentiment classifier for encoder/classifier systems."""
    labels = list(SENTIMENT_LABELS)

    if name in GLINER:
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained(GLINER[name],
                                              map_location="cpu")

        def one(text: str):
            return model.classify_text(text, {"task": labels})["task"]
    elif name in GLIFORMER:
        from gliformer import GLiFormer

        model = GLiFormer.from_pretrained(GLIFORMER[name],
                                          load_tokenizer=True).to("cpu").eval()

        def one(text: str):
            out = model.classify(text, labels, threshold=0.5)
            return out[0]["class_name"] if out else None
    else:  # gliclass
        from transformers import AutoTokenizer

        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        model = GLiClassModel.from_pretrained(GLICLASS[name])
        pipe = ZeroShotClassificationPipeline(
            model, AutoTokenizer.from_pretrained(GLICLASS[name]),
            classification_type="multi-label", device="cpu")

        def one(text: str):
            out = pipe(text, labels, threshold=0.0)[0]
            return max(out, key=lambda row: row["score"])["label"] if out else None
    return one


def make_decider(name: str):
    """Per-text sentiment decider for von / so1, label-head classifier
    for JevK5-Lite (same per-text call shape)."""
    labels = list(SENTIMENT_LABELS)
    if name == "von":
        from von_client import load_von_decider

        decide = load_von_decider()

        def one(text: str):
            return decide(
                state=text, choices=dict(SENTIMENT_LABELS),
                instructions="What is the overall sentiment of this "
                             "text?").choice
    elif name == "JevK5-Lite":
        # in-process label-head classifier, run under the .venv-von python
        # (transformers 5.17 + jevk5)
        from jevk5 import JevK5Lite

        lite = JevK5Lite.from_pretrained("alibiserikbay/JevK5-Lite",
                                         threads=16)

        def one(text: str):
            out = lite.classify(text, {"task": labels})
            return out["task"]["labels"][0] if out["task"]["labels"] else None
    elif name == "LFM2.5-RLCD 350M":
        # in-process constrained-decision engine (vendored rlcd_engine/),
        # run under the .venv-von python (transformers 5.17 + jsonschema)
        from rlcd_engine.engine import Engine

        engine = Engine(device="cpu", dtype="float32")
        schema = {"type": "object",
                  "properties": {"sentiment": {"type": "string",
                                               "description": "The overall "
                                                             "sentiment of "
                                                             "the text",
                                               "enum": labels}},
                  "required": ["sentiment"],
                  "additionalProperties": False}

        def one(text: str):
            res = engine.constrained(text, schema)
            try:
                return json.loads(res["text"])["sentiment"]
            except (KeyError, ValueError):
                return None
    else:
        from so1 import Choice, Decider

        decider = Decider.from_pretrained("Qwen/Qwen2.5-0.5B", backend="hf")

        def one(text: str):
            return decider.decide(state=text,
                                  questions=[Choice("sentiment", labels)],
                                  mode="separate")[0].choice
    return one


def run_laya_router(texts, lang_of):
    from laya import Router

    from jev_client import choice

    router = Router(device="cpu")
    question = choice(
        "What is the overall sentiment of the text in the state: positive, "
        "negative, or neutral?",
        {label: None for label in SENTIMENT_LABELS})
    # Keep every checkpoint used by this pool resident. The Router's default
    # single-model cache otherwise reloads weights whenever scripts alternate.
    models = list(dict.fromkeys(
        router.route({"text": text}, {"q": question})["model"] for text in texts))
    router.preload(names=models)
    preds, lat = [], []
    routed: dict[str, dict[str, int]] = {}
    for text, lang in zip(texts, lang_of):
        t0 = time.perf_counter()
        out = router.predict({"text": text}, {"q": question})
        lat.append(time.perf_counter() - t0)
        preds.append(out["answers"]["q"].get("choice"))
        model = out.get("routing", {}).get("model") or "unknown"
        counts = routed.setdefault(lang, {})
        counts[model] = counts.get(model, 0) + 1
    return preds, statistics.mean(lat), routed


def run_laya_typed(texts):
    """Single English-only typed-decisions checkpoint, no routing."""
    import laya

    from jev_client import choice

    agent = laya.load("convaiinnovations/laya-typed-decisions")
    question = choice(
        "What is the overall sentiment of the text in the state: positive, "
        "negative, or neutral?",
        {label: None for label in SENTIMENT_LABELS})
    preds, lat = [], []
    for text in texts:
        t0 = time.perf_counter()
        out = agent.predict({"text": text}, {"q": question})
        lat.append(time.perf_counter() - t0)
        preds.append(out["answers"]["q"].get("choice"))
    return preds, statistics.mean(lat)


def run_jev(texts):
    from jev_client import JevClient, choice

    client = JevClient()
    questions = {
        f"t{i}": choice(
            {"task": "sentiment classification of this text",
             "text": text}, SENTIMENT_LABELS)
        for i, text in enumerate(texts)
    }
    t0 = time.perf_counter()
    payload = client.ask({"task": "sentiment classification",
                          "labels": SENTIMENT_LABELS}, questions)
    dt = time.perf_counter() - t0
    return ([payload["answers"][f"t{i}"].get("choice")
             for i in range(len(texts))], dt / len(texts))


def run_systemone(texts, port: int, model: str):
    """Local System One server (Kev/decider/OpenThai), one request per text
    so latency is comparable with the other local models. String
    instructions — the shape these servers and Laya both expect."""
    from jev_client import JevClient, choice

    client = JevClient(base_url=f"http://127.0.0.1:{port}/v1/systemone",
                       model=model)
    # pay any lazy model loading before the timed loop; OpenThai's cold
    # load runs minutes, past ask()'s 120 s default
    client.ask({"task": "warmup"},
               {"w": choice('Sentiment of "good"?',
                            {"positive": None, "negative": None})},
               timeout=600)
    preds, lat = [], []
    for text in texts:
        question = choice(
            f'What is the overall sentiment of this text: "{text}"',
            {label: None for label in SENTIMENT_LABELS})
        t0 = time.perf_counter()
        out = client.ask({"task": "sentiment"}, {"q": question})
        lat.append(time.perf_counter() - t0)
        preds.append(out["answers"]["q"].get("choice"))
    return preds, statistics.mean(lat)


def run_verdict(texts):
    """Verdict 151M: in-process rlcd DecisionEngine (Verdict-open-jev
    checkout, agent-jev venv); abstention counts as no prediction."""
    home = os.environ.get("VERDICT_HOME", r"C:\src\verdict")
    sys.path.insert(0, home)
    from rlcd import Choice, DecisionEngine, Option

    engine = DecisionEngine(model_name_or_path=os.path.join(
        home, "artifacts", "v2"), device="cpu")
    options = [Option(id=l, description=d)
               for l, d in SENTIMENT_LABELS.items()]
    preds, lat = [], []
    for text in texts:
        query = Choice(id="q", question="What is the overall sentiment "
                                        "of this text?", options=options)
        t0 = time.perf_counter()
        res = engine.evaluate(context=text, queries=[query]).results[0]
        lat.append(time.perf_counter() - t0)
        preds.append(None if res.is_abstention else res.selected_id)
    return preds, statistics.mean(lat)


def run_agentjev(texts):
    """AgentJev-0.6B: own loopback contract (port 8149), one request per
    text, fixed per-label option descriptions (no per-question leakage)."""
    from agentjev_client import ask

    options = {label: f"The text expresses {label} sentiment"
               for label in SENTIMENT_LABELS}
    preds, lat = [], []
    for text in texts:
        question = {"id": "q", "type": "choice",
                    "question": f'What is the overall sentiment of this '
                                f'text: "{text}"',
                    "options": options}
        t0 = time.perf_counter()
        out = ask({"task": "sentiment"}, {"q": question})
        lat.append(time.perf_counter() - t0)
        preds.append(out["q"]["value"])
    return preds, statistics.mean(lat)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=ALL_SYSTEMS)
    name = parser.parse_args().system

    texts, gold, lang_of = [], [], []
    for lang, items in LANGUAGES.items():
        for text, label in items:
            texts.append(text)
            gold.append(label)
            lang_of.append(lang)

    laya_routes = None
    print(f"{name}: 54 texts (9 languages x 6) ...")
    t0 = time.perf_counter()
    if name in GLINER or name in GLIFORMER or name in GLICLASS:
        one = make_classifier(name)
        preds, lat = [], []
        for text in texts:
            t1 = time.perf_counter()
            preds.append(one(text))
            lat.append(time.perf_counter() - t1)
        lat = statistics.mean(lat)
    elif name in ("von", "so1 (Qwen2.5-0.5B)", "JevK5-Lite",
                  "LFM2.5-RLCD 350M"):
        one = make_decider(name)
        preds, lat = [], []
        for text in texts:
            t1 = time.perf_counter()
            preds.append(one(text))
            lat.append(time.perf_counter() - t1)
        lat = statistics.mean(lat)
    elif name == "Laya Router":
        preds, lat, laya_routes = run_laya_router(texts, lang_of)
    elif name == "Laya typed-decisions":
        preds, lat = run_laya_typed(texts)
    elif name == "Kev 0.8B (local)":
        preds, lat = run_systemone(texts, 8009, "kev-latest")
    elif name == "AgentJev 0.6B (local)":
        preds, lat = run_agentjev(texts)
    elif name == "decider 0.8B (local)":
        preds, lat = run_systemone(texts, 8018, "decider-0.8b")
    elif name == "OpenThai 0.8B (local)":
        preds, lat = run_systemone(texts, 8029, "openthai-systemone")
    elif name == "Verdict 151M (local)":
        preds, lat = run_verdict(texts)
    else:  # Jev
        preds, lat = run_jev(texts)

    try:
        with open(RESULTS_FILE, encoding="utf-8") as fh:
            out = json.load(fh)
        if not isinstance(out, dict) or not all(
                isinstance(out.get(k), dict)
                for k in ("by_language", "by_tier", "latency")):
            raise ValueError
    except (OSError, ValueError):
        out = {"meta": {}, "by_language": {}, "by_tier": {}, "latency": {}}
    out.setdefault("meta", {}).setdefault("notes", []).append(
        f"{name} recorded {time.strftime('%Y-%m-%d %H:%M')}, "
        f"{time.perf_counter() - t0:.0f}s, CPU")

    by_lang = {}
    for lang in LANGUAGES:
        idx = [i for i, lg in enumerate(lang_of) if lg == lang]
        by_lang[lang] = round(sum(1 for i in idx if preds[i] == gold[i])
                              / len(idx), 4)
    by_tier = {}
    for tier in ("popular", "medium", "rare"):
        langs = [lg for lg, t in TIERS.items() if t == tier]
        by_tier[tier] = round(statistics.mean([by_lang[lg] for lg in langs]), 4)

    out["by_language"][name] = by_lang
    out["by_tier"][name] = by_tier
    out["latency"][name] = round(lat, 3)
    notes = out.setdefault("meta", {}).setdefault("notes", [])
    if notes and notes[0].startswith("All "):
        notes[0] = (f"All {len(out['by_language'])} systems answer "
                    "the same 54 texts.")
    warmed = name.startswith(("Kev", "decider", "OpenThai"))
    out.setdefault("timing", {})[name] = (
        "Model download and loading excluded; "
        + ("one untimed warm-up question pays the server's lazy weight "
           "load first - timed latencies are warmed." if warmed
           else "first forward pass included."))
    if name == "von":
        from von_client import provenance

        out.setdefault("provenance", {})[name] = provenance()
    if laya_routes:
        out["laya_routing"] = laya_routes
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS_FILE)

    overall = statistics.mean(by_lang.values())
    print(f"  overall {overall * 100:.1f}%  "
          f"(popular {by_tier['popular'] * 100:.0f}% / medium "
          f"{by_tier['medium'] * 100:.0f}% / rare "
          f"{by_tier['rare'] * 100:.0f}%)  "
          f"{lat:.3f} s/text -> {RESULTS_FILE}")
    for lang in LANGUAGES:
        print(f"    {lang:<12} {by_lang[lang] * 100:5.1f}%")


if __name__ == "__main__":
    main()
