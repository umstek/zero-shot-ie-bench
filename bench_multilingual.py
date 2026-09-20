"""Multilingual zero-shot classification benchmark (no English).

9 languages x 6 texts (2 positive, 2 negative, 2 neutral sentiment), same
sentence meanings across languages, labels in English (standard zero-shot
cross-lingual setup):

  popular : Spanish, French, Chinese (Simplified)
  medium  : Vietnamese, Turkish, Ukrainian
  rare    : Sinhala, Icelandic, Welsh  (Sinhala required by the owner)

Systems
  - GLiNER2.5-multi  (fastino, mDeBERTa, multilingual checkpoint) - per text
  - GLiFormer-large  (knowledgator, English-only) - CONTROL: expected to fail
  - Laya Router      (convaiinnovations, mmBERT multilingual) - per text
  - Jev              (cloud, jev-latest) - one batched request for all texts

Ground-truth verification: sentences were blind-translated back to English
by one independent cold-context model instance (separate agent, no labels
in its instructions); it caught 8 errors (2 sentiment-flipping) which were
fixed and re-verified. The full sentence table is in the PR description
for human (native-speaker) review, Sinhala especially.

Output: bench_multilingual_results.json (app.py "Benchmarks v2" tab).
"""

from __future__ import annotations

import json
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


def run_gliner_multi(texts, gold):
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1",
                                          map_location="cpu")
    preds, lat = [], []
    for text in texts:
        t0 = time.perf_counter()
        preds.append(model.classify_text(
            text, {"task": list(SENTIMENT_LABELS)})["task"])
        lat.append(time.perf_counter() - t0)
    return preds, statistics.mean(lat)


def run_gliformer_control(texts, gold):
    from gliformer import GLiFormer

    model = GLiFormer.from_pretrained("knowledgator/gliformer-large-v1",
                                      load_tokenizer=True).to("cpu").eval()
    preds, lat = [], []
    for text in texts:
        t0 = time.perf_counter()
        out = model.classify(text, list(SENTIMENT_LABELS), threshold=0.5)
        lat.append(time.perf_counter() - t0)
        preds.append(out[0]["class_name"] if out else None)
    return preds, statistics.mean(lat)


def run_laya_router(texts, gold):
    from laya import Router

    from jev_client import choice

    router = Router()
    question = choice(
        "What is the overall sentiment of the text in the state: positive, "
        "negative, or neutral?",
        {label: None for label in SENTIMENT_LABELS})
    preds, lat, routes = [], [], []
    for text in texts:
        t0 = time.perf_counter()
        out = router.predict({"text": text}, {"q": question})
        lat.append(time.perf_counter() - t0)
        preds.append(out["answers"]["q"].get("choice"))
        routes.append(out.get("routing", {}).get("model"))
    return preds, statistics.mean(lat), routes


def run_jev(texts, gold):
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    texts, gold, lang_of = [], [], []
    for lang, items in LANGUAGES.items():
        for text, label in items:
            texts.append(text)
            gold.append(label)
            lang_of.append(lang)

    results = {}
    print("GLiNER2.5-multi ...")
    preds, lat = run_gliner_multi(texts, gold)
    results["GLiNER2.5-multi"] = (preds, lat)

    print("GLiFormer-large (English-only control) ...")
    preds, lat = run_gliformer_control(texts, gold)
    results["GLiFormer-large (control)"] = (preds, lat)

    print("Laya Router ...")
    preds, lat, routes = run_laya_router(texts, gold)
    results["Laya Router"] = (preds, lat)

    print("Jev (1 batched request) ...")
    preds, lat = run_jev(texts, gold)
    results["Jev"] = (preds, lat)

    out = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M"),
                    "device": "CPU",
                    "texts_per_language": 6,
                    "labels": list(SENTIMENT_LABELS),
                    "verification": "blind back-translation by two "
                    "independent model instances; table in PR for human "
                    "review (Sinhala flagged for native-speaker owner)",
                    "notes": [
                        "GLiFormer-large is an English-only CONTROL - it is "
                        "expected to fail on non-English text; included to "
                        "show what monolingual collapse looks like.",
                        "Laya Router picks its checkpoint per input by "
                        "script detection (english vs multilingual).",
                        "Jev ran all 54 texts as one batched request.",
                    ]},
          "by_language": {}, "by_tier": {}, "latency": {}}

    for system, (preds, lat) in results.items():
        out["latency"][system] = round(lat, 3)
        by_lang = {}
        for lang in LANGUAGES:
            idx = [i for i, l in enumerate(lang_of) if l == lang]
            correct = sum(1 for i in idx if preds[i] == gold[i])
            by_lang[lang] = round(correct / len(idx), 4)
        out["by_language"][system] = by_lang
        by_tier = {}
        for tier in ("popular", "medium", "rare"):
            langs = [l for l, t in TIERS.items() if t == tier]
            scores = [by_lang[l] for l in langs if l in by_lang]
            by_tier[tier] = round(statistics.mean(scores), 4) if scores else None
        out["by_tier"][system] = by_tier

    if "Laya Router" in results:
        out["laya_routing"] = {
            lang: sorted({r for l, r in zip(lang_of, routes) if l == lang})
            for lang in LANGUAGES}

    with open("bench_multilingual_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("\n=== Accuracy by language tier ===")
    for system in results:
        t = out["by_tier"][system]
        print(f"  {system:<28} popular={t['popular']*100:5.1f}%  "
              f"medium={t['medium']*100:5.1f}%  rare={t['rare']*100:5.1f}%")
    print("\n=== Per-language ===")
    for lang in LANGUAGES:
        row = "  ".join(f"{s.split(' (')[0]}:"
                        f"{out['by_language'][s][lang]*100:4.0f}%"
                        for s in results)
        print(f"  {lang:<12} {row}")
    print("\nWrote bench_multilingual_results.json")


if __name__ == "__main__":
    main()
