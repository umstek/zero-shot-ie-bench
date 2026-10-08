"""Loader + call paths for Knowledgator's GLiNER-X family
(knowledgator/gliner-x-{small,base,large}, Apache-2.0, released
2026-04-29) — the multilingual GLiNER generation: mT5 encoder backbone
replacing DeBERTa, trained on fresh LLM-annotated multilingual synthetic
data built from FineWeb-2 (HF-card-only release: no blog, no paper).

Runs in its OWN venv (.venv-glinerx), not the main one: the family loads
through the classic `gliner` package with the `stanza` extra
(`pip install gliner[stanza]`, per the model card), and a fresh resolve
of that stack wants transformers 5.x — incompatible with the main
venv's validated transformers==4.57.6 pin (see README setup). The
checkpoint config sets words_splitter_type=stanza, so stanza + langdetect
are hard requirements: the word splitter language-detects every text and
lazily builds a stanza tokenize pipeline per language (downloaded once
into %LOCALAPPDATA%/StanfordNLP/stanza/Cache on Windows; a failed
download for any detected language falls back to the default 'en'
pipeline, observed for Catalan misdetections on English/Spanish texts).

NER is the card's only task (span-level gliner over an mT5 encoder) and
is the strong path: predict_entities(text, labels, threshold=0.5) with
the same char-offset (start, end, label) tuples the other extractors
return. Text classification is NOT supported by these checkpoints: the
card never claims it, the training data is NER synthetic, and every
mapping we probed collapses onto one label at all three sizes (measured
2026-10-05 on the easy-tier sentiment texts with all of small/base/large:
the vendor multitask prompt below, per-label max scoring over that
prompt, raw-text entity labels, descriptive labels, review-frame and
header-line prompts — all chance-level). For the benches we still answer
the classification sections with the vendor's own documented method (the
gliner.multitask.GLiNERClassifier pipeline: the classification prompt,
top-scoring entity's text as the label, "other" when nothing survives
the threshold) and report those rows as census rows, like the other
chance-level systems — never as a competitive classifier.
"""

MODELS = {
    "GLiNER-X-small": "knowledgator/gliner-x-small",
    "GLiNER-X-base": "knowledgator/gliner-x-base",
    "GLiNER-X-large": "knowledgator/gliner-x-large",
}

# the gliner.multitask classification prompt (GLiNERClassifier.prompt),
# the vendor's documented classification surface for classic GLiNER
CLASSIFICATION_PROMPT = "Classify text into the following classes: {} \n {}"

CLASSIFICATION_THRESHOLD = 0.5  # the multitask pipeline's default
NER_THRESHOLD = 0.5  # predict_entities' default, same as GLiFormer's bench


def classification_prompt(labels: list[str], text: str) -> str:
    """The vendor multitask classification prompt: labels listed in the
    header, the text below (GLiNERClassifier.prepare_texts' exact shape)."""
    return CLASSIFICATION_PROMPT.format(", ".join(labels), text)


def reduce_classification(entities: list[dict], labels: list[str]) -> str:
    """GLiNERClassifier.process_predictions' single-label reduction:
    the highest-scoring entity's text becomes the prediction (whatever
    it says — the benches score it against the gold label either way);
    with no entities above the threshold the vendor answers the literal
    'other'."""
    if not entities:
        return "other"
    return max(entities, key=lambda entity: entity["score"])["text"].strip()


def load_glinerx(model_id: str):
    """Load one GLiNER-X checkpoint on the CPU (fp32 weights, eval mode).
    Requires the .venv-glinerx stack (gliner[stanza]); the error names
    the venv because importing this module from the main venv only fails
    at load time — stanza is absent there by design."""
    try:
        import stanza  # noqa: F401
        import langdetect  # noqa: F401
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            "GLiNER-X needs the classic gliner package with the stanza "
            "extra — run the benchmarks with .venv-glinerx/Scripts/python "
            f"(see README setup; missing module: {getattr(exc, 'name', exc)}"
            ")") from exc
    return GLiNER.from_pretrained(model_id).to("cpu").eval()


def warm_splitter(model, texts: list[str]) -> None:
    """Pay the stanza word-splitter's per-language lazy pipeline builds
    (and any first-use model downloads) before a timed section. The
    'en' pipeline is already built at model load; every other language
    loads on its first text, so the multilingual bench warms one text
    per language here — untimed, like the other local engines' warm-up."""
    splitter = model.data_processor.words_splitter
    for text in texts:
        list(splitter(text))
