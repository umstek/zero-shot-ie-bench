"""Intern-Decision IMAGE-input demo + one-shot runner for the web UI -
runs inside .venv-von.

The Intern-Decision checkpoints are MULTIMODAL decision models: each HF
snapshot ships a vision tower + projector (model-vision-*/model-projector-*
safetensors) next to the language weights, and the snapshot runtime's
predict() accepts 1-8 local image paths per request
({"state": ..., "questions": {...}, "images": [path, ...]}). The images
join the prompt as image_url parts ahead of the state text, the vision
tower encodes them in the same single causal forward pass that scores
the masked <decision> slots - no generation, same choice/score/noul
question types as the text path (demos/intern_decision_demo.py). This
demo exercises that surface; measured on the 0.8B (CPU, fp32), an image
request runs ~3.4 s vs ~1.3 s text-only - the vision tower adds ~2 s.

The sample images are GENERATED with Pillow at run time (deterministic
draws, written to a temp dir) so the repo ships no binary assets:
a support-ticket screenshot, a cafe receipt and a quarterly-revenue
bar chart - one choice, noul and score question each, the same three
typed questions the text demo asks of a ticket string.

Tour (interactive):
    .venv-von/Scripts/python demos/intern_decision_image_demo.py        # 0.8B
    .venv-von/Scripts/python demos/intern_decision_image_demo.py --model 2b

Web-UI runner (the app's Intern-Decision tab spawns this and talks JSON
over stdin/stdout):
    .venv-von/Scripts/python demos/intern_decision_image_demo.py --serve
    stdin:  {"image": "/abs/path.png",          # or "images": [path, ...]
             "instructions": str,
             "type": "choice" | "noul" | "score",
             "criteria": {...choice...} | [...score rubric...],
             "state": str (optional context), "model": "0.8b" (optional)}
    stdout: {"answer": {"choice"/"noul"/"score": ...,
                        "probabilities": ..., "confidence": ...},
             "usage": {...}, "timing": {...}}
            or {"error": "Type: message"}
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import sys
import tempfile
import time
from pathlib import Path

from engines.intern_decision_client import SIZES, normalize_size, size_home

QUESTION_TYPES = ("choice", "noul", "score")


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def load(model_key=None):
    """intern_decision_client.load_engine over the size's local snapshot;
    heavy imports stay inside so the module imports without torch and
    transformers (offline tests). The load time goes to stderr: in
    --serve mode stdout must stay pure JSON for the web UI."""
    from engines.intern_decision_client import load_engine

    size = normalize_size(model_key)
    t0 = time.perf_counter()
    engine = load_engine(size)
    print(f"Loaded {SIZES[size]} from {size_home(size)} in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


# ------------------------------------------------------- sample images
def build_samples(directory=None) -> dict[str, str]:
    """Draw the three deterministic sample images with Pillow and return
    {name: absolute path}. They land in a fresh temp dir unless one is
    given (tests pass their own). No binary assets ship in the repo."""
    from PIL import Image, ImageDraw

    root = Path(directory or tempfile.mkdtemp(prefix="intern-decision-img-"))
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    ticket = Image.new("RGB", (360, 130), "white")
    draw = ImageDraw.Draw(ticket)
    draw.rectangle((0, 0, 359, 129), outline="black", width=2)
    draw.text((14, 12), "SUPPORT TICKET #1042", fill="black")
    draw.text((14, 42), "My card was charged twice", fill="black")
    draw.text((14, 66), "this month. Please refund.", fill="black")
    draw.text((14, 100), "submitted: 2026-09-28", fill="gray")
    paths["ticket"] = str(root / "ticket.png")
    ticket.save(paths["ticket"])

    receipt = Image.new("RGB", (300, 220), "white")
    draw = ImageDraw.Draw(receipt)
    draw.rectangle((0, 0, 299, 219), outline="black", width=2)
    draw.text((60, 12), "CORNER CAFE", fill="black")
    lines = ["2026-09-28", "", "Espresso        4.50", "Latte           6.00",
             "Croissant       3.50", "", "TOTAL          14.00"]
    for offset, line in enumerate(lines):
        draw.text((18, 42 + offset * 20), line, fill="black")
    paths["receipt"] = str(root / "receipt.png")
    receipt.save(paths["receipt"])

    chart = Image.new("RGB", (340, 200), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((10, 8), "Quarterly revenue ($k)", fill="black")
    bars = {"Q1": 40, "Q2": 60, "Q3": 50, "Q4": 70}
    for index, (quarter, value) in enumerate(bars.items()):
        x0 = 40 + index * 75
        height = value
        draw.rectangle((x0, 170 - height, x0 + 45, 170), fill="steelblue")
        draw.text((x0 + 8, 174), quarter, fill="black")
        draw.text((x0 + 4, 160 - height), str(value), fill="black")
    paths["chart"] = str(root / "chart.png")
    chart.save(paths["chart"])
    return paths


SAMPLE_QUESTIONS = {
    "ticket": {
        "team": {"type": "choice",
                 "instructions": "Which team should handle the ticket "
                                 "shown in the image?",
                 "criteria": {"billing": "Payments, invoices and refunds",
                              "technical": "Bugs, errors and outages",
                              "account": "Login, profile and settings"}},
        "refund": {"type": "noul",
                   "instructions": "Is the customer in the image asking "
                                   "for a refund?"},
        "urgency": {"type": "score",
                    "instructions": "How urgent is the issue in the image?",
                    "criteria": ["could wait a few days",
                                 "should be fixed soon",
                                 "needs immediate action"]},
    },
    "receipt": {
        "venue": {"type": "choice",
                  "instructions": "What kind of venue issued the receipt "
                                  "in the image?",
                  "criteria": {"coffee shop": "Cafe, espresso, pastries",
                               "grocery": "Supermarket food shopping",
                               "pharmacy": "Medicines and health items"}},
        "over20": {"type": "noul",
                   "instructions": "Does the receipt's TOTAL exceed "
                                   "$20?"},
        "value": {"type": "score",
                  "instructions": "How expensive was this purchase?",
                  "criteria": ["a few dollars", "a modest amount",
                               "a large amount"]},
    },
    "chart": {
        "best": {"type": "choice",
                 "instructions": "Which quarter had the highest revenue "
                                 "in the chart image?",
                 "criteria": {"Q1": "First quarter", "Q2": "Second quarter",
                              "Q3": "Third quarter", "Q4": "Fourth quarter"}},
        "q2vq3": {"type": "noul",
                  "instructions": "Did Q2 revenue exceed Q3 revenue in "
                                  "the chart?"},
        "growth": {"type": "score",
                   "instructions": "How strong was revenue growth across "
                                   "the year in the chart?",
                   "criteria": ["flat or shrinking", "modest growth",
                                "strong growth"]},
    },
}

SAMPLE_STATES = {
    "ticket": "A customer photographed their support ticket.",
    "receipt": "A photographed cafe receipt.",
    "chart": "A company's quarterly revenue chart.",
}


def sample_request(name: str) -> dict:
    """predict() request for one sample image: the image path, a short
    state line and its three typed questions."""
    return {"state": SAMPLE_STATES[name],
            "images": [build_samples()[name]],
            "questions": SAMPLE_QUESTIONS[name]}


# ------------------------------------------------------------- serve
def build_question(question_type: str, instructions: str, criteria):
    """Serve payload pieces -> one predict() question dict. ValueError
    names the problem; serve() turns it into the error JSON."""
    question_type = str(question_type or "").strip().lower()
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question type must be one of "
                         f"{', '.join(QUESTION_TYPES)}")
    if not str(instructions or "").strip():
        raise ValueError("payload needs an 'instructions' question")
    question = {"type": question_type,
                "instructions": str(instructions).strip()}
    if question_type == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice questions need a 'criteria' object "
                             "of at least two label -> description pairs")
        question["criteria"] = {str(k): str(v) for k, v in criteria.items()}
    elif question_type == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score questions need a 'criteria' list of "
                             "at least two rubric levels, low to high")
        question["criteria"] = [str(level) for level in criteria]
    return question


def parse_serve_payload(payload: dict) -> tuple[list[str], str, dict]:
    """stdin payload -> (image paths, state, predict() request dict
    minus images/state). ValueError names the problem."""
    images = [str(p).strip() for p in
              ([payload.get("image")] if payload.get("image")
               else payload.get("images", [])) if str(p).strip()]
    if not images:
        raise ValueError("payload needs an 'image' path or an 'images' "
                         "list (1-8 local image files)")
    if len(images) > 8:
        raise ValueError("at most 8 images per request")
    for path in images:
        if not Path(path).is_file():
            raise ValueError(f"image file not found: {path}")
    state = str(payload.get("state") or "").strip() or "An uploaded image."
    question = build_question(payload.get("type"),
                              payload.get("instructions"),
                              payload.get("criteria"))
    return images, state, question


def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        images, state, question = parse_serve_payload(payload)
        engine = load(payload.get("model"))
        result = engine.predict({"state": state, "images": images,
                                 "questions": {"q": question}})
        answer = result["answers"]["q"]
        row = {"answer": {key: answer[key] for key in
                          ("choice", "noul", "score", "probabilities",
                           "confidence") if key in answer},
               "usage": result.get("usage"),
               "timing": result.get("timing")}
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps(row))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------- tour
def tour(model_key=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    size = normalize_size(model_key)
    print(f"Loading {SIZES[size]} (multimodal path: vision tower + "
          f"projector from {size_home(size)})...")
    engine = load(size)

    for name in ("ticket", "receipt", "chart"):
        banner(f"{name.capitalize()} - one image, three typed questions, "
               "one predict()")
        request = sample_request(name)
        print(f"  image: {request['images'][0]}")
        answers = timed(engine.predict, request)
        for field, answer in answers["answers"].items():
            if answer["type"] == "choice":
                print(f"  {field:<8} -> {answer['choice']} "
                      f"(confidence {answer['confidence']:.2f})")
            elif answer["type"] == "noul":
                print(f" {field:<8} -> yes-probability "
                      f"{answer['noul']:.2f}")
            else:
                print(f"  {field:<8} -> score {answer['score']:.2f} "
                      f"over {len(answer['probabilities'])} rubric levels")

    print("\nDone. The text-only path: .venv-von/Scripts/python "
          "demos/intern_decision_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    model_key = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("--model needs a value; known: "
                             f"{', '.join(SIZES)}")
        model_key = sys.argv[idx + 1]
    tour(model_key)


if __name__ == "__main__":
    main()
