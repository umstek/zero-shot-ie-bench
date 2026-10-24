"""LiquidAI open d1 demo + one-shot runner for the web UI - runs in .venv-von.

Open d1 (LFM Open License v1.0) is the open-weights branch of the line
behind the hosted `liquid/d1`: pass a state (text, JSON, images - and on
d1-omni-600M a 16 kHz mono voice clip) plus named typed questions
(noul / choice / score) and every question is answered from the model's
distribution over its options in ONE forward pass, zero output tokens.

Two checkpoints, picked with --model:
    d1-3b (default)        3.12B on LFM2.5-VL-3B, 32k ctx, 17 languages
    d1-omni-600m           587M (381M LFM2.5-Encoder trunk + 94M SigLIP2
                           vision + 112M FastConformer audio), 16k ctx

The snapshots live at C:\\src\\d1-3B / C:\\src\\d1-omni-600M (D1_HOME
overrides the prefix); engines/d1_client.py loads them in-process on the
CPU (fp32) through the card's trust_remote_code AutoModel route.

    .venv-von/Scripts/python demos/d1_demo.py [--model d1-omni-600m]

Tour (interactive):
    .venv-von/Scripts/python demos/d1_demo.py

Web-UI runner (the app's d1 tab spawns this and talks JSON over
stdin/stdout):
    .venv-von/Scripts/python demos/d1_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...],
             "model"?: "d1-3b" | "d1-omni-600m"}
    stdout: {"results": [{"choice": str | None,
                          "probabilities": {label: float},
                          "confidence": float}, ...]}
            or {"error": "Type: message"}
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import subprocess
import sys
import tempfile
import time

SHARED_TEXTS = [
    "The food was cold and the waiter was rude.",
    "This is the best laptop I have ever owned.",
    "The meeting is scheduled for 3 PM in the main conference room.",
    "The flight was delayed for six hours with no explanation.",
    "She was thrilled with her exam results.",
    "Water boils at 100 degrees Celsius at sea level.",
]
QUESTION = {"sentiment": "What is the overall sentiment of this text",
            "topic": "Which topic category does this text belong to"}

# described criteria for the choice questions (same wording as bench.py's
# label dicts; descriptions are what the decision head scores)
CRITERIA = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual or mixed text without a clear attitude",
    "technology": "Software, hardware, AI, gadgets, engineering",
    "business": "Companies, markets, revenue, deals, management",
    "sports": "Athletes, matches, teams, tournaments",
    "politics": "Government, elections, policy, legislation",
}


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


def describe_labels(labels, task: str) -> dict:
    """criteria dict for arbitrary labels: the bench descriptions where
    they exist, otherwise a task-phrased one."""
    return {label: CRITERIA.get(label, f"Text whose {task} is {label}")
            for label in labels}


def choice_question(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def load(model: str | None = None):
    """d1_client.load_engine over the local snapshot; heavy imports stay
    inside so the module imports without torch (offline tests). The load
    time goes to stderr: in --serve mode stdout must stay pure JSON for
    the web UI."""
    from engines import d1_client

    t0 = time.perf_counter()
    engine = d1_client.load_engine(model)
    print(f"Loaded {engine.name} from {d1_client.model_dir(model)} in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


def question_for(task: str, text: str) -> str:
    """Instruction for one text: the question with the text restated in
    it (the house shape across the typed-decision benchmarks)."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to")
    return f'{question}: "{text}"'


def answer_row(answer: dict) -> dict:
    """predict() answer -> web-UI row; confidence is d1's own calibration
    measure, not pmax."""
    return {"choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence")}


def decide_one(engine, text: str, task: str, criteria: dict) -> dict:
    answers = engine.predict(text, {"q": choice_question(
        question_for(task, text), criteria)})
    return answer_row(answers["answers"]["q"])


def parse_serve_payload(payload: dict) -> tuple[list[str], str, list[str]]:
    """stdin payload -> (texts, task, labels). ValueError names the
    problem; serve() turns it into the error JSON."""
    texts = [str(t).strip() for t in payload.get("texts", [])
             if str(t).strip()]
    if not texts:
        raise ValueError("payload needs a non-empty 'texts' list")
    task = str(payload.get("task") or "").strip()
    if not task:
        raise ValueError("payload needs a 'task' word")
    labels = [str(l).strip() for l in payload.get("labels", [])
              if str(l).strip()]
    if not labels:
        raise ValueError("payload needs a non-empty 'labels' list")
    return texts, task, labels


# ------------------------------------------------------------------ runner
def serve(model: str | None = None) -> None:
    # the checkpoint comes from --model on the command line (the one-shot
    # form) or the stdin payload's "model" key (the web-UI form); the
    # payload wins only when --model was not given
    try:
        payload = json.loads(sys.stdin.read())
        texts, task, labels = parse_serve_payload(payload)
        engine = load(model or payload.get("model"))
        criteria = describe_labels(labels, task)
        results = [decide_one(engine, text, task, criteria)
                   for text in texts]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# ------------------------------------------------------------------ extras
def drawn_circles_image(n: int = 2):
    """A PIL-drawn image with n filled circles - the visual cousin of the
    card's two-cats photo: the pixels are the whole state, no caption."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(image)
    spots = [(96, 128, 44), (224, 96, 40), (288, 176, 36)]
    for i in range(n):
        x, y, r = spots[i]
        draw.ellipse((x - r, y - r, x + r, y + r), fill="#1f6feb")
    return image


def sapi_voice_clip(phrase: str = "I was charged twice this month "
                                  "and I would like a refund."):
    """A real 16 kHz mono speech clip via Windows SAPI TTS, as an int16
    numpy array (the input d1-omni-600M's audio= argument wants).
    RuntimeError names the failure so the tour can skip the stop."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = _os.path.join(tmp, "voice.wav")
        # SAPI defaults to 22.05 kHz; the omni tower wants 16 kHz mono, and
        # SpeechAudioFormatInfo pins the WAV to exactly that
        ps = ("Add-Type -AssemblyName System.Speech;"
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
              "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo"
              "(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
              " [System.Speech.AudioFormat.AudioChannel]::Mono);"
              "$s.SetOutputToWaveFile('" + wav + "', $f);"
              "$s.Speak('" + phrase.replace("'", "") + "');"
              "$s.Dispose()")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=120)
        if not _os.path.isfile(wav):
            raise RuntimeError(f"SAPI synthesis produced no file "
                               f"(exit {out.returncode})")
        audio, rate = sf.read(wav, dtype="int16")
        if rate != 16000:
            raise RuntimeError(f"unexpected SAPI rate {rate}")
        return audio


# -------------------------------------------------------------------- tour
def tour(model: str | None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    engine = load(model)
    print(f"\n{engine.name}: LiquidAI's open d1 - one forward pass, zero "
          "output tokens; the hosted liquid/d1 row is this line's API twin")

    # ---------------------------------------------------------------- 1
    banner("1. One state, three typed questions, a single system_one()")
    ticket = ("I was charged twice for my subscription this month and "
              "want a refund.")
    questions = {
        "team": choice_question("Which team should handle this?", {
            "billing": "Payments, invoices and refunds",
            "technical": "Bugs, errors and outages",
            "account": "Login, profile and settings"}),
        "refund": {"type": "noul",
                   "instructions": "Is the customer asking for a refund?"},
        "urgency": {"type": "score",
                    "instructions": "How urgent is this issue?",
                    "criteria": ["could wait a few days",
                                 "should be fixed soon",
                                 "needs immediate action"]},
    }
    print(f'  state: "{ticket}"  (the state is read once for every '
          "question; no generation)")
    answers = timed(engine.predict, ticket, questions)
    show(answers)
    print("  (a score answer's 'score' is the expected rubric index and "
          "its probabilities are keyed by index)")

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one predict() per text")
    criteria = describe_labels(["positive", "negative", "neutral"],
                               "sentiment")
    for text in SHARED_TEXTS:
        row = timed(decide_one, engine, text, "sentiment", criteria)
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in row["probabilities"].items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    # ---------------------------------------------------------------- 3
    banner("3. The image IS the state - two drawn circles, one question")
    image = drawn_circles_image(2)
    cats = {"type": "choice",
            "instructions": "How many circles are in this image?",
            "criteria": {"one": "One", "two": "Two",
                         "more": "Three or more"}}
    print("  (384x256 white canvas, two blue discs; state=None - the "
          "pixels carry the whole request, the card's photo-of-a-form "
          "pattern)")
    answers = timed(engine.predict, None, {"q": cats}, images=[image])
    show(answers["answers"]["q"])

    # ---------------------------------------------------------------- 4
    if engine.name == "d1-omni-600M":
        banner("4. Text + a real voice clip in one state (omni only)")
        try:
            audio = sapi_voice_clip()
            probes = {
                "money": {"type": "noul",
                          "instructions": "Does this voice note mention "
                                          "money or payments?"},
                "weather": {"type": "noul",
                            "instructions": "Is this voice note about "
                                            "the weather?"},
            }
            print(f"  (SAPI-synthesized 16 kHz mono clip, "
                  f"{len(audio) / 16000:.1f}s; the FastConformer tower "
                  "hears it in the same forward as the text - measured on "
                  "this clip, money yes / weather no)")
            answers = timed(engine.predict, "Voice note from a user.",
                            probes, audio=audio)
            show(answers["answers"])
        except RuntimeError as exc:
            print(f"  (skipped - no speech clip available: {exc})")

    # ---------------------------------------------------------------- 5
    banner(f"{5 if engine.name == 'd1-omni-600M' else 4}. "
           "system_one_batch - two tickets packed, no padding")
    packed = [("Where is my parcel? It was due Monday.", {"team": questions["team"]}),
              ("The app crashes when I open settings.", {"team": questions["team"]})]
    rows = timed(engine.predict_batch, packed)
    for (state, _), row in zip(packed, rows):
        print(f'  {row["answers"]["team"]["choice"]:<10} "{state}"')

    print("\nDone. Same texts through the other typed-decision engines:  "
          ".venv-von/Scripts/python demos/k2type_demo.py / "
          ".venv-von/Scripts/python demos/decision2_demo.py\n")


def main() -> None:
    model = None
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if "--serve" in sys.argv:
        serve(model)
        return
    tour(model)


if __name__ == "__main__":
    main()
