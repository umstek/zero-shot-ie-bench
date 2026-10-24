"""IFM's K2-Type-0.9B decision model, loaded in-process from a local snapshot.

Ported from legacy/engines/k2type_client.py: the weights AND the minimal
runtime (`jev/`: Encoder + collate + DecisionModel) ship inside the
IFM/K2-Type-0.9B HF repo; nothing is pip-installed — load() sys.path-inserts
the snapshot and imports its `jev` package. The upstream server (jev/serve.py)
hardcodes one CUDA GPU, so the engine runs in-process instead: DecisionModel
with fp32 weights on CPU (the house dtype; on this CPU the bf16 forward ran
~7x slower with identical predictions). The snapshot's requirements pin
transformers >=5.17,<6 — which is why this island pins 5.17.0.

K2-Type is a Jev-style typed-decision model: one state (text or JSON) plus
any number of noul/choice/score questions go in through ONE forward pass —
the state is encoded once and every question is isolated by a block-causal
attention mask; a pointer head scores each option's </opt> hidden state
against its question's <decide> hidden state, softmax at a fitted
temperature (1.478). It never generates.

to_record()/answer() re-implement the server's pure helpers because the
originals import fastapi at module level and raise HTTPException; the
layout math is copied verbatim (record label fields are placeholders the
forward never reads — only the option layout matters).

Home: K2TYPE_HOME, default C:\src\K2-Type-0.9B.

Catalog: ifm/k2-type, version k2-type-0.9b@unversioned, deployment
k2-type-0.9b.local (runtime torch/fp32/cpu).
r"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any

from . import _systemone

MODEL_ID = "ifm/k2-type"
VERSION = "k2-type-0.9b@unversioned"  # catalog/models/ifm/k2-type.yaml
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}
ENTRY: dict[str, Any] = {"model_id": MODEL_ID, "version": VERSION, "runtime": dict(RUNTIME)}

HF_REPO = "IFM/K2-Type-0.9B"

_ENGINE: list[Any] = []
_LOCK = threading.Lock()


def model_home() -> str:
    """Snapshot directory: K2TYPE_HOME, default C:\\src\\K2-Type-0.9B."""
    return os.environ.get("K2TYPE_HOME", os.path.abspath(r"C:\src\K2-Type-0.9B"))


def available() -> bool:
    """The snapshot (its jev/ runtime + decision_config.json) is present."""
    home = model_home()
    return os.path.isdir(os.path.join(home, "jev")) and os.path.isfile(
        os.path.join(home, "decision_config.json")
    )


def _ensure_runtime_on_path() -> None:
    """Put the snapshot on sys.path so `import jev` resolves (idempotent).

    Unconditional insert: a FAILED earlier load could leave a stale path
    entry, and a "not already present" guard would then import the wrong
    directory's package. Insertion at index 0 always wins."""
    home = model_home()
    if not os.path.isdir(os.path.join(home, "jev")):
        raise FileNotFoundError(
            f"no K2-Type-0.9B snapshot at {home} — download it with "
            f"`hf download {HF_REPO} --local-dir <dir>` and point "
            "K2TYPE_HOME at the directory"
        )
    while home in sys.path:
        sys.path.remove(home)
    sys.path.insert(0, home)


def _render(value: Any) -> str:
    """jev.encode.render for dict/list instructions (lazy: the module
    imports torch, so the import stays behind the path setup)."""
    _ensure_runtime_on_path()
    from jev.encode import render  # pyrefly: ignore[missing-import] — snapshot-resolved

    return render(value)


def to_record(req: dict[str, Any]) -> dict[str, Any]:
    """API request -> jev record (labels are placeholders; only the option
    layout is used). ValueError (not HTTPException) names bad questions."""
    qs: dict[str, Any] = {}
    for qid, q in req["questions"].items():
        t = q.get("type")
        instr = q.get("instructions") or ""
        instr = instr if isinstance(instr, str) else _render(instr)
        if t == "choice":
            crit = q.get("criteria") or {}
            if not 1 <= len(crit) <= 255:
                raise ValueError(f"{qid}: choice needs 1..255 options")
            crit = {k: (v if v is None or isinstance(v, str) else _render(v)) for k, v in crit.items()}
            qs[qid] = {"type": "choice", "instructions": instr, "criteria": crit, "label": next(iter(crit))}
        elif t == "score":
            levels = [c if isinstance(c, str) else _render(c) for c in (q.get("criteria") or [])]
            if not 2 <= len(levels) <= 255:
                raise ValueError(f"{qid}: score needs 2..255 levels")
            qs[qid] = {"type": "score", "instructions": instr, "criteria": levels, "label": 0}
        elif t == "noul":
            crit = q.get("criteria") or {}
            crit = {k: (v if v is None or isinstance(v, str) else _render(v)) for k, v in crit.items() if k in ("false", "true")}
            qs[qid] = {"type": "noul", "instructions": instr, "label": False, **({"criteria": crit} if crit else {})}
        else:
            raise ValueError(f"{qid}: unknown type {t!r}")
    return {"state": req["state"], "questions": qs}


def _r4(x: Any) -> float:
    return round(float(x), 4)


def answer(q: dict[str, Any], p: list[float]) -> dict[str, Any]:
    """Per-type answer row over the softmaxed option scores p."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": _r4(p[1])}
    if q["type"] == "choice":
        keys = list(q["criteria"])
        k = len(p)
        conf = 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)
        return {
            "type": "choice",
            "choice": keys[max(range(k), key=lambda i: p[i])],
            "confidence": _r4(conf),
            "probabilities": {key: _r4(v) for key, v in zip(keys, p)},
        }
    levels = len(p)
    mode = max(range(levels), key=lambda i: p[i])
    return {
        "type": "score",
        "score": _r4(sum(i * x for i, x in enumerate(p))),
        "legend": {str(i): c for i, c in enumerate(q["criteria"])},
        "probabilities": {str(i): _r4(x) for i, x in enumerate(p)},
        "confidence": _r4(1.0 - sum(x * abs(i - mode) for i, x in enumerate(p)) / (levels - 1)),
    }


class K2TypeEngine:
    """One loaded decision model; predict() answers every question of a
    request from one forward pass (no autocast on the CPU path)."""

    def __init__(self, model: Any, encoder: Any, pad_id: int, name: str, max_len: int):
        self._model = model
        self._enc = encoder
        self._pad = pad_id
        self.name = name
        self._max_len = max_len

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        import torch

        from jev.encode import collate  # pyrefly: ignore[missing-import] — snapshot-resolved

        record = to_record({"state": state, "questions": questions})
        encoded = self._enc.encode(record)
        if encoded is None or len(encoded["decide"]) != len(record["questions"]):
            raise ValueError(f"request does not fit in {self._max_len} tokens")
        with torch.no_grad():
            scores = self._model(collate([encoded], self._pad))
        answers = {
            key: answer(record["questions"][key], torch.softmax(s.float(), -1).tolist())
            for key, s in zip(encoded["qkeys"], scores)
        }
        return {"answers": answers, "model": self.name, "input_tokens": len(encoded["ids"])}


def load_engine(device: str = "cpu", dtype: str = "float32") -> K2TypeEngine:
    """K2TypeEngine over the local snapshot.

    Heavy imports stay inside (offline tests import this module freely).
    fp32 by default — the snapshot stores bf16, but the bf16 CPU forward
    buys nothing here."""
    _ensure_runtime_on_path()
    # a re-load with a different K2TYPE_HOME must not reuse the first
    # snapshot's modules; pop the package (and any stale copy) so the
    # fresh path entry wins
    for stale in ("jev", "jev.encode", "jev.model"):
        sys.modules.pop(stale, None)
    import torch

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from jev.encode import Encoder  # pyrefly: ignore[missing-import] — snapshot-resolved
    from jev.model import DecisionModel  # pyrefly: ignore[missing-import]

    home = model_home()
    with open(os.path.join(home, "decision_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    tok = AutoTokenizer.from_pretrained(home, trust_remote_code=True)
    if tok is None:  # unreachable: from_pretrained raises on a failed load
        raise ValueError("AutoTokenizer returned nothing for the K2-Type snapshot")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise ValueError("the K2-Type tokenizer ships neither a pad nor an eos token id")
    model = DecisionModel(home, dtype=getattr(torch, dtype))
    model.head.load_state_dict(load_file(os.path.join(home, "pointer_head.safetensors")))
    model.temperature.fill_(cfg["temperature"])
    model.to(device).eval()
    encoder = Encoder(tok, cfg["max_len"], cfg["max_len"] - 1024)
    return K2TypeEngine(model, encoder, pad, cfg["name"], cfg["max_len"])


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}."""
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    with _LOCK:
        if not _ENGINE:
            _ENGINE.append(load_engine())
        engine = _ENGINE[0]
    out = engine.predict(state, {"q": question})
    return _systemone.answer_to_result(question["type"], out["answers"]["q"])
