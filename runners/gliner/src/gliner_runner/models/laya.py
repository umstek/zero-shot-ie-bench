"""Laya — Convai Innovations' typed-decision ModernBERT (the local Jev).

Serves the three catalog deployments (catalog/models/convaiinnovations/*.yaml,
all `runner: gliner`):
  - laya.local                    -> convaiinnovations/laya (421M English root,
                                     benched as "Laya (local)")
  - laya.router                   -> the laya.Router mode: the multilingual pool
                                     picks the checkpoint per input (mmBERT
                                     English root vs typed-decisions), a v0 row
                                     of its own
  - laya-typed-decisions.local    -> convaiinnovations/laya-typed-decisions

Track: decision — Laya answers the Jev typed questions (choice / score /
noul, the `engines/jev_client.choice/score/noul` dict shape) over a state in
one forward pass. The call paths are the bench branches verbatim:
`laya.load(repo)` / `laya.Router(device="cpu")` then `predict(state,
{"q": question})`, the answer read off `out["answers"]["q"]`.

Answer -> wire mapping (v0 recorded the primitive, the response schema has a
separate `prob` column):
  - choice -> prediction = answer["choice"],  prob = answer["confidence"]
  - score  -> prediction = answer["score"],   prob = None
  - noul   -> prediction = answer["noul"],    prob = None

The Router's per-input routing decision (out["routing"]["model"]) has no
slot in the wire response's per-item shape (the response-level `usage` is
protocol-owned and always None in the current protocol) — v0 recorded it in
its results entries; a protocol slot would be needed to surface it again.

States: a string input becomes {"text": input} (run_laya_typed's shape); a
dict input passes through as the state (demo_laya's {"excuses": [...]}).
"""

from __future__ import annotations

from typing import Any

# Weights live here after the first load; tests stub entries in this dict
# instead of touching the network. Never populated at import time.
_LOADED: dict[str, Any] = {}
_ROUTER: list[Any] = []  # the Router is a process-wide singleton


def load(repo: str) -> Any:
    """Load one Laya agent checkpoint (laya.load). First use downloads the
    checkpoint into the HF cache (~808 MB once) — never at import, never in
    tests without weights."""
    if repo not in _LOADED:
        import laya

        _LOADED[repo] = laya.load(repo)
    return _LOADED[repo]


def load_router(repo: str = "") -> Any:
    """Build the laya.Router (mmBERT routing between the English root and
    the typed-decisions checkpoints). The repo argument is unused by the
    Router (it routes across checkpoints); the registry passes the model's
    hf_repo for signature parity with the other loaders."""
    if not _ROUTER:
        from laya import Router

        _ROUTER.append(Router(device="cpu"))
    return _ROUTER[0]


def run_decision(model: Any, input_: Any, question: dict[str, Any],
                 params: dict[str, Any]) -> tuple[Any, float | None]:
    """Answer one typed question (choice/score/noul) over one state."""
    qtype = question.get("type")
    if qtype not in ("choice", "score", "noul"):
        raise ValueError("laya expects a Jev typed question: 'type' in "
                         "{'choice', 'score', 'noul'}")
    state = input_ if isinstance(input_, dict) else {"text": input_}
    out = model.predict(state, {"q": question})
    answer = out.get("answers", {}).get("q", {})
    if qtype == "choice":
        return answer.get("choice"), answer.get("confidence")
    if qtype == "score":
        return answer.get("score"), None
    return answer.get("noul"), None
