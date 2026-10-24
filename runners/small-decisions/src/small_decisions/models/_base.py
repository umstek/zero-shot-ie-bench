"""Shared base for the island's engine modules.

Each engine module exposes the same surface the registry dispatches on:

    MODEL_ID / VERSION / RUNTIME    — the catalog deployment's identity
                                      (multi-size engines carry ENTRIES +
                                      MODEL_IDS instead, one entry per
                                      catalog model)
    available()                     — True when this process can actually
                                      load the weights (env-var homes +
                                      HF cache; no network, so tests
                                      skip on absence)
    predict(payload)                — one runner decision item: the
                                      payload {input, question} ->
                                      {prediction, prob}

Weights load lazily inside predict (first call pays the load); module
import is stdlib-only everywhere. Ported from the systemone-local island's
models/_base.py (runners never import each other) — the env-var homes are
the legacy engines' own (D1_HOME / DECISION2_HOME / D3_HOME / K2TYPE_HOME /
INTERN_DECISION_HOME / JULIA_HOME / OLLAYA_BASE_URL).
"""

from __future__ import annotations

import os
from pathlib import Path

# runners/small-decisions <- runners <- <repo root>; this file sits in
# models/, so parents[3] is the island and parents[1] its parent dir.
ISLAND_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = ISLAND_DIR.parents[1]

# The snapshots followed the K2-Type / Intern-Decision precedent onto C:\src
# because the repo's own D: drive is full (see the legacy clients'
# docstrings). Every location is env-overridable in the legacy style.
DEFAULT_HOME_BASE = r"C:\src"


def env_home(name: str, default: Path | str) -> Path:
    """A snapshot location: the <NAME>_HOME env var or the given default."""
    return Path(os.environ.get(name) or default)


def hf_snapshot(repo_id: str) -> Path | None:
    """The newest local HF-cache snapshot of `repo_id`, or None when absent.

    Scans the hub cache the way huggingface_hub names it
    (models--<org>--<name>/snapshots/<sha>); never touches the network, so
    availability checks stay honest on a machine that never pulled the
    weights.
    """
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    ) or str(Path.home() / ".cache" / "huggingface" / "hub")
    base = Path(cache) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not base.is_dir():
        return None
    for snapshot in sorted(base.iterdir(), reverse=True):
        if snapshot.is_dir() and any(snapshot.iterdir()):
            return snapshot
    return None
