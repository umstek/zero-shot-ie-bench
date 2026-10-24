"""Shared base for the island's engine modules.

Each engine module exposes the same surface the registry dispatches on:

    MODEL_ID / VERSION / RUNTIME  — the catalog deployment's identity
    available()                   — True when this process can actually load
                                    the weights (env-var homes + HF cache;
                                    no network, so tests skip on absence)
    predict(payload)              — one runner decision item: the payload
                                    {input, question} -> {prediction, prob}

Weights load lazily inside predict (first call pays the load); module
import is stdlib-only everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# runners/systemone-local <- runners <- <repo root>; this file sits in
# models/, so parents[3] is the island and parents[5] the repo.
ISLAND_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = ISLAND_DIR.parents[1]

# The engines came from three places, none of them PyPI-installable as one
# package: the kev and agent-jev clones sit next to this repo (they served
# the historical 8009/8018/8029/8149 ports), and the decider / OpenThai
# weights come from the HF hub. Every location is env-overridable, in the
# D1_HOME / DECISION2_HOME style legacy/engines uses.
DEFAULT_KEV_HOME = REPO_DIR.parent / "kev"
DEFAULT_AGENTJEV_HOME = REPO_DIR.parent / "agent-jev"


def env_home(name: str, default: Path) -> Path:
    """The engine clone's location: <NAME>_HOME or the sibling-clone default."""
    return Path(os.environ.get(name, default))


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


def pick_path(repo_id: str, weights_env: str) -> str:
    """Weights location: an explicit <WEIGHTS>_env dir, else the HF-cache
    snapshot, else the hub id (the loader then downloads on first use —
    live serving only; available() never reports True on the miss)."""
    explicit = os.environ.get(weights_env)
    if explicit:
        return explicit
    snapshot = hf_snapshot(repo_id)
    return str(snapshot) if snapshot else repo_id
