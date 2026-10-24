"""HF hub cache helpers for the weight-gated tests.

CI never downloads model weights: every test that would load a checkpoint
is skipped unless its snapshot is already in the local HF hub cache. The
cache directory is resolved the way huggingface_hub does it (HF_HUB_CACHE,
then HF_HOME/hub, then the default ~/.cache/huggingface/hub) without
importing huggingface_hub or touching the network.
"""

from __future__ import annotations

import os
from pathlib import Path


def hf_hub_cache() -> Path:
    """The hub cache directory huggingface_hub would use on this machine."""
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def weights_cached(repo_id: str) -> bool:
    """True when at least one non-empty snapshot of the repo is cached."""
    snapshots = hf_hub_cache() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        snapshot.is_dir() and any(snapshot.iterdir())
        for snapshot in snapshots.iterdir()
    )
