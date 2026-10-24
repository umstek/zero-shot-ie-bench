"""Test fixtures for the gliner island.

runner_protocol import fallback: the island depends on runners/_protocol as
an editable path dependency, so `uv sync` installs it. If the import still
fails (tests run without uv sync), put the protocol's REAL source on
sys.path — the repo always ships ../_protocol/src next door. No stub is
involved: the shared protocol code is what the tests exercise either way.

Model weights are never downloaded: the heavy tests guard themselves
through tests/hf_weights.py (HF hub cache presence) and skip otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import runner_protocol  # noqa: F401
except ImportError:  # pragma: no cover — only without uv sync
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_protocol" / "src"))
