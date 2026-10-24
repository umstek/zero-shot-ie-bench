"""Test fixtures for the relik runner island.

runner_protocol import fallback: the island depends on runners/_protocol as
an editable path dependency, so `uv sync` installs it. If the import still
fails (tests run without uv sync), put the protocol's REAL source on
sys.path — the repo always ships ../_protocol/src next door. No stub is
involved: the shared protocol code is what the tests exercise either way.

The heavy relik stack is never imported for real except in the slow tests:
the offline tests install a fake `relik` package replicating the two
1.0.7 defects the engine patches (tests/fake_relik.py), and the FastAPI
contract is exercised over stubbed pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import runner_protocol  # noqa: F401
except ImportError:  # pragma: no cover — only without uv sync
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_protocol" / "src"))

import pytest

import relik_runner.relik as relik_module


@pytest.fixture(autouse=True)
def _no_shared_pipelines():
    """Keep the module-level pipeline cache empty around every test:
    nothing may leak a loaded (or stubbed) pipeline into another test."""
    relik_module.reset_pipelines()
    yield
    relik_module.reset_pipelines()
