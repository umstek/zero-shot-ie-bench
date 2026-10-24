"""runner-protocol — shared library for every runner island.

Exposes the FastAPI app factory, the host fingerprint and the generated
wire-contract models. Runners import this as a path dependency; runners never
import each other.
"""

from .app import create_app
from .host import host_fingerprint
from .schema_models import (
    Gpu,
    Host,
    RunnerInfo,
    ServedModel,
    TrackItem,
    TrackRequest,
    TrackResponse,
    TrackResult,
)

__all__ = [
    "Gpu",
    "Host",
    "RunnerInfo",
    "ServedModel",
    "TrackItem",
    "TrackRequest",
    "TrackResponse",
    "TrackResult",
    "create_app",
    "host_fingerprint",
]
