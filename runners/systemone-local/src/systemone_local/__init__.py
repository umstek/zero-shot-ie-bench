"""systemone-local — the local System One-contract decision engines as runner models.

Serves four catalog deployments as runner models on one FastAPI app
(runner-protocol create_app): kev-0.8b, decider-0.8b, openthai-systemone
and agentjev-v1. Every engine loads its weights in-process, lazily, on
the first predict. Importing this package (or the server module) pulls
no heavy stack and touches no weights.
"""

from .server import create_app

__all__ = ["create_app"]
