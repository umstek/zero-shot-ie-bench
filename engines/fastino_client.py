"""Minimal client for Fastino's hosted GLiDE decision model.

Wire format is the TypeSafe System One shape (the one engines/jev_client.py
implements) behind Fastino's REST: POST {model, state, questions} to
https://api.fastino.ai/v1/systemone with a Bearer key, and the payload
arrives flat ({model, answers, usage}) - no envelope to unwrap. Questions
and answers are Jev-compatible; choice criteria descriptions may be null.

GLiDE (Generalized Lightweight Decision Engine, released Sep 2026) is a
"thinking decision model": one fast pass, then extra reasoning tokens when
the leading option is uncertain. usage.output_tokens counts those thinking
tokens, but they are priced at $0 - only input bills, at $0.15/M (Fastino's
pricing page; GET /v1/base-models agrees). 40k context, served on one B200,
and the model catalog flags it ZDR. The response usage block reports tokens
but no $, so cost charts derive GLiDE's cost as measured input tokens x
that price (app.derived_cost) - the same rule the README's cost table
applies to the Clef pair and Jev.

Fastino's hosted GLiNER extraction models are the same open-weight
checkpoints this repo already benches locally through the gliner2 package
(fastino/gliner2.5-*-v1, fastino/GLiNER-2.5-Decide - see bench.py and
bench_spectrum.py), so only GLiDE is wired here.

API key: set the FASTINO_API_KEY environment variable, or put
`FASTINO_API_KEY=...` in a `.env` file in the repo root (gitignored).
Paid API; each ask() call is a request.

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import os

# body "model" selector -> endpoint model id (lowercase fastino/glide is
# accepted too; the catalog's canonical id is used)
MODELS = {"glide": "fastino/GLiDE"}
SYSTEM_ONE_URL = "https://api.fastino.ai/v1/systemone"
# Fastino's published input price, $ per 1M input tokens; output (thinking)
# tokens are $0, so input pricing is the whole cost
INPUT_USD_PER_MTOK = {"glide": 0.15}


def load_api_key() -> str:
    """FASTINO_API_KEY from the environment, else from a local .env file."""
    env = os.environ.get("FASTINO_API_KEY")
    if env:
        return env.strip()
    # .env lives in the repo root, one level above engines/
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("FASTINO_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def glide(usage_sink=None):
    """A JevClient pointed at Fastino's hosted GLiDE."""
    from engines.jev_client import JevClient

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "No FASTINO_API_KEY found. Set the environment variable or "
            "create a .env file in the repo root with FASTINO_API_KEY=... "
            "(see README)."
        )
    return JevClient(api_key=api_key, model=MODELS["glide"],
                     base_url=SYSTEM_ONE_URL, usage_sink=usage_sink)
