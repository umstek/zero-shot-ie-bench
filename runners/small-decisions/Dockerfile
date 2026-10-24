# small-decisions runner: the catch-all local decision-model island.
# CPU-only image (torch CPU index; see pyproject.toml). Weights are NOT
# baked in — mount the snapshot homes (DECISION2_HOME / D3_HOME /
# K2TYPE_HOME / INTERN_DECISION_HOME prefixes) or an HF cache, and run an
# Ollaya daemon beside the container for the ten proxied models
# (OLLAYA_BASE_URL points at it).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# uv reads [tool.uv.sources] ../_protocol — the repo layout ships with it.
COPY runners/_protocol /app/runners/_protocol
COPY runners/small-decisions /app/runners/small-decisions

# torch resolves from the pytorch-cpu index declared in pyproject.toml; no
# model weights are downloaded here.
RUN cd /app/runners/small-decisions && uv sync --frozen --no-dev

WORKDIR /app/runners/small-decisions
EXPOSE 8000

ENV RUNNER_HOST=0.0.0.0 RUNNER_PORT=8000
CMD ["uv", "run", "--no-dev", "python", "-m", "small_decisions.server"]
