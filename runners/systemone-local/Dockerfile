# systemone-local runner: the four local System One-contract decision engines
# on one process. CPU-only image (torch CPU index; see pyproject.toml).
# Weights are NOT baked in — mount an HF cache or let the first predict
# download them (OpenThai is gated: pass HF_TOKEN).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# uv reads [tool.uv.sources] ../_protocol — the repo layout ships with it.
COPY runners/_protocol /app/runners/_protocol
COPY runners/systemone-local /app/runners/systemone-local

# torch resolves from the pytorch-cpu index declared in pyproject.toml; no
# model weights are downloaded here.
RUN cd /app/runners/systemone-local && uv sync --frozen --no-dev

WORKDIR /app/runners/systemone-local
EXPOSE 8000

# The historical System One servers owned 8009/8018/8029 (+8149 for
# AgentJev); this island serves all four models on one port.
ENV RUNNER_HOST=0.0.0.0 RUNNER_PORT=8000
CMD ["uv", "run", "--no-dev", "python", "-m", "systemone_local.server"]
