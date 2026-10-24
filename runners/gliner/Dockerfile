# gliner runner island — GLiFormer / GLiREL / GLiNER-relex / GLiClass / Laya
# on the transformers 4.57.x CPU stack (torch CPU wheels via the pytorch-cpu
# index baked into uv.lock).
#
# Build from the REPO ROOT: the island depends on runners/_protocol as a path
# dependency, which is outside the island directory, so _protocol comes in as
# a named build context:
#
#   docker build -f runners/gliner/Dockerfile \
#       --build-context protocol=runners/_protocol \
#       -t gliner-runner runners/gliner
#
# Model weights are NOT in the image — they download into the HF cache on
# first use per model; mount a volume for it:
#
#   docker run -p 8009:8009 -v hf-cache:/root/.cache/huggingface gliner-runner
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

# the interpreter is the image's python 3.13 (no managed-python downloads)
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app/runners/gliner

# dependency layers first (cached until pyproject.toml / uv.lock change):
# the path dependency's source must exist before uv sync resolves it
COPY pyproject.toml uv.lock .python-version ./
COPY --from=protocol . /app/runners/_protocol
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# the island itself
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/runners/gliner/.venv/bin:$PATH"
EXPOSE 8009
CMD ["uvicorn", "gliner_runner.server:app", "--host", "0.0.0.0", "--port", "8009"]
