# relik runner island: relik-ie ReLiK on CPU (relik 1.0.7, torch 2.3.1,
# python 3.11 — the only 3.11 island: relik pins torch==2.3.1, which has
# no 3.13 wheels).
#
# Build from the REPO ROOT so the runner-protocol path dependency
# (../_protocol) is inside the build context:
#   docker build -f runners/relik/Dockerfile -t relik-runner .
#
# The checkpoints (~700 MB relation / ~586 MB sibling reader) download
# into the container's HF cache on the first predict. To keep them across
# restarts, mount one:
#   docker run -v relik-hf-cache:/root/.cache/huggingface -p 8031:8031 relik-runner
# (baking weights into the image is left to the deployer — CI must never
# download model weights).

FROM python:3.11-slim

# uv: the island is a uv project; uv.lock is the reproducible environment
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /runner/relik

# dependency manifests first so the (torch-heavy) layer caches
COPY runners/relik/pyproject.toml runners/relik/uv.lock runners/relik/.python-version runners/relik/README.md ./
COPY runners/_protocol /runner/_protocol
RUN uv sync --frozen --no-dev

# then the island's own code (the editable self-install refreshes)
COPY runners/relik/src /runner/relik/src
RUN uv sync --frozen --no-dev

EXPOSE 8031

CMD ["uv", "run", "--frozen", "--no-dev", "python", "-m", \
     "relik_runner.server", "--host", "0.0.0.0", "--port", "8031"]
