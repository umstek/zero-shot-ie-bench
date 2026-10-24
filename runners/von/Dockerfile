# von runner island: wfzyx/von-1.0 on CPU (transformers 5.x, python 3.13).
#
# Build from the REPO ROOT so the runner-protocol path dependency
# (../_protocol) is inside the build context:
#   docker build -f runners/von/Dockerfile -t von-runner .
#
# The checkpoint (~0.4 GB) downloads into the container's HF cache on the
# first predict. To keep it across restarts, mount one:
#   docker run -v von-hf-cache:/root/.cache/huggingface -p 8030:8030 von-runner
# (baking weights into the image is left to the deployer — CI must never
# download model weights).

FROM python:3.13-slim

# uv: the island is a uv project; uv.lock is the reproducible environment
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /runner/von

# dependency manifests first so the (torch-heavy) layer caches
COPY runners/von/pyproject.toml runners/von/uv.lock runners/von/.python-version runners/von/README.md ./
COPY runners/_protocol /runner/_protocol
RUN uv sync --frozen --no-dev

# then the island's own code (the editable self-install refreshes)
COPY runners/von/src /runner/von/src
RUN uv sync --frozen --no-dev

EXPOSE 8030

CMD ["uv", "run", "--frozen", "--no-dev", "python", "-m", \
     "von_runner.server", "--host", "0.0.0.0", "--port", "8030"]
