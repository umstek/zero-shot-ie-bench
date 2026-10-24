# nanodiff runner island: the 350M diffusion decision engine on CPU
# (torch main-env stack, python 3.13).
#
# Build from the REPO ROOT so the runner-protocol path dependency
# (../_protocol) is inside the build context:
#   docker build -f runners/nanodiff/Dockerfile -t nanodiff-runner .
#
# The checkpoint (~1.5 GB fp32) downloads into the container's HF cache on
# the first predict. To keep it across restarts, mount one:
#   docker run -v nanodiff-hf-cache:/root/.cache/huggingface -p 8032:8032 nanodiff-runner
# (baking weights into the image is left to the deployer — CI must never
# download model weights).

FROM python:3.13-slim

# uv: the island is a uv project; uv.lock is the reproducible environment
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /runner/nanodiff

# dependency manifests first so the (torch-heavy) layer caches
COPY runners/nanodiff/pyproject.toml runners/nanodiff/uv.lock runners/nanodiff/.python-version runners/nanodiff/README.md ./
COPY runners/_protocol /runner/_protocol
RUN uv sync --frozen --no-dev

# then the island's own code (the editable self-install refreshes)
COPY runners/nanodiff/src /runner/nanodiff/src
RUN uv sync --frozen --no-dev

EXPOSE 8032

CMD ["uv", "run", "--frozen", "--no-dev", "python", "-m", \
     "nanodiff_runner.server", "--host", "0.0.0.0", "--port", "8032"]
