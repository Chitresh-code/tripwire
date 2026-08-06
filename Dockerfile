FROM python:3.11-slim

# libgomp1: python:3.11-slim doesn't ship it, but LightGBM's compiled
# extension links against it at import time (OSError: libgomp.so.1 without
# this) — confirmed by an actual failed container start, not assumed.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv: this project's own dependency manager (AGENTs.md) — same tool locally and in the image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies first, so they cache separately from application code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY configs/ configs/
RUN uv sync --frozen --no-dev

# No models/registry/ here — fetched from object storage at startup
# (src/models/artifact_store.py) if configs/deployment.yaml's model_bucket
# is set via env/secret at deploy time; a stale baked-in model would mean
# redeploying just to pick up a retrain.

ENV PATH="/app/.venv/bin:$PATH"

# Runtime uses the venv's own binaries directly, not `uv run` — `uv run`
# re-checks/re-syncs against pyproject.toml on every invocation, including
# the dev dependency group, which means a network call and a multi-MB
# download (mypy, ruff, ...) on every container start. Already installed
# everything needed at build time above; no reason to touch the network again.

RUN useradd --create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
    "import httpx; httpx.get('http://localhost:8000/v1/health').raise_for_status()"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
