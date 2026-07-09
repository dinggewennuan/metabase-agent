FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm

WORKDIR /app

RUN useradd --system --create-home --home-dir /home/app app \
    && mkdir -p /app/data \
    && chown -R app:app /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app src ./src
# skills/ ships with the image: AGENT_SKILLS_PATH defaults to the relative
# "skills" directory and the feature silently no-ops when it is missing.
COPY --chown=app:app skills ./skills

USER app

# Keep mutable state out of the read-only app tree; mount a volume at /app/data.
ENV AGENT_STATE_PATH=/app/data/.metabase_agent_state.json \
    AGENT_MEMORY_PATH=/app/data/.metabase_agent_memory.json

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/config', timeout=4)"]

CMD ["/app/.venv/bin/uvicorn", "metabase_agent.api.app:app", "--host", "0.0.0.0", "--port", "8765"]
