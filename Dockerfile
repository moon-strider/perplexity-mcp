FROM ghcr.io/astral-sh/uv:0.12.11 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --python /usr/local/bin/python \
    && useradd --uid 10001 --create-home app
USER 10001:10001
ENTRYPOINT ["/app/.venv/bin/perplexity-mcp-ultra"]
