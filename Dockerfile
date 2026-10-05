# ==========================================
# Stage 1: Build & Dependency Resolution
# ==========================================
FROM python:3.12-slim AS builder

# Install uv for fast, deterministic dependency builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency specifications first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv without project source code first
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code and install project into venv
COPY context_guard ./context_guard
COPY README.md ./
RUN uv sync --frozen --no-dev

# ==========================================
# Stage 2: Minimal Production Runtime
# ==========================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Create unprivileged non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Copy virtual environment and project from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/context_guard /app/context_guard
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/README.md /app/README.md

# Switch to unprivileged user
USER appuser

# Expose default reverse-proxy port
EXPOSE 8080

# Run reverse proxy with Uvicorn
CMD ["uvicorn", "context_guard.proxy.server:app", "--host", "0.0.0.0", "--port", "8080"]
