# ─────────────────────────────────────────────────────────────
# Kinbridge-Sync — Unified container image (Phase 3a)
#
# Single image serving all three roles: client, edge-a, edge-b.
# Role-specific behaviour is controlled at runtime via docker-compose
# command / environment / port settings — not baked into the image.
#
# Base:  python:3.12-slim
# System: iproute2 (tc), iptables
# Python: fastapi, uvicorn, pydantic, pyyaml
# Source: tool_world/, client/, pilot/ (tool defs), core/ (config)
# ─────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr (required for Docker logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system packages required by Phase 3 network testbed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        iproute2 \
        iptables && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────
# Install only the packages needed to run tool_world (FastAPI) and
# client (buffer + link_monitor).  Heavy ML / stats packages are
# excluded from the image to keep it small; they are only needed
# on the host for Phase 1b / Phase 5 analysis.

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi>=0.109.0 \
        uvicorn>=0.27.0 \
        pydantic>=2.6.0 \
        pyyaml>=6.0.1

# ── Source code ──────────────────────────────────────────────
# Copy the frozen Phase 2 packages and the pilot tool definitions
# that tool_world imports at startup.

COPY tool_world/   /app/tool_world/
COPY client/       /app/client/
COPY pilot/tools.py /app/pilot/tools.py
COPY pilot/__init__.py /app/pilot/__init__.py
COPY core/         /app/core/
COPY config/       /app/config/

# Ensure Python can import the packages from /app
ENV PYTHONPATH=/app

# ── Runtime ──────────────────────────────────────────────────
# No CMD — the docker-compose.yml will provide the command:
#   edge:  uvicorn tool_world.main:app --host 0.0.0.0 --port 8000
#   client: <future Phase 4/5 script>
#
# No EXPOSE — port mapping is handled by docker-compose.
# No capabilities — NET_ADMIN / NET_RAW are supplied by docker-compose.
