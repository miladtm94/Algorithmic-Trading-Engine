FROM python:3.11-slim

LABEL maintainer="AI Trading Engine"
LABEL description="Production-grade algorithmic trading engine"

# Security: run as non-root
RUN groupadd -r trader && useradd -r -g trader -d /app trader

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and source before editable install
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e "." && \
    pip install --no-cache-dir "ccxt>=4.0.0" "python-dotenv>=1.0.0"

COPY scripts/ scripts/

# Persistent data directory
RUN mkdir -p /app/data && chown trader:trader /app/data
VOLUME ["/app/data"]

USER trader

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Default: paper mode; override via docker-compose or env
ENV TRADING_MODE=paper
ENV LOG_LEVEL=INFO

ENTRYPOINT ["python", "-m", "ai_trading_engine"]
