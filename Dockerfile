# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages that some pip wheels need to build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py ./
COPY evals/ ./evals/

# Sensible defaults; override at runtime if needed
ENV DUCKDB_PATH=data/duckdb/olist.duckdb \
    GROQ_MODEL=openai/gpt-oss-120b

# 8501 = Streamlit, 6006 = Phoenix UI, 4317 = Phoenix OTLP collector
EXPOSE 8501 6006 4317

# Default: launch the Streamlit chat UI.
# Override the CMD to run the CLIs, e.g.:
#   docker run --rm insight-agent python -m src.agent_cli "How many orders in 2018?"
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
