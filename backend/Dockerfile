FROM python:3.13-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY libs/ ./libs/
COPY backend/ ./backend/

# Install dependencies using uv
RUN uv sync --frozen

CMD ["uv", "run", "--package", "backend", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9200", "--reload"]