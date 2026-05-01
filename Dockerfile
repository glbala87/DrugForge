FROM python:3.11-slim

WORKDIR /app

# System dependencies for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy application code
COPY *.py ./
COPY config.py model.py modules.py tokenizer.py dataset.py metrics.py task_balancer.py infer.py api.py ./

# Default: serve the API
ENV DRUGFORGE_MODEL=/app/models/drugforge_davis.pth
ENV DRUGFORGE_TOKENIZER=/app/models/davis_tokenizer.json
ENV DRUGFORGE_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
