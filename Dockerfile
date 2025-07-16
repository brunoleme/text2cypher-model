FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# Install uv
RUN pip install --upgrade pip && \
    pip install uv

# Copy project files
COPY . .

# Install project dependencies in editable mode
RUN uv pip install --system -e .

# Optional: install extra tools if needed (e.g., testing, linting)
# RUN uv pip install pytest ruff black mypy

# Set environment variable for Python module resolution
ENV PYTHONPATH=/app

# Default entrypoint for training (can override with CMD or docker run ...)
CMD ["python", "train.py", "--config-path=src/text2cypher/finetuning/config", "--config-name=config.prod"]
