FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install only essential packages - let pip handle dependencies
RUN pip install transformers>=4.51.0 runpod accelerate

# Copy handler
COPY handler.py .

# Environment variables
ENV MODEL_NAME="Qwen/Qwen3-1.7B"
ENV PYTHONPATH="/app"

CMD ["python", "handler.py"]
