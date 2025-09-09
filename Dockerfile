
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV MODEL_NAME="Qwen/Qwen3-32B"        # Which AI model to load
ENV MAX_LENGTH="4096"                  # Maximum tokens to generate per request
ENV TEMPERATURE="0.7"                  # Default randomness level (0-1)
ENV TOP_P="0.9"                       # Nucleus sampling parameter (0-1)
ENV TOP_K="50"                        # Top-k sampling parameter
ENV PYTHONPATH="/app"                  # Ensure Python can find our modules
EXPOSE 8000
CMD ["python", "handler.py"]
