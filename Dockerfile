FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install specific compatible versions
RUN pip install --no-cache-dir \
    "transformers==4.35.2" \
    runpod \
    accelerate

# Copy handler
COPY handler.py .

# Environment variables - Use a model that works with older transformers
ENV MODEL_NAME="microsoft/DialoGPT-medium"
ENV PYTHONPATH="/app"

CMD ["python", "handler.py"]
