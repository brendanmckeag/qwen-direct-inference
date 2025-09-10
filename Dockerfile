# Use Python 3.9 slim image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for PyTorch and transformers
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY handler.py .

# Environment variables
ENV MODEL_NAME="Qwen/Qwen3-1.7B"
ENV PYTHONPATH="/app"

# Command to run when the container starts
CMD [ "python", "-u", "handler.py" ]
