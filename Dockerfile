FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies (without flash-attn first)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Try to install flash-attn separately (optional)
RUN pip install flash-attn>=2.3.0 || echo "Flash-attention installation failed, continuing without it"

# Copy handler
COPY handler.py .

# Environment variables for Qwen3-8B
ENV MODEL_NAME="Qwen/Qwen3-8B"
ENV MAX_NEW_TOKENS="2048"
ENV TEMPERATURE="0.7"
ENV TOP_P="0.9"
ENV TOP_K="50"
ENV ENABLE_THINKING="true"
ENV PYTHONPATH="/app"

# Set CUDA memory fraction to avoid OOM
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

CMD ["python", "handler.py"]
