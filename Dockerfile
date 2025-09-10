FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Environment variables
ENV MODEL_NAME="Qwen/Qwen3-1.7B"       
ENV MAX_LENGTH="4096"               
ENV TEMPERATURE="0.7"                 
ENV TOP_P="0.9"                    
ENV TOP_K="50"                       
ENV PYTHONPATH="/app"                 

EXPOSE 8000

CMD ["python", "handler.py"]
