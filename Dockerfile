# =============================================================================
# BASE IMAGE: Start with RunPod's optimized PyTorch image
# =============================================================================
# This image comes pre-installed with:
# - PyTorch 2.1.0 (deep learning framework)
# - Python 3.10 (programming language)
# - CUDA 12.1.1 (NVIDIA GPU computing platform)
# - Ubuntu 22.04 (Linux operating system)
# - Development tools for compiling packages
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# =============================================================================
# WORKING DIRECTORY: Set where our app files will live inside the container
# =============================================================================
WORKDIR /app

# =============================================================================
# SYSTEM DEPENDENCIES: Install additional system-level packages we need
# =============================================================================
# Update package list and install essential tools:
# - git: For cloning repositories (some Python packages need this)
# - wget: For downloading files
# - curl: For making HTTP requests
# Clean up package cache afterward to reduce image size
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# PYTHON DEPENDENCIES: Install our Python packages
# =============================================================================
# Copy requirements.txt first (before copying code) for better Docker caching
# If requirements.txt doesn't change, Docker can reuse this layer
COPY requirements.txt .

# Install Python packages:
# --no-cache-dir: Don't store package cache (saves space)
# -r requirements.txt: Install from requirements file
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# APPLICATION CODE: Copy our inference script
# =============================================================================
# Copy the main handler.py file into the container
COPY handler.py .

# =============================================================================
# ENVIRONMENT VARIABLES: Set default configuration values
# =============================================================================
# These can be overridden when deploying to RunPod
# They provide sensible defaults for the model behavior

ENV MODEL_NAME="Qwen/Qwen3-32B"        # Which AI model to load
ENV MAX_LENGTH="4096"                  # Maximum tokens to generate per request
ENV TEMPERATURE="0.7"                  # Default randomness level (0-1)
ENV TOP_P="0.9"                       # Nucleus sampling parameter (0-1)
ENV TOP_K="50"                        # Top-k sampling parameter
ENV PYTHONPATH="/app"                  # Ensure Python can find our modules

# =============================================================================
# NETWORK: Expose the port our application will use
# =============================================================================
# RunPod serverless functions typically use port 8000
# This tells Docker that our app will listen on this port
EXPOSE 8000

# =============================================================================
# STARTUP COMMAND: What to run when the container starts
# =============================================================================
# This runs our Python script which starts the serverless endpoint
CMD ["python", "handler.py"]
