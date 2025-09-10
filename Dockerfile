FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install flash-attn first with specific flags
RUN pip install flash-attn>=2.4.0 --no-build-isolation

COPY requirements.txt .

# Install remaining requirements (remove flash-attn from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV MODEL_NAME="Qwen/Qwen3-32B"       
ENV MAX_LENGTH="4096"               
ENV TEMPERATURE="0.7"                 
ENV TOP_P="0.9"                    
ENV TOP_K="50"                       
ENV PYTHONPATH="/app"                 

EXPOSE 8000
CMD ["python", "handler.py"]
