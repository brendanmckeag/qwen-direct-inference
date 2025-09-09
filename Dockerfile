
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

ENV MODEL_NAME="Qwen/Qwen3-32B"       
ENV MAX_LENGTH="4096"               
ENV TEMPERATURE="0.7"                 
ENV TOP_P="0.9"                    
ENV TOP_K="50"                       
ENV PYTHONPATH="/app"                 
EXPOSE 8000
CMD ["python", "handler.py"]
