FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install packages directly (no requirements.txt)
RUN pip install --no-cache-dir \
    runpod==1.5.1 \
    transformers==4.35.2 \
    accelerate==0.24.1 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    numpy==1.24.3 \
    huggingface-hub==0.19.4

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
