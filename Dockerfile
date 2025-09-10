FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install flash-attn from pre-compiled wheel (much faster)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

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
