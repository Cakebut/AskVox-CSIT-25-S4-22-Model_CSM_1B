# Base image with CUDA + PyTorch
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade huggingface_hub

# Download public CSM-1B model from Hugging Face
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='cakebut/askvoxcsm-1b', subfolder='csm-1b', local_dir='./csm-1b')"

# Copy app code
COPY app.py .

# Run server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
