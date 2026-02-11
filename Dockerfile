# -------------------------------
# Base image with PyTorch + CUDA
# -------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set HF_TOKEN if private repo
# ENV HF_TOKEN=hf_xxxxx

# Download CSM-1B model from Hugging Face
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='cakebut/askvoxcsm-1b', subfolder='csm-1b', local_dir='./csm-1b', token=None)"

# Copy app code
COPY app.py .

# Run the app
CMD ["python", "app.py"]
