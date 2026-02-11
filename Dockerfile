# -------------------------------
# Base image with newer PyTorch
# -------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY app.py .
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Correct model path
ENV MODEL_ID=cakebut/askvoxcsm-1b/csm-1b

# If repo is private
# ENV HF_TOKEN=hf_xxxxx

CMD ["python", "app.py"]
