FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV HF_HOME=/tmp/hf_cache
ENV MODEL_ID=cakebut/askvoxcsm-1b

COPY app.py .

CMD ["python", "app.py"]
