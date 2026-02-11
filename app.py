import os
import io
import base64
import torch
import soundfile as sf
import runpod

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# -------------------------------
# Settings
# -------------------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")

print("Loading model from:", MODEL_ID)

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()
print("Model loaded successfully!")

# -------------------------------
# Generate audio
# -------------------------------
def generate_audio(text):
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        audio_tokens = model.generate(**inputs)

    waveform = processor.decode(audio_tokens[0])

    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    return buffer.read()

# -------------------------------
# RunPod handler
# -------------------------------
def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text")

    if not text:
        return {"error": "Missing input.text"}

    try:
        audio_bytes = generate_audio(text)
        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "format": "wav"
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Start worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
