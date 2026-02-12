import os
import io
import base64
import torch
import runpod
import soundfile as sf
from transformers import AutoProcessor, CsmForConditionalGeneration

# -------------------------------
# Settings
# -------------------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {MODEL_ID}")
print(f"Using device: {device}")

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully!")

# -------------------------------
# Generate audio for a single text chunk
# -------------------------------
def generate_audio(text: str) -> bytes:
    text = f"[0]{text}"  # Default speaker
    inputs = processor(text, add_special_tokens=True).to(device)

    with torch.no_grad():
        audio_tensor = model.generate(**inputs, output_audio=True)

    waveform = audio_tensor[0].cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

# -------------------------------
# Split text into manageable chunks
# -------------------------------
def chunk_text(text, max_words=200):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# -------------------------------
# RunPod serverless handler
# -------------------------------
def handler(job):
    text = job.get("input", {}).get("text")
    if not text:
        return {"error": "Missing input.text"}

    try:
        audio_base64_list = []

        for chunk in chunk_text(text, max_words=200):
            audio_bytes = generate_audio(chunk)
            audio_base64_list.append(base64.b64encode(audio_bytes).decode())

        return {
            "audio_base64_list": audio_base64_list,
            "format": "wav",
            "sample_rate": 24000
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

# -------------------------------
# Start RunPod worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
