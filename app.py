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
MAX_CHARS = 350   # safe size for serverless

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
# Text splitter
# -------------------------------
def split_text(text, max_chars=MAX_CHARS):
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks

# -------------------------------
# Generate audio for one chunk
# -------------------------------
def generate_audio_chunk(text):
    text = f"[0]{text}"
    inputs = processor(text, add_special_tokens=True).to(device)

    with torch.no_grad():
        audio_tensor = model.generate(**inputs, output_audio=True)

    waveform = audio_tensor[0].cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()

# -------------------------------
# RunPod handler
# -------------------------------
def handler(job):
    text = job.get("input", {}).get("text")
    if not text:
        return {"error": "Missing input.text"}

    try:
        chunks = split_text(text)
        print(f"Text split into {len(chunks)} chunks")

        audio_list = []
        for chunk in chunks:
            audio_b64 = generate_audio_chunk(chunk)
            audio_list.append(audio_b64)

        return {
            "audio_base64_list": audio_list,
            "format": "wav",
            "sample_rate": 24000
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

# -------------------------------
# Start worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
