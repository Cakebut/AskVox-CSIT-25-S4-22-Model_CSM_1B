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
# Utilities
# -------------------------------
def split_text_into_chunks(text, max_words=50):
    """Split text into chunks of up to max_words each."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def generate_audio(text: str) -> bytes:
    """Generate audio for a single chunk of text."""
    text = f"[0]{text}"  # Use default speaker 0
    inputs = processor(text, add_special_tokens=True).to(device)

    with torch.no_grad():
        audio_tensor = model.generate(**inputs, output_audio=True)

    waveform = audio_tensor[0].cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

def generate_audio_chunks(text: str):
    """Split long text and generate audio for each chunk."""
    chunks = split_text_into_chunks(text, max_words=50)
    audio_bytes_list = []

    for chunk in chunks:
        audio_bytes = generate_audio(chunk)
        audio_bytes_list.append(audio_bytes)

    return audio_bytes_list

# -------------------------------
# RunPod serverless handler
# -------------------------------
def handler(job):
    text = job.get("input", {}).get("text")
    if not text:
        return {"error": "Missing input.text"}

    try:
        audio_chunks = generate_audio_chunks(text)

        # Convert each chunk to base64
        audio_b64_list = [base64.b64encode(chunk).decode() for chunk in audio_chunks]

        return {
            "audio_base64_list": audio_b64_list,
            "format": "wav",
            "sample_rate": 24000,
            "chunk_count": len(audio_b64_list)
        }
    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

# -------------------------------
# Start RunPod worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
