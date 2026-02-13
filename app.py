import os
import io
import base64
import subprocess
import runpod
import torch
from transformers import AutoProcessor, AutoModel
from huggingface_hub import login

# =====================
# HuggingFace Login
# =====================
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)

MODEL_ID = os.environ.get("MODEL_ID", "cakebut/askvoxcsm-1b")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model:", MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

model.eval()
print("Model loaded.")

# =====================
# WAV -> MP3
# =====================
def wav_to_mp3(wav_bytes):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "quiet",
            "-f", "wav",
            "-i", "pipe:0",
            "-vn",
            "-ar", "24000",
            "-ac", "1",
            "-b:a", "64k",
            "-f", "mp3",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    mp3_bytes, _ = process.communicate(wav_bytes)
    return mp3_bytes

# =====================
# Generate audio
# =====================
@torch.inference_mode()
def generate_audio(text: str):
    inputs = processor(text=text, return_tensors="pt").to(device)

    audio = model.generate(**inputs)

    audio_array = audio.cpu().numpy()[0]

    # Save WAV to memory
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 24000, format="WAV")
    wav_bytes = buffer.getvalue()

    # Convert to MP3
    mp3_bytes = wav_to_mp3(wav_bytes)

    return mp3_bytes

# =====================
# RunPod handler
# =====================
def handler(event):
    try:
        text = event["input"].get("text", "")

        if not text:
            return {"error": "No text provided"}

        print(f"TTS request len={len(text)}")

        mp3_bytes = generate_audio(text)

        audio_base64 = base64.b64encode(mp3_bytes).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "format": "mp3"
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

# =====================
# Start worker
# =====================
runpod.serverless.start({"handler": handler})
