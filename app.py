import os
import base64
import io
import runpod
import torch
import soundfile as sf
from transformers import AutoModelForTextToWaveform

MODEL_ID = os.getenv("MODEL_ID")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CSM-1B on", device)

model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()

print("Model loaded")


# -----------------------
# TTS generation
# -----------------------
def generate_audio(text):
    with torch.no_grad():
        audio = model.generate(text=text)

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=22050, format="WAV")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()


# -----------------------
# RunPod handler
# -----------------------
def handler(job):
    inp = job.get("input", {})
    text = inp.get("text")

    if not text:
        return {"error": "Missing input.text"}

    if not isinstance(text, str):
        return {"error": "input.text must be a string"}

    try:
        audio_b64 = generate_audio(text)

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 22050
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------
# Start serverless
# -----------------------
runpod.serverless.start({"handler": handler})
