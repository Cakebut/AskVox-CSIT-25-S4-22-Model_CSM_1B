import base64
import io
import runpod
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModel

# -----------------------
# Configuration
# -----------------------
MODEL_ID = "cakebut/askvoxcsm-1b"   # your HF repo

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[CSM] Loading model on {device}...")

# -----------------------
# Model loading (cold start)
# -----------------------
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
)

model = model.to(device)
model.eval()

if device == "cuda":
    model.half()
    torch.backends.cudnn.benchmark = True

print("[CSM] Model loaded successfully")

# -----------------------
# TTS Generation
# -----------------------
def generate_audio(text: str):
    if not text.strip():
        raise ValueError("Empty text")

    inputs = processor(
        text=text,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        audio = model.generate(**inputs)

    # Convert to numpy
    audio = audio.cpu().numpy().squeeze()

    # Save WAV to memory
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=22050, format="WAV")
    buffer.seek(0)

    # Encode base64
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return audio_base64

# -----------------------
# RunPod Handler
# -----------------------
def handler(job):
    try:
        inp = job.get("input", {})
        text = inp.get("text")

        if text is None:
            return {"error": "Missing input.text"}

        if not isinstance(text, str):
            return {"error": "input.text must be a string"}

        print(f"[CSM] Generating audio for text: {text[:60]}")

        audio_b64 = generate_audio(text)

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 22050
        }

    except Exception as e:
        print("[CSM ERROR]", str(e))
        return {"error": str(e)}

# -----------------------
# Start Serverless
# -----------------------
runpod.serverless.start({"handler": handler})
