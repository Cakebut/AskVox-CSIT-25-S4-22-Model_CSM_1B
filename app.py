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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Voice safety limits (important for latency)
MAX_TEXT_CHARS = 600          # prevents very long TTS requests
MAX_NEW_TOKENS = 1500         # ~6–10 seconds of audio

print(f"Loading model: {MODEL_ID}")
print(f"Device: {DEVICE}")

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("Model loaded successfully!")
print("Worker ready.")

# -------------------------------
# Audio Generation
# -------------------------------
def generate_audio(text: str) -> bytes:
    # Limit text length to prevent long generation / timeouts
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    # Speaker 0
    text = f"[0]{text}"

    # Tokenize
    inputs = processor(text, add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate audio
    with torch.no_grad():
        audio_tensor = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,      # faster + stable
        )

    waveform = audio_tensor[0].cpu().numpy()

    # Convert to WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    return buffer.read()

# -------------------------------
# RunPod Handler
# -------------------------------
def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text")

    if not text:
        return {"error": "Missing input.text"}

    try:
        print(f"[TTS] Received text length: {len(text)}")

        audio_bytes = generate_audio(text)
        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000
        }

    except Exception as e:
        print("[TTS ERROR]", str(e))
        return {"error": str(e)}

# -------------------------------
# Start RunPod worker
# -------------------------------
runpod.serverless.start({"handler": handler})
