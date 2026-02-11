from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, List
import os
import io
import base64
import torch
import soundfile as sf
from transformers import AutoProcessor, CsmForConditionalGeneration

# =====================
# FastAPI setup
# =====================
app = FastAPI(title="AskVox CSM-1B CloudRun")

# =====================
# Configuration
# =====================
MODEL_DIR = "./csm-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Voice prompts inside the container
VOICE_PROMPTS = {
    "conversational_a": "prompts/conversational_a.wav",
    "conversational_b": "prompts/conversational_b.wav",
    "read_a": "prompts/read_speech_a.wav",
    "read_b": "prompts/read_speech_b.wav",
    "read_c": "prompts/read_speech_c.wav",
    "read_d": "prompts/read_speech_d.wav",
}

PROMPT_AUDIO = {}
DEFAULT_VOICE = "conversational_a"

# =====================
# Load processor & model
# =====================
print(f"Loading CSM-1B from {MODEL_DIR} on {DEVICE}...")

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = CsmForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("✅ Model loaded successfully!")

# =====================
# Load prompt audio
# =====================
for voice, path in VOICE_PROMPTS.items():
    if not os.path.exists(os.path.join(MODEL_DIR, path)):
        raise RuntimeError(f"Prompt file missing: {path}")
    audio, sr = sf.read(os.path.join(MODEL_DIR, path))
    PROMPT_AUDIO[voice] = (audio, sr)
print("✅ All prompts loaded!")

# =====================
# Input / output schemas
# =====================
class GenerateRequest(BaseModel):
    text: str
    voice: Literal["conversational_a","conversational_b","read_a","read_b","read_c","read_d"] = DEFAULT_VOICE

class GenerateResponse(BaseModel):
    audio_base64: str
    format: str
    sample_rate: int
    voice_used: str

# =====================
# Audio generation function
# =====================
def generate_audio(text: str, voice: str) -> bytes:
    if voice not in PROMPT_AUDIO:
        voice = DEFAULT_VOICE

    audio_prompt, sr = PROMPT_AUDIO[voice]
    text_input = f"[0]{text}"  # speaker prefix for CSM

    # Tokenize
    inputs = processor(text_input, add_special_tokens=True, return_tensors="pt").to(DEVICE)

    # Generate
    with torch.no_grad():
        audio_tokens = model.generate(**inputs, max_new_tokens=1024, output_audio=True)

    waveform = processor.decode(audio_tokens[0])

    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

# =====================
# Endpoints
# =====================
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Model ready"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Missing text input")
    try:
        audio_bytes = generate_audio(req.text, req.voice)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return GenerateResponse(
            audio_base64=audio_b64,
            format="wav",
            sample_rate=24000,
            voice_used=req.voice
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
