import os
import io
import base64
import torch
import soundfile as sf
import runpod
from transformers import AutoProcessor, CsmForConditionalGeneration

# -----------------------
# Configuration
# -----------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")
HF_TOKEN = os.getenv("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} on {DEVICE}")

# -----------------------
# Load processor & model
# -----------------------
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN
)

model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)

model.eval()
print("Model loaded successfully")

# -----------------------
# Voice prompts
# -----------------------
VOICE_MAP = {
    "conversational_a": "prompts/conversational_a.wav",
    "conversational_b": "prompts/conversational_b.wav",
    "read_a": "prompts/read_speech_a.wav",
    "read_b": "prompts/read_speech_b.wav",
    "read_c": "prompts/read_speech_c.wav",
    "read_d": "prompts/read_speech_d.wav",
}

DEFAULT_VOICE = "conversational_a"

# Preload prompt audio into memory (faster)
PROMPT_AUDIO = {}
for voice, path in VOICE_MAP.items():
    try:
        audio, sr = sf.read(path)
        PROMPT_AUDIO[voice] = (audio, sr)
        print(f"Loaded prompt voice: {voice}")
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")

# -----------------------
# Audio generation
# -----------------------
def generate_audio(text: str, voice: str) -> bytes:
    # Get prompt audio
    audio_prompt, sr = PROMPT_AUDIO.get(voice, PROMPT_AUDIO[DEFAULT_VOICE])

    # Prepare inputs
    inputs = processor(
        text=text,
        audio=audio_prompt,
        sampling_rate=sr,
        return_tensors="pt"
    )

    # Move tensors to GPU
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        audio_tokens = model.generate(**inputs)

    # Decode tokens → waveform
    waveform = processor.decode(audio_tokens[0])

    # Save to buffer
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    return buffer.read()

# -----------------------
# RunPod handler
# -----------------------
def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text")
    voice = job_input.get("voice", DEFAULT_VOICE)

    if not text:
        return {"error": "Missing input.text"}

    if voice not in PROMPT_AUDIO:
        voice = DEFAULT_VOICE

    try:
        audio_bytes = generate_audio(text, voice)

        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000,
            "voice_used": voice
        }

    except Exception as e:
        print("Generation error:", e)
        return {"error": str(e)}

# -----------------------
# Start RunPod
# -----------------------
runpod.serverless.start({"handler": handler})
