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
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b/csm-1b")
HF_TOKEN = os.getenv("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CSM-1B model: {MODEL_ID} on {DEVICE}...")

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
print("Model loaded successfully!")

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

# Preload prompts into memory
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
def generate_audio(text: str, voice: str, use_context: bool = False) -> bytes:
    """
    Generate TTS audio from text using CSM-1B.
    - text: the sentence to speak
    - voice: which prompt audio to use
    - use_context: if True, include the prompt audio as context
    """
    if voice not in PROMPT_AUDIO:
        voice = DEFAULT_VOICE
    audio_prompt, sr = PROMPT_AUDIO[voice]

    # Always prefix with speaker ID
    text_input = f"[0]{text}"

    if use_context:
        # Multi-turn / context-aware TTS
        conversation = [
            {"role": "0", "content": [{"type": "audio", "path": audio_prompt}, {"type": "text", "text": text}]}
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True
        ).to(DEVICE)
    else:
        # Simple single-sentence TTS
        inputs = processor(text_input, add_special_tokens=True, return_tensors="pt").to(DEVICE)

    # Generate audio
    with torch.no_grad():
        audio_tokens = model.generate(**inputs, output_audio=True)

    # Decode to waveform
    waveform = processor.decode(audio_tokens[0])

    # Save to bytes
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

# -----------------------
# RunPod serverless handler
# -----------------------
def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text")
    voice = job_input.get("voice", DEFAULT_VOICE)
    use_context = job_input.get("use_context", False)

    if not text or not isinstance(text, str):
        return {"error": "Missing or invalid input.text"}

    try:
        audio_bytes = generate_audio(text, voice, use_context)
        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000,
            "voice_used": voice,
            "use_context": use_context
        }
    except Exception as e:
        print("Generation error:", e)
        return {"error": str(e)}

# -----------------------
# Start RunPod serverless
# -----------------------
runpod.serverless.start({"handler": handler})
