from fastapi import APIRouter, UploadFile, File, Form, Request
from pydantic import BaseModel

from core.audio import preprocess_audio
from core.model import extract_embedding
from core.ratelimit import check_rate_limit

router = APIRouter()


class EmbedResponse(BaseModel):
    embedding: list[float]
    duration: float
    processed_audio: str
    denoise_applied: bool


@router.post("/embed", response_model=EmbedResponse)
async def embed(
    request: Request,
    file: UploadFile = File(...),
    denoise: bool = Form(False),
):
    """
    Extract a voice embedding from an audio file.

    - **file**: Audio file (wav, mp3, m4a, ogg). Min 3s of clean speech, max 10MB.
    - **denoise**: Apply noise reduction before processing (default: false).
      Warning: may affect similarity accuracy on clean recordings.

    Returns a 192-dimensional L2-normalized embedding vector, the duration
    of clean speech detected (in seconds), and the processed audio as a base64
    WAV string (use as `data:audio/wav;base64,<processed_audio>`).
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    await check_rate_limit(ip)

    file_bytes = await file.read()
    audio, duration, processed_audio_b64 = preprocess_audio(file_bytes, denoise=denoise)
    embedding = extract_embedding(audio)
    return EmbedResponse(
        embedding=embedding.tolist(),
        duration=round(duration, 2),
        processed_audio=processed_audio_b64,
        denoise_applied=denoise,
    )
