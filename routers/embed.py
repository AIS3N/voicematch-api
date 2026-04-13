from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from core.audio import preprocess_audio
from core.model import extract_embedding

router = APIRouter()


class EmbedResponse(BaseModel):
    embedding: list[float]
    duration: float


@router.post("/embed", response_model=EmbedResponse)
async def embed(file: UploadFile = File(...)):
    """
    Extract a voice embedding from an audio file.

    - **file**: Audio file (wav, mp3, m4a, ogg). Min 5s of clean speech, max 10MB.

    Returns a 192-dimensional L2-normalized embedding vector and the duration
    of clean speech detected (in seconds).
    """
    file_bytes = await file.read()
    audio, duration = preprocess_audio(file_bytes)
    embedding = extract_embedding(audio)
    return EmbedResponse(embedding=embedding.tolist(), duration=round(duration, 2))
