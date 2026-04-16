import io
import subprocess
import numpy as np
import librosa
import noisereduce as nr
from fastapi import HTTPException

TARGET_SR = 16000
MIN_CLEAN_SECONDS = 3.0
MAX_CLEAN_SECONDS = 30.0  # truncate after silence stripping — model gains nothing beyond this
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_DURATION_SECONDS = 300  # 5 minutes — reject before loading


def _to_wav(file_bytes: bytes) -> bytes:
    """Convert any ffmpeg-supported format to 16kHz mono WAV via stdin/stdout.
    Truncates to MAX_CLEAN_SECONDS at decode time to cap memory usage.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-t", str(int(MAX_CLEAN_SECONDS)),
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "-f", "wav",
            "pipe:1",
        ],
        input=file_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {result.stderr.decode(errors='replace').strip()}",
        )
    return result.stdout


def preprocess_audio(file_bytes: bytes, denoise: bool = False) -> tuple[np.ndarray, float]:
    """
    Convert audio to mono 16kHz, strip silence, normalize amplitude.
    If denoise=True, applies noise reduction before silence stripping (may affect similarity accuracy).
    Returns (audio_array, duration_seconds).
    Raises HTTPException 400 if audio is invalid or too short after cleaning.
    """
    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit.")

    try:
        wav_bytes = _to_wav(file_bytes)
        audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    if len(audio) / TARGET_SR > MAX_DURATION_SECONDS:
        raise HTTPException(status_code=400, detail="Audio exceeds 5-minute limit.")

    if denoise:
        audio = nr.reduce_noise(y=audio, sr=TARGET_SR, prop_decrease=0.5)

    # Voice activity detection — strip silent segments
    audio = _strip_silence(audio)

    duration = len(audio) / TARGET_SR
    if duration < MIN_CLEAN_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {duration:.1f}s of speech detected after cleaning "
                f"(minimum: {MIN_CLEAN_SECONDS}s). Record in a quieter environment."
            ),
        )

    # Peak normalize to 0.95
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio, duration


def _strip_silence(audio: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Remove silent frames using energy-based VAD."""
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    return np.concatenate([audio[start:end] for start, end in intervals])
