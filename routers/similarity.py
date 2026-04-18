import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from core.model import cosine_similarity

router = APIRouter()


EMBEDDING_DIM = 192


class SimilarityRequest(BaseModel):
    a: list[float]
    b: list[float]

    @field_validator("a", "b")
    @classmethod
    def must_be_valid_embedding(cls, v):
        if len(v) == 0:
            raise ValueError("Embedding must not be empty.")
        if len(v) != EMBEDDING_DIM:
            raise ValueError(f"Embedding must be {EMBEDDING_DIM}-dimensional, got {len(v)}.")
        return v


class SimilarityResponse(BaseModel):
    score: float  # 0.0 → 1.0


@router.post("/similarity", response_model=SimilarityResponse)
def similarity(body: SimilarityRequest):
    """
    Compute similarity score between two voice embeddings.

    - **a**, **b**: Embedding vectors (must be same dimension).

    Returns a score between 0.0 (completely different) and 1.0 (identical voice).
    """
    a = np.array(body.a, dtype=np.float32)
    b = np.array(body.b, dtype=np.float32)

    if a.shape != b.shape:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dimensions don't match: {a.shape} vs {b.shape}.",
        )

    score = cosine_similarity(a, b)
    return SimilarityResponse(score=score)
