import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from core.model import cosine_similarity

router = APIRouter()


class Candidate(BaseModel):
    id: str
    embedding: list[float]


class MatchRequest(BaseModel):
    query: list[float]
    candidates: list[Candidate]

    @field_validator("query")
    @classmethod
    def query_non_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Query embedding must not be empty.")
        return v

    @field_validator("candidates")
    @classmethod
    def candidates_non_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Candidates list must not be empty.")
        return v


class MatchResult(BaseModel):
    id: str
    score: float


class MatchResponse(BaseModel):
    results: list[MatchResult]  # sorted descending by score


@router.post("/match", response_model=MatchResponse)
def match(body: MatchRequest):
    """
    Rank candidates by voice similarity to a query embedding.

    - **query**: The voice embedding to match against.
    - **candidates**: List of `{ id, embedding }` objects to rank.

    Returns all candidates sorted by similarity score (highest first).
    """
    query = np.array(body.query, dtype=np.float32)
    results = []

    for candidate in body.candidates:
        emb = np.array(candidate.embedding, dtype=np.float32)
        if emb.shape != query.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Candidate '{candidate.id}' embedding dimension mismatch.",
            )
        score = cosine_similarity(query, emb)
        results.append(MatchResult(id=candidate.id, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return MatchResponse(results=results)
