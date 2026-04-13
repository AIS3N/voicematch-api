from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import embed, similarity, match


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up model at startup so first request isn't slow
    from core.model import get_model
    get_model()
    yield


app = FastAPI(
    title="VoiceMatch API",
    description=(
        "Voice fingerprinting and similarity scoring. "
        "Extract voice embeddings from audio, then compare or rank them."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(embed.router, tags=["Embedding"])
app.include_router(similarity.router, tags=["Similarity"])
app.include_router(match.router, tags=["Matching"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
