import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from routers import embed, similarity, match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voicematch")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.model import get_model
    logger.info("Loading model...")
    get_model()
    logger.info("Model loaded.")
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = round((time.perf_counter() - start) * 1000)
    forwarded_for = request.headers.get("x-forwarded-for")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")
    logger.info("%s %s %s %dms %s", request.method, request.url.path, response.status_code, duration, ip)
    return response


app.include_router(embed.router, tags=["Embedding"])
app.include_router(similarity.router, tags=["Similarity"])
app.include_router(match.router, tags=["Matching"])


@app.get("/health", tags=["Health"])
def health(response: Response):
    from core.model import _model
    if _model is None:
        response.status_code = 503
        return {"status": "starting", "model": "not_loaded"}
    return {"status": "ok", "model": "loaded"}
