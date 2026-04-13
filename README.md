# VoiceMatch API

Domain-agnostic voice fingerprinting and similarity scoring. Extract compact numerical representations of a speaker's voice from audio, then compare or rank them.

## How it works

1. **Embed** — upload an audio file, get back a 192-dimensional voice fingerprint (embedding)
2. **Compare** — send two embeddings, get a similarity score between 0.0 and 1.0
3. **Match** — send one query embedding and a list of candidates, get them ranked by similarity

### Under the hood

- Audio is decoded, converted to 16kHz mono, denoised, and stripped of silence before processing
- The [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) model (SpeechBrain, trained on VoxCeleb) extracts a 192-dim L2-normalized speaker embedding
- Similarity is computed as cosine similarity, remapped to [0.0, 1.0]

---

## Routes

### `GET /health`
Returns `{ "status": "ok" }`. Use to verify the service is running.

---

### `POST /embed`
Extract a voice embedding from an audio file.

**Request:** `multipart/form-data`
- `file` — audio file (wav, mp3, m4a, ogg). Min 3s of clean speech, max 10MB, max 5 minutes.

**Response:**
```json
{
  "embedding": [0.042, -0.117, 0.334, ...],  // 192 floats, L2-normalized
  "duration": 4.83                             // seconds of clean speech detected
}
```

**Errors:**
- `400` — file too large, audio too short after noise removal, unsupported format

---

### `POST /similarity`
Compute a similarity score between two pre-computed embeddings.

**Request:**
```json
{
  "a": [0.042, -0.117, ...],
  "b": [0.031, -0.098, ...]
}
```

**Response:**
```json
{
  "score": 0.9142  // 0.0 = different voices, 1.0 = same voice
}
```

**Errors:**
- `400` — embeddings have different dimensions or are empty

---

### `POST /match`
Rank a list of candidate embeddings against a query embedding.

**Request:**
```json
{
  "query": [0.042, -0.117, ...],
  "candidates": [
    { "id": "user_1", "embedding": [0.031, -0.098, ...] },
    { "id": "user_2", "embedding": [-0.012, 0.204, ...] }
  ]
}
```

**Response:**
```json
{
  "results": [
    { "id": "user_1", "score": 0.9142 },
    { "id": "user_2", "score": 0.4231 }
  ]
}
```
Results are sorted by score descending (best match first).

**Errors:**
- `400` — query or candidates list is empty, embedding dimension mismatch

---

## Running locally

```bash
pyenv local 3.12.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`.

## Running with Docker

```bash
docker build -t voicematch-api .
docker run -p 8000:8000 voicematch-api
```

The model weights (~200MB) are downloaded during the Docker build step and baked into the image.
