import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

_model = None
_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_MODEL_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"


def get_model():
    global _model
    if _model is None:
        _model = EncoderClassifier.from_hparams(
            source=_MODEL_SOURCE,
            savedir=_MODEL_SAVEDIR,
            run_opts={"device": "cpu"},
        )
    return _model


def extract_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Audio -> PyTorch Tensor
    Extract an L2-normalized speaker embedding from a 16kHz mono audio array.
    Returns a 1D numpy array (192-dim for ECAPA-TDNN).
    """
    model = get_model()

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  #  (time,) -> (1, time)
    wav_lens = torch.tensor([1.0])

    with torch.no_grad():
        embeddings = model.encode_batch(audio_tensor, wav_lens)  # (1, 1, dim)

    embedding = embeddings.squeeze().numpy() # (1, 1, dim) tensor -> (dim,) numpy array

    # L2 normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Returned value mapped from [-1, 1] to [0, 1] for intuitives scores.
    """
    raw = float(np.dot(a, b))
    return round((raw + 1) / 2, 4)
