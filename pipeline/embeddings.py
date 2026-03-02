"""Stage 1a: Embed documents using a sentence-transformer model."""

import numpy as np
from sentence_transformers import SentenceTransformer


def embed_documents(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    Args:
        texts: List of document strings.
        model_name: Sentence-transformer model to use.
        batch_size: Encoding batch size.
        device: Device to run on (None = auto-detect).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized.
    """
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # dot product = cosine similarity
    )
    return embeddings
