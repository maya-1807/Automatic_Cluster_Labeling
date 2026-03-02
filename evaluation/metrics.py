"""Evaluation metrics for generated cluster labels."""

import numpy as np
from sentence_transformers import SentenceTransformer


def semantic_similarity(
    generated_labels: dict[str, str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
) -> dict[str, float]:
    """
    Cosine similarity between ground-truth and generated label embeddings.

    Args:
        generated_labels: dict mapping ground-truth label -> generated label.
        model_name: Sentence-transformer model for encoding labels.
        device: Device for the model (e.g. "cpu", "cuda"). None = auto-detect.

    Returns:
        dict with per-cluster scores and a "__mean__" key.
    """
    kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **kwargs)

    gt_labels = list(generated_labels.keys())
    gen_labels = list(generated_labels.values())

    gt_embs = model.encode(gt_labels, normalize_embeddings=True)
    gen_embs = model.encode(gen_labels, normalize_embeddings=True)

    scores = {}
    for i, gt in enumerate(gt_labels):
        scores[gt] = float(gt_embs[i] @ gen_embs[i])

    scores["__mean__"] = np.mean([v for k, v in scores.items() if k != "__mean__"])
    return scores


def token_overlap_f1(generated_labels: dict[str, str]) -> dict[str, float]:
    """
    Token-level F1 between ground-truth and generated labels.

    Args:
        generated_labels: dict mapping ground-truth label -> generated label.

    Returns:
        dict with per-cluster F1 scores and a "__mean__" key.
    """
    scores = {}
    for gt, gen in generated_labels.items():
        gt_tokens = set(gt.lower().replace("_", " ").replace(".", " ").split())
        gen_tokens = set(gen.lower().split())

        if not gen_tokens:
            scores[gt] = 0.0
            continue

        common = gt_tokens & gen_tokens
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(gt_tokens)
        if precision + recall > 0:
            scores[gt] = 2 * precision * recall / (precision + recall)
        else:
            scores[gt] = 0.0

    scores["__mean__"] = np.mean([v for k, v in scores.items() if k != "__mean__"])
    return scores


def evaluate_labels(
    generated_labels: dict[str, str],
    embedding_model: str = "all-MiniLM-L6-v2",
    device: str | None = None,
) -> dict:
    """
    Run all evaluation metrics and return combined results.

    Args:
        generated_labels: dict mapping ground-truth label -> generated label.
        embedding_model: Model for semantic similarity computation.
        device: Device for the model (e.g. "cpu", "cuda"). None = auto-detect.

    Returns:
        dict with "semantic_similarity" and "token_overlap_f1" results.
    """
    sem_sim = semantic_similarity(generated_labels, embedding_model, device=device)
    tok_f1 = token_overlap_f1(generated_labels)

    print(f"  Semantic Similarity (mean): {sem_sim['__mean__']:.3f}")
    print(f"  Token Overlap F1 (mean):    {tok_f1['__mean__']:.3f}")

    return {
        "semantic_similarity": sem_sim,
        "token_overlap_f1": tok_f1,
    }
