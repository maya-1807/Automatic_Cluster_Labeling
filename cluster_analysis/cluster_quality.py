"""
Per-cluster quality measures from the Cluster Hypothesis in IR literature.

Five measures (all are "higher = better focused/distinct" except KL divergence):
  1. Voorhees ('85) nearest-neighbor test      [focus, higher = better]
  2. Jardine & van Rijsbergen ('71) overlap    [focus+distinctiveness, higher = better]
  3. KL divergence between doc LMs             [focus, LOWER = better / more similar]
  4. El-Hamdouchi & Willett ('87) density      [focus, higher = more concentrated vocab]
  5. Smucker & Allan ('09) nMRD                [focus, higher = better]

Usage:
    python cluster_analysis/cluster_quality.py --dataset bbc_news
    python cluster_analysis/cluster_quality.py --dataset 20newsgroups --k 15
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PipelineConfig
from pipeline.embeddings import embed_documents

LOADERS = {
    "bbc_news": ("data_collection.bbc_news", "load_bbc_news"),
    "banking77": ("data_collection.banking77", "load_banking77"),
    "20newsgroups": ("data_collection.20newsgroups", "load_20newsgroups"),
    "ag_news": ("data_collection.ag_news", "load_ag_news"),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_dataset(name: str) -> pd.DataFrame:
    module_path, func_name = LOADERS[name]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def _load_embeddings(df: pd.DataFrame, dataset: str, cfg: PipelineConfig) -> np.ndarray:
    cache_path = os.path.join(cfg.cache_dir, f"{dataset}_{cfg.embedding_model}.npy")
    if cfg.use_cache and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    print("  Computing embeddings (this may take a while)...")
    embs = embed_documents(
        df["text"].tolist(),
        model_name=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
        device=cfg.embedding_device,
    )
    if cfg.use_cache:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        np.save(cache_path, embs)
        print(f"  Cached embeddings to {cache_path}")
    return embs


# ---------------------------------------------------------------------------
# Measure 1 – Voorhees ('85) Nearest-Neighbor Test
# ---------------------------------------------------------------------------

def voorhees_nn(
    embeddings: np.ndarray,
    label_array: np.ndarray,
    k: int = 10,
    medoid_sample: int = 2000,
    seed: int = 42,
) -> dict[str, float | None]:
    """
    Find the cluster medoid (the document with the highest mean cosine similarity
    to all other cluster members), then return the fraction of the medoid's k
    global nearest neighbors that belong to the same cluster.

    The medoid acts as the canonical "relevant document" for the cluster.
    For large clusters (> medoid_sample) an approximate medoid is found by
    sampling medoid_sample documents and picking the most central among them.

    Range: [0, 1]. Higher = the cluster is tightly focused around its medoid.

    Reference: Voorhees (1985), "The Cluster Hypothesis Revisited."
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(label_array)
    scores: dict[str, float | None] = {}

    for label in unique_labels:
        cluster_idx = np.where(label_array == label)[0]
        if len(cluster_idx) <= 1:
            scores[str(label)] = None
            continue

        # --- Find medoid (exact or approximate) ---
        sample = (
            cluster_idx
            if len(cluster_idx) <= medoid_sample
            else rng.choice(cluster_idx, medoid_sample, replace=False)
        )
        intra_sims = embeddings[sample] @ embeddings[sample].T  # (n_sample, n_sample)
        np.fill_diagonal(intra_sims, 0.0)
        mean_intra = intra_sims.sum(axis=1) / (len(sample) - 1)
        medoid_idx = sample[int(np.argmax(mean_intra))]

        # --- Voorhees score: fraction of k-NN of medoid in same cluster ---
        sims = embeddings[medoid_idx] @ embeddings.T  # (N,)
        sims[medoid_idx] = -2.0  # exclude self
        top_k = np.argpartition(sims, -k)[-k:]
        same = int((label_array[top_k] == label).sum())
        scores[str(label)] = float(same / k)

    return scores


# ---------------------------------------------------------------------------
# Measure 2 – Jardine & van Rijsbergen ('71) Overlap Test
# ---------------------------------------------------------------------------

def overlap_test(
    embeddings: np.ndarray,
    label_array: np.ndarray,
    max_sample: int = 500,
    seed: int = 42,
) -> dict[str, float | None]:
    """
    Mean intra-cluster cosine similarity minus mean inter-cluster cosine similarity.

    Operationalises the original overlap test as a continuous score.
    Range: [-1, 1]. Higher = distributions are better separated (focused + distinct).

    Reference: Jardine & van Rijsbergen (1971), "The Use of Hierarchical Clustering in
    Information Retrieval."
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(label_array)
    scores: dict[str, float | None] = {}

    for label in unique_labels:
        cluster_idx = np.where(label_array == label)[0]
        other_idx = np.where(label_array != label)[0]

        if len(cluster_idx) <= 1:
            scores[str(label)] = None
            continue

        sc = (
            cluster_idx
            if len(cluster_idx) <= max_sample
            else rng.choice(cluster_idx, max_sample, replace=False)
        )

        # Intra-cluster mean (upper triangle only, excludes diagonal)
        intra_mat = embeddings[sc] @ embeddings[sc].T
        n = len(sc)
        tri_i, tri_j = np.triu_indices(n, k=1)
        intra_mean = float(intra_mat[tri_i, tri_j].mean()) if len(tri_i) > 0 else 1.0

        # Inter-cluster mean (subsample both sides to bound memory)
        so = (
            other_idx
            if len(other_idx) <= max_sample
            else rng.choice(other_idx, max_sample, replace=False)
        )
        sc_sub = sc[: min(len(sc), 200)]
        so_sub = so[: min(len(so), 200)]
        inter_mat = embeddings[sc_sub] @ embeddings[so_sub].T
        inter_mean = float(inter_mat.mean())

        scores[str(label)] = float(intra_mean - inter_mean)

    return scores


# ---------------------------------------------------------------------------
# Measure 3 – KL Divergence between Document Language Models
# ---------------------------------------------------------------------------

def kl_divergence_lm(
    texts: list[str],
    label_array: np.ndarray,
    max_docs: int = 200,
    max_pairs: int = 500,
    smooth: float = 1e-5,
    seed: int = 42,
) -> dict[str, float | None]:
    """
    Mean symmetric KL divergence between unigram language models of document pairs
    within a cluster. Uses cluster-local vocabulary with Laplace-style smoothing.

    LOWER = documents share more similar vocabulary distributions = more focused.
    Range: [0, ∞). Typical values depend on vocabulary size and smoothing.

    Reference: Raiber & Kurland (2012), "Ranking Document Clusters Using Markov
    Random Fields."
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(label_array)
    scores: dict[str, float | None] = {}

    for label in unique_labels:
        cluster_idx = np.where(label_array == label)[0]
        if len(cluster_idx) <= 1:
            scores[str(label)] = None
            continue

        # Sample documents to bound memory (dense LM matrix)
        sample_idx = (
            cluster_idx
            if len(cluster_idx) <= max_docs
            else rng.choice(cluster_idx, max_docs, replace=False)
        )
        cluster_texts = [texts[i] for i in sample_idx]
        n = len(sample_idx)

        try:
            vec = CountVectorizer(min_df=1, token_pattern=r"(?u)\b\w+\b")
            X = vec.fit_transform(cluster_texts).toarray().astype(np.float64)
        except ValueError:
            scores[str(label)] = None
            continue

        # Smoothed unigram LMs: each row sums to 1
        X += smooth
        X /= X.sum(axis=1, keepdims=True)

        # Sample document pairs
        if n * (n - 1) // 2 <= max_pairs:
            idx_i, idx_j = np.triu_indices(n, k=1)
        else:
            a = rng.integers(0, n, max_pairs * 3)
            b = rng.integers(0, n, max_pairs * 3)
            valid = a != b
            idx_i = a[valid][:max_pairs]
            idx_j = b[valid][:max_pairs]

        P = X[idx_i]  # (pairs, V)
        Q = X[idx_j]
        # Symmetric KL: (KL(P‖Q) + KL(Q‖P)) / 2
        kl_pq = (P * np.log(P / Q)).sum(axis=1)
        kl_qp = (Q * np.log(Q / P)).sum(axis=1)
        kl_sym = (kl_pq + kl_qp) / 2

        scores[str(label)] = float(kl_sym.mean())

    return scores


# ---------------------------------------------------------------------------
# Measure 4 – El-Hamdouchi & Willett ('87) Density-Based Test
# ---------------------------------------------------------------------------

def density_test(
    texts: list[str],
    label_array: np.ndarray,
) -> dict[str, float | None]:
    """
    Fill factor of the cluster's document-term matrix:
        density = (non-zero entries) / (vocab_size × n_docs)

    Equivalently, the average fraction of vocabulary terms present in each document.
    Higher = more concentrated, overlapping vocabulary across cluster documents.
    Range: (0, 1].

    Reference: El-Hamdouchi & Willett (1987), "Comparison of Hierarchic Agglomerative
    Clustering Methods for Document Retrieval."
    """
    unique_labels = np.unique(label_array)
    scores: dict[str, float | None] = {}

    for label in unique_labels:
        cluster_idx = np.where(label_array == label)[0]
        if len(cluster_idx) == 0:
            scores[str(label)] = None
            continue

        cluster_texts = [texts[i] for i in cluster_idx]
        try:
            vec = CountVectorizer(min_df=1, token_pattern=r"(?u)\b\w+\b")
            X = vec.fit_transform(cluster_texts)  # sparse (n_docs, vocab)
        except ValueError:
            scores[str(label)] = None
            continue

        n_docs, vocab_size = X.shape
        postings = X.nnz
        density = postings / (vocab_size * n_docs)
        scores[str(label)] = float(density)

    return scores


# ---------------------------------------------------------------------------
# Measure 5 – Smucker & Allan ('09) Normalized Mean Reciprocal Distance (nMRD)
# ---------------------------------------------------------------------------

def nmrd(
    embeddings: np.ndarray,
    label_array: np.ndarray,
    max_sample: int = 200,
    batch_size: int = 20,
    seed: int = 42,
) -> dict[str, float | None]:
    """
    For each (sampled) document d in cluster C, compute the mean reciprocal rank
    of every other same-cluster document in d's globally sorted similarity list.
    Normalize by the ideal MRD (same-cluster docs occupy the top positions).

    nMRD(d, C) = mean_{d' in C\{d}} [1 / rank(d')] / ideal_MRD(|C|)
    nMRD(C)    = mean_{d in sample} nMRD(d, C)

    Range: [0, 1]. Higher = same-cluster documents are each other's nearest neighbors.

    Reference: Smucker & Allan (2009), "A New Measure of the Cluster Hypothesis."
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(label_array)
    N = len(embeddings)
    scores: dict[str, float | None] = {}

    rank_array = np.empty(N, dtype=np.int32)  # reused across iterations

    for label in unique_labels:
        cluster_idx = np.where(label_array == label)[0]
        cluster_size = len(cluster_idx)

        if cluster_size <= 1:
            scores[str(label)] = None
            continue

        # Ideal MRD: if the cluster_size-1 nearest neighbors are all same-cluster
        # ideal_MRD = H(cluster_size - 1) / (cluster_size - 1)
        ideal_mrd = float(
            np.sum(1.0 / np.arange(1, cluster_size)) / (cluster_size - 1)
        )

        sample = (
            cluster_idx
            if cluster_size <= max_sample
            else rng.choice(cluster_idx, max_sample, replace=False)
        )

        mrd_vals: list[float] = []

        for b_start in range(0, len(sample), batch_size):
            batch = sample[b_start : b_start + batch_size]
            sims = embeddings[batch] @ embeddings.T  # (b, N)

            for i, doc_idx in enumerate(batch):
                sim_row = sims[i].copy()
                sim_row[doc_idx] = -2.0  # exclude self

                # Assign ranks: rank_array[j] = 1-based position of doc j
                sorted_positions = np.argsort(-sim_row)
                rank_array[sorted_positions] = np.arange(1, N + 1)

                # Ranks of all same-cluster docs (excluding self)
                same_cluster_docs = cluster_idx[cluster_idx != doc_idx]
                same_ranks = rank_array[same_cluster_docs]

                mrd = float(np.mean(1.0 / same_ranks))
                mrd_vals.append(mrd / ideal_mrd)

        scores[str(label)] = float(np.mean(mrd_vals))

    return scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_measures(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k_nn: int = 10,
    medoid_sample: int = 2000,
    max_sample_overlap: int = 500,
    max_docs_kl: int = 200,
    max_pairs_kl: int = 500,
    max_sample_nmrd: int = 200,
    batch_size: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, float | None]]:
    """
    Compute all 5 cluster quality measures and combine into a single dict.

    Returns:
        {cluster_label: {measure_name: score, ...}, ...}
    """
    texts = df["text"].tolist()
    label_array = df["label"].values

    print("  [1/5] Voorhees nearest-neighbor test (medoid-based)...")
    v = voorhees_nn(
        embeddings, label_array, k=k_nn,
        medoid_sample=medoid_sample, seed=seed,
    )

    print("  [2/5] Overlap test (Jardine & van Rijsbergen)...")
    o = overlap_test(embeddings, label_array, max_sample=max_sample_overlap, seed=seed)

    print("  [3/5] KL divergence between document LMs...")
    kl = kl_divergence_lm(
        texts, label_array,
        max_docs=max_docs_kl, max_pairs=max_pairs_kl, seed=seed,
    )

    print("  [4/5] Density-based test (El-Hamdouchi & Willett)...")
    d = density_test(texts, label_array)

    print("  [5/5] Normalized mean reciprocal distance (nMRD)...")
    n = nmrd(
        embeddings, label_array,
        max_sample=max_sample_nmrd, batch_size=max(1, batch_size // 5), seed=seed,
    )

    all_labels = sorted(
        set(v) | set(o) | set(kl) | set(d) | set(n),
        key=lambda x: x,
    )
    return {
        label: {
            "voorhees_nn": v.get(label),
            "overlap": o.get(label),
            "kl_divergence": kl.get(label),
            "density": d.get(label),
            "nmrd": n.get(label),
        }
        for label in all_labels
    }


def _print_table(results: dict, dataset: str) -> None:
    measures = ["voorhees_nn", "overlap", "kl_divergence", "density", "nmrd"]
    headers = ["cluster", "voorhees_nn↑", "overlap↑", "kl_div↓", "density↑", "nmrd↑"]

    col_w = max(max(len(lbl) for lbl in results) + 2, 20)
    row_fmt = f"{{:<{col_w}}} " + " ".join(["{:>13}"] * 5)

    print(f"\n{'='*80}")
    print(f"Cluster quality measures — {dataset}")
    print(f"{'='*80}")
    print(row_fmt.format(*headers))
    print("-" * 80)

    for label, scores in sorted(results.items()):
        vals = []
        for m in measures:
            v = scores[m]
            vals.append(f"{v:.4f}" if v is not None else "  n/a")
        print(row_fmt.format(label, *vals))

    print("-" * 80)
    # Column means (skip None)
    means = []
    for m in measures:
        vs = [results[lbl][m] for lbl in results if results[lbl][m] is not None]
        means.append(f"{np.mean(vs):.4f}" if vs else "  n/a")
    print(row_fmt.format("MEAN", *means))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-cluster quality measures (Cluster Hypothesis in IR)."
    )
    parser.add_argument("--dataset", default="bbc_news", choices=list(LOADERS))
    parser.add_argument("--k", type=int, default=10, help="k for Voorhees NN test")
    parser.add_argument(
        "--medoid-sample", type=int, default=2000,
        help="Max docs sampled per cluster to find the approximate medoid",
    )
    parser.add_argument(
        "--max-sample", type=int, default=500,
        help="Max docs sampled per cluster for overlap measure",
    )
    parser.add_argument(
        "--max-docs-kl", type=int, default=200,
        help="Max docs sampled per cluster for KL divergence",
    )
    parser.add_argument(
        "--max-sample-nmrd", type=int, default=200,
        help="Max docs sampled per cluster for nMRD",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n=== Cluster Quality Analysis: {args.dataset} ===")

    print("\n[1/3] Loading dataset...")
    df = _load_dataset(args.dataset)
    print(f"  {len(df)} documents, {df['label'].nunique()} clusters")

    print("\n[2/3] Loading embeddings...")
    cfg = PipelineConfig(dataset=args.dataset)
    embeddings = _load_embeddings(df, args.dataset, cfg)

    print("\n[3/3] Computing measures...")
    results = compute_all_measures(
        df,
        embeddings,
        k_nn=args.k,
        medoid_sample=args.medoid_sample,
        max_sample_overlap=args.max_sample,
        max_docs_kl=args.max_docs_kl,
        max_pairs_kl=500,
        max_sample_nmrd=args.max_sample_nmrd,
        batch_size=100,
        seed=args.seed,
    )

    _print_table(results, args.dataset)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.dataset}_cluster_quality.json")
    csv_path = os.path.join(args.output_dir, f"{args.dataset}_cluster_quality.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    rows = [{"cluster": lbl, **scores} for lbl, scores in results.items()]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved CSV  → {csv_path}")


if __name__ == "__main__":
    main()
