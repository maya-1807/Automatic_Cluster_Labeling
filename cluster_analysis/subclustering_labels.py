"""
Sub-cluster labeling: for each ground-truth cluster, partition into C sub-clusters,
extract one medoid per sub-cluster, and send all C medoids to the LLM as context.

Idea: broad clusters (e.g. BBC "sport", AG "World") contain multiple coherent subtopics.
Rather than selecting a few PageRank-central documents (which tend to represent only the
dominant subtopic), we explicitly cover the cluster's diversity by partitioning it into C
sub-clusters and taking the medoid of each.

Usage:
    python cluster_analysis/subclustering_labels.py --dataset bbc_news
    python cluster_analysis/subclustering_labels.py --dataset ag_news --n-subclusters 3 5 8
    python cluster_analysis/subclustering_labels.py --dataset 20newsgroups --n-subclusters 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PipelineConfig
from pipeline.embeddings import embed_documents
from pipeline.labeling import generate_label
from evaluation.metrics import evaluate_labels
from groq import Groq

LOADERS = {
    "bbc_news":     ("data_collection.bbc_news",     "load_bbc_news"),
    "banking77":    ("data_collection.banking77",     "load_banking77"),
    "20newsgroups": ("data_collection.20newsgroups",  "load_20newsgroups"),
    "ag_news":      ("data_collection.ag_news",       "load_ag_news"),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_dataset(name: str) -> pd.DataFrame:
    module_path, func_name = LOADERS[name]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def _load_embeddings(df: pd.DataFrame, dataset: str, cfg: PipelineConfig) -> np.ndarray:
    cache_path = Path(cfg.cache_dir) / f"{dataset}_{cfg.embedding_model}.npy"
    if cfg.use_cache and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    print("  Computing embeddings...")
    embs = embed_documents(
        df["text"].tolist(),
        model_name=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
        device=cfg.embedding_device,
    )
    if cfg.use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)
        print(f"  Cached embeddings to {cache_path}")
    return embs


# ---------------------------------------------------------------------------
# Core: sub-cluster a single cluster and return one medoid per sub-cluster
# ---------------------------------------------------------------------------

def subclustering_medoids(
    cluster_embs: np.ndarray,
    cluster_df: pd.DataFrame,
    n_subclusters: int,
    seed: int = 42,
) -> list[dict]:
    """
    Partition cluster_embs into n_subclusters via K-Means, then pick the medoid
    (document closest to the sub-cluster centroid in cosine similarity) for each.

    Embeddings are assumed L2-normalised, so dot product == cosine similarity.
    The medoid is the cluster member that maximises dot-product with the centroid.

    Returns:
        List of n_subclusters dicts with keys: index, text, subcluster_id.
        Deduplication is applied in the rare case where two sub-clusters share
        the same medoid (only possible when a sub-cluster has one member).
    """
    n = len(cluster_embs)
    C = min(n_subclusters, n)  # can't have more sub-clusters than documents

    if C == 1:
        # Degenerate case: return the cluster centroid's nearest neighbour
        centroid = cluster_embs.mean(axis=0)
        idx = int(np.argmax(cluster_embs @ centroid))
        row = cluster_df.iloc[idx]
        return [{"index": int(row.name), "text": str(row["text"]), "subcluster_id": 0}]

    km = KMeans(
        n_clusters=C,
        init="k-means++",
        n_init=10,
        random_state=seed,
    )
    sub_labels = km.fit_predict(cluster_embs)   # (n,)  int in [0, C)
    centroids  = km.cluster_centers_             # (C, dim)  – not L2-normalised

    seen_indices: set[int] = set()
    medoids: list[dict] = []

    for sub_id in range(C):
        mask = sub_labels == sub_id
        if not mask.any():
            continue
        sub_embs      = cluster_embs[mask]      # (n_sub, dim)
        sub_df_rows   = cluster_df[mask]

        # Cosine similarity to centroid: embs are unit-norm, centroid need not be
        sims          = sub_embs @ centroids[sub_id]        # (n_sub,)
        local_best    = int(np.argmax(sims))

        global_row    = sub_df_rows.iloc[local_best]
        global_idx    = int(global_row.name)

        if global_idx in seen_indices:
            # Fall back to the second-best in this sub-cluster (very rare)
            order = np.argsort(-sims)
            for alt in order:
                alt_idx = int(sub_df_rows.iloc[alt].name)
                if alt_idx not in seen_indices:
                    global_idx  = alt_idx
                    global_row  = sub_df_rows.iloc[alt]
                    break

        seen_indices.add(global_idx)
        medoids.append({
            "index":        global_idx,
            "text":         str(global_row["text"]),
            "subcluster_id": sub_id,
        })

    return medoids


def select_all_subcluster_medoids(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_subclusters: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Run subclustering_medoids for every ground-truth cluster in df.

    Returns:
        dict mapping ground-truth label -> list of medoid dicts.
    """
    result: dict[str, list[dict]] = {}
    for label, group in df.groupby("label"):
        idx        = group.index.tolist()
        cl_embs    = embeddings[idx]        # already indexed by df position
        cl_df      = df.loc[idx]
        medoids    = subclustering_medoids(cl_embs, cl_df, n_subclusters, seed)
        result[label] = medoids
        print(f"  Cluster '{label}': {len(group)} docs → {len(medoids)} sub-cluster medoids")
    return result


# ---------------------------------------------------------------------------
# LLM labeling (thin wrapper that reuses pipeline.labeling.generate_label)
# ---------------------------------------------------------------------------

def label_all_clusters(
    medoids_by_cluster: dict[str, list[dict]],
    model: str = "llama-3.1-8b-instant",
    max_words: int = 300,
) -> dict[str, str]:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    labels: dict[str, str] = {}
    total = len(medoids_by_cluster)
    for i, (cluster_label, docs) in enumerate(medoids_by_cluster.items(), 1):
        generated = generate_label(docs, client, model=model, max_words=max_words)
        labels[cluster_label] = generated
        print(f"  [{i}/{total}] '{cluster_label}' -> '{generated}'")
    return labels


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------

def _load_baseline(dataset: str, results_dir: str) -> dict | None:
    path = Path(results_dir) / f"{dataset}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("test_result")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    dataset: str,
    n_subclusters_list: list[int],
    model: str = "llama-3.1-8b-instant",
    max_words: int = 300,
    results_dir: str = "results",
    output_dir: str = "results",
    seed: int = 42,
) -> dict:
    """
    Run the sub-cluster labeling approach for one dataset across multiple C values.

    Returns:
        dict with baseline + per-C results.
    """
    print(f"\n{'='*60}")
    print(f"Sub-cluster labeling: {dataset}")
    print(f"{'='*60}")

    # --- Load data and embeddings ---
    print("\n[1/3] Loading dataset & embeddings...")
    df = _load_dataset(dataset)
    print(f"  {len(df)} docs, {df['label'].nunique()} clusters")

    cfg = PipelineConfig(dataset=dataset)
    embeddings = _load_embeddings(df, dataset, cfg)

    # --- Baseline ---
    print("\n[2/3] Loading baseline results...")
    baseline = _load_baseline(dataset, results_dir)
    if baseline:
        bsem = baseline["metrics"]["semantic_similarity"]["__mean__"]
        btok = baseline["metrics"]["token_overlap_f1"]["__mean__"]
        print(f"  Baseline  sem_sim={bsem:.3f}  tok_f1={btok:.3f}")
        for cluster, v in baseline["labels"].items():
            print(f"    {cluster:<45} -> '{v['generated']}'")
    else:
        print("  No baseline found.")

    # --- Sub-cluster sweep ---
    print("\n[3/3] Sub-cluster labeling...")
    sweep_results: list[dict] = []

    for C in n_subclusters_list:
        print(f"\n  ── C={C} sub-clusters ──")

        medoids = select_all_subcluster_medoids(df, embeddings, C, seed)
        generated_labels = label_all_clusters(medoids, model=model, max_words=max_words)

        print("  Evaluating...")
        metrics = evaluate_labels(generated_labels, cfg.embedding_model, device=cfg.embedding_device)

        sweep_results.append({
            "n_subclusters": C,
            "labels": {gt: {"generated": gen} for gt, gen in generated_labels.items()},
            "metrics": metrics,
        })

        sem = metrics["semantic_similarity"]["__mean__"]
        tok = metrics["token_overlap_f1"]["__mean__"]
        delta_sem = f" ({sem - bsem:+.3f} vs baseline)" if baseline else ""
        print(f"  C={C}: sem_sim={sem:.3f}{delta_sem}  tok_f1={tok:.3f}")

    # --- Summary table ---
    print(f"\n{'─'*60}")
    print(f"{'C':>5}  {'sem_sim':>8}  {'tok_f1':>8}  {'Δ sem_sim':>10}")
    print(f"{'─'*60}")
    if baseline:
        print(f"{'base':>5}  {bsem:>8.3f}  {btok:>8.3f}  {'':>10}")
    for r in sweep_results:
        C   = r["n_subclusters"]
        sem = r["metrics"]["semantic_similarity"]["__mean__"]
        tok = r["metrics"]["token_overlap_f1"]["__mean__"]
        delta = f"{sem - bsem:+.3f}" if baseline else "n/a"
        print(f"{C:>5}  {sem:>8.3f}  {tok:>8.3f}  {delta:>10}")
    print(f"{'─'*60}")

    output = {
        "dataset":  dataset,
        "baseline": baseline,
        "sweep":    sweep_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{dataset}_subclustering.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sub-cluster labeling: K-Means sub-clusters → medoid selection → LLM."
    )
    parser.add_argument(
        "--dataset",
        choices=list(LOADERS),
        default="bbc_news",
        help="Dataset to run on.",
    )
    parser.add_argument(
        "--n-subclusters",
        type=int,
        nargs="+",
        default=[4],
        metavar="C",
        help="Number of sub-clusters per cluster. Multiple values = sweep (default: 3 5 8).",
    )
    parser.add_argument(
        "--model",
        default="llama-3.1-8b-instant",
        help="Groq model for label generation.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=300,
        help="Max words per document sent to the LLM.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        dataset=args.dataset,
        n_subclusters_list=args.n_subclusters,
        model=args.model,
        max_words=args.max_words,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
