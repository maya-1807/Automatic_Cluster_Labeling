"""Main entry point for the Graph Guided RAG pipeline."""

import importlib
import json
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from config import PipelineConfig
from pipeline.embeddings import embed_documents
from pipeline.graph import build_all_graphs
from pipeline.centrality import select_all_central_documents
from pipeline.labeling import label_all_clusters
from evaluation.metrics import evaluate_labels

# Maps dataset name -> (module path, function name)
LOADERS = {
    "20newsgroups": ("data_collection.20newsgroups", "load_20newsgroups"),
    "ag_news": ("data_collection.ag_news", "load_ag_news"),
    "banking77": ("data_collection.banking77", "load_banking77"),
    "bbc_news": ("data_collection.bbc_news", "load_bbc_news"),
}


def load_dataset(name: str):
    """Dynamically import and call the appropriate data loader."""
    module_path, func_name = LOADERS[name]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def run(cfg: PipelineConfig):
    print("=== Graph Guided RAG Pipeline ===")
    print(f"Dataset: {cfg.dataset}")

    # --- Load data ---
    print("\n[1/5] Loading dataset...")
    df = load_dataset(cfg.dataset)
    print(f"  {len(df)} documents, {df['label'].nunique()} clusters")

    # --- Embed ---
    print("\n[2/5] Embedding documents...")
    cache_path = os.path.join(
        cfg.cache_dir, f"{cfg.dataset}_{cfg.embedding_model}.npy"
    )

    if cfg.use_cache and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
    else:
        embeddings = embed_documents(
            df["text"].tolist(),
            model_name=cfg.embedding_model,
            batch_size=cfg.embedding_batch_size,
            device=cfg.embedding_device,
        )
        if cfg.use_cache:
            os.makedirs(cfg.cache_dir, exist_ok=True)
            np.save(cache_path, embeddings)
            print(f"  Cached embeddings to {cache_path}")

    # --- Build graphs ---
    print("\n[3/5] Building cluster graphs...")
    graphs = build_all_graphs(
        df,
        embeddings,
        similarity_threshold=cfg.similarity_threshold,
        chunk_size=cfg.chunk_size,
    )

    # --- Select central documents ---
    print("\n[4/5] Selecting central documents via PageRank...")
    central_docs = select_all_central_documents(
        graphs,
        df,
        top_k=cfg.top_k,
        pagerank_alpha=cfg.pagerank_alpha,
    )

    # --- Generate labels ---
    print("\n[5/5] Generating cluster labels...")
    generated_labels = label_all_clusters(
        central_docs,
        model=cfg.labeling_model,
        max_words=cfg.max_doc_words,
    )

    # --- Evaluate ---
    print("\n=== Evaluation ===")
    results = evaluate_labels(generated_labels, cfg.embedding_model)

    # --- Save results ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    output = {
        "config": cfg.__dict__,
        "labels": {
            gt: {"generated": gen} for gt, gen in generated_labels.items()
        },
        "metrics": results,
    }
    return output


def run_hyperparameter_sweep(
    dataset: str,
    similarity_thresholds: list[float] = [0.3, 0.5, 0.7],
    top_ks: list[int] = [5, 10, 15],
    pagerank_alphas: list[float] = [0.85],
    **config_overrides,
) -> list[dict]:
    """
    Run the pipeline on a dataset across combinations of hyperparameters.

    Args:
        dataset: Dataset name (e.g. "bbc_news").
        similarity_thresholds: Thresholds to sweep.
        top_ks: Values of k to sweep.
        pagerank_alphas: Damping factors to sweep.
        **config_overrides: Any other PipelineConfig fields to override.

    Returns:
        List of result dicts, one per hyperparameter combination.
    """
    all_results = []
    combos = [
        (t, k, a)
        for t in similarity_thresholds
        for k in top_ks
        for a in pagerank_alphas
    ]
    total = len(combos)

    for i, (threshold, top_k, alpha) in enumerate(combos, 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{total}: threshold={threshold}, top_k={top_k}, alpha={alpha}")
        print(f"{'='*60}")

        cfg = PipelineConfig(
            dataset=dataset,
            similarity_threshold=threshold,
            top_k=top_k,
            pagerank_alpha=alpha,
            **config_overrides,
        )
        result = run(cfg)
        all_results.append(result)

    # Save all results as a list
    output_dir = config_overrides.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset}_sweep.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSweep results saved to {out_path}")

    return all_results


def run_full_sweep(
    datasets: list[str] | None = None,
    similarity_thresholds: list[float] = [0.3, 0.5, 0.7],
    top_ks: list[int] = [3, 5, 10],
    pagerank_alphas: list[float] = [0.5, 0.85, 0.95],
    **config_overrides,
) -> dict[str, list[dict]]:
    """
    Run hyperparameter sweep across all datasets.

    Args:
        datasets: List of dataset names. Defaults to all available datasets.
        similarity_thresholds: Thresholds to sweep.
        top_ks: Values of k to sweep.
        pagerank_alphas: Damping factors to sweep.
        **config_overrides: Any other PipelineConfig fields to override.

    Returns:
        dict mapping dataset name -> list of result dicts.
    """
    if datasets is None:
        datasets = list(LOADERS.keys())

    all_results = {}
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*60}")
        all_results[dataset] = run_hyperparameter_sweep(
            dataset=dataset,
            similarity_thresholds=similarity_thresholds,
            top_ks=top_ks,
            pagerank_alphas=pagerank_alphas,
            **config_overrides,
        )

    # Save combined results
    output_dir = config_overrides.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "full_sweep.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull sweep results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_full_sweep()
