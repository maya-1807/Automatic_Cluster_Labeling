"""Main entry point for the Graph Guided RAG pipeline."""

import importlib
import json
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from config import PipelineConfig
from data_collection.split import stratified_split
from pipeline.embeddings import embed_documents
from pipeline.graph import build_all_graphs
from pipeline.centrality import select_all_central_documents
from pipeline.labeling import label_all_clusters
from evaluation.metrics import evaluate_labels

# Maps dataset name -> (module path, function name)
LOADERS = {
    "bbc_news": ("data_collection.bbc_news", "load_bbc_news"),
    "banking77": ("data_collection.banking77", "load_banking77"),
    "20newsgroups": ("data_collection.20newsgroups", "load_20newsgroups"),
    "ag_news": ("data_collection.ag_news", "load_ag_news"),
}


def load_dataset(name: str):
    """Dynamically import and call the appropriate data loader."""
    module_path, func_name = LOADERS[name]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def run(
    cfg: PipelineConfig,
    df: pd.DataFrame | None = None,
    embeddings: np.ndarray | None = None,
):
    split_tag = f" ({cfg.split})" if cfg.split else ""
    print("=== Graph Guided RAG Pipeline ===")
    print(f"Dataset: {cfg.dataset}{split_tag}")

    # --- Load data ---
    if df is None:
        print("\n[1/5] Loading dataset...")
        df = load_dataset(cfg.dataset)
    else:
        print("\n[1/5] Using pre-loaded dataset...")
    print(f"  {len(df)} documents, {df['label'].nunique()} clusters")

    # --- Embed ---
    print("\n[2/5] Embedding documents...")
    if embeddings is not None:
        print("  Using pre-loaded embeddings")
    else:
        split_suffix = f"_{cfg.split}" if cfg.split else ""
        cache_path = os.path.join(
            cfg.cache_dir, f"{cfg.dataset}{split_suffix}_{cfg.embedding_model}.npy"
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
    results = evaluate_labels(generated_labels, cfg.embedding_model, device=cfg.embedding_device)

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
) -> dict:
    """
    Sweep hyperparameters on an 80/20 dev/test split.

    Runs all combinations on the dev split, picks the best config by
    mean semantic similarity, then evaluates once on the held-out test split.

    Args:
        dataset: Dataset name (e.g. "bbc_news").
        similarity_thresholds: Thresholds to sweep.
        top_ks: Values of k to sweep.
        pagerank_alphas: Damping factors to sweep.
        **config_overrides: Any other PipelineConfig fields to override.

    Returns:
        dict with keys: dev_sweep, best_config, best_dev_metrics, test_result.
    """
    # --- Load and split once ---
    print(f"\nLoading and splitting {dataset}...")
    full_df = load_dataset(dataset)
    split = stratified_split(full_df, test_size=0.2, dataset_name=dataset)
    print(f"  Full: {len(full_df)} docs | Dev: {len(split.dev)} docs | Test: {len(split.test)} docs")

    # --- Pre-compute dev embeddings once ---
    base_cfg = PipelineConfig(dataset=dataset, split="dev", **config_overrides)
    cache_path = os.path.join(
        base_cfg.cache_dir, f"{dataset}_dev_{base_cfg.embedding_model}.npy"
    )
    if base_cfg.use_cache and os.path.exists(cache_path):
        print(f"  Loading cached dev embeddings from {cache_path}")
        dev_embeddings = np.load(cache_path)
    else:
        print("  Computing dev embeddings...")
        dev_embeddings = embed_documents(
            split.dev["text"].tolist(),
            model_name=base_cfg.embedding_model,
            batch_size=base_cfg.embedding_batch_size,
            device=base_cfg.embedding_device,
        )
        if base_cfg.use_cache:
            os.makedirs(base_cfg.cache_dir, exist_ok=True)
            np.save(cache_path, dev_embeddings)
            print(f"  Cached dev embeddings to {cache_path}")

    # --- Sweep on dev ---
    combos = [
        (t, k, a)
        for t in similarity_thresholds
        for k in top_ks
        for a in pagerank_alphas
    ]
    total = len(combos)

    output_dir = config_overrides.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    sweep_path = os.path.join(output_dir, f"{dataset}_sweep.json")

    # Resume from partial results if they exist
    dev_results = []
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            dev_results = json.load(f)
        if dev_results:
            print(f"  Resuming from run {len(dev_results) + 1}/{total} (loaded {len(dev_results)} saved results)")

    for i, (threshold, top_k, alpha) in enumerate(combos, 1):
        if i <= len(dev_results):
            print(f"[Dev] Run {i}/{total}: threshold={threshold}, top_k={top_k}, alpha={alpha} (skipped - already done)")
            continue
        print(f"\n{'='*60}")
        print(f"[Dev] Run {i}/{total}: threshold={threshold}, top_k={top_k}, alpha={alpha}")
        print(f"{'='*60}")

        cfg = PipelineConfig(
            dataset=dataset,
            split="dev",
            similarity_threshold=threshold,
            top_k=top_k,
            pagerank_alpha=alpha,
            **config_overrides,
        )
        result = run(cfg, df=split.dev, embeddings=dev_embeddings)
        dev_results.append(result)

        with open(sweep_path, "w") as f:
            json.dump(dev_results, f, indent=2, default=str)

    # --- Select best config by mean semantic similarity on dev ---
    best_idx = max(
        range(len(dev_results)),
        key=lambda i: dev_results[i]["metrics"]["semantic_similarity"]["__mean__"],
    )
    best_dev = dev_results[best_idx]
    best_params = best_dev["config"]
    print(f"\n{'='*60}")
    print(f"Best dev config (run {best_idx + 1}/{total}):")
    print(f"  threshold={best_params['similarity_threshold']}, "
          f"top_k={best_params['top_k']}, alpha={best_params['pagerank_alpha']}")
    print(f"  Dev semantic similarity: {best_dev['metrics']['semantic_similarity']['__mean__']:.3f}")
    print(f"{'='*60}")

    # --- Final evaluation on test split ---
    print(f"\n{'='*60}")
    print(f"[Test] Final evaluation with best config")
    print(f"{'='*60}")

    test_cfg = PipelineConfig(
        dataset=dataset,
        split="test",
        similarity_threshold=best_params["similarity_threshold"],
        top_k=best_params["top_k"],
        pagerank_alpha=best_params["pagerank_alpha"],
        **config_overrides,
    )
    test_result = run(test_cfg, df=split.test)

    # --- Save combined output ---
    sweep_output = {
        "dev_sweep": dev_results,
        "best_config": {
            "similarity_threshold": best_params["similarity_threshold"],
            "top_k": best_params["top_k"],
            "pagerank_alpha": best_params["pagerank_alpha"],
        },
        "best_dev_metrics": best_dev["metrics"],
        "test_result": test_result,
    }

    results_path = os.path.join(output_dir, f"{dataset}_results.json")
    with open(results_path, "w") as f:
        json.dump(sweep_output, f, indent=2, default=str)
    print(f"\nDev+test results saved to {results_path}")

    return sweep_output


def run_full_sweep(
    datasets: list[str] | None = None,
    similarity_thresholds: list[float] = [0.3, 0.5, 0.7],
    top_ks: list[int] = [3, 5, 10],
    pagerank_alphas: list[float] = [0.5, 0.85, 0.95],
    **config_overrides,
) -> dict[str, dict]:
    """
    Run hyperparameter sweep with dev/test evaluation across all datasets.

    Args:
        datasets: List of dataset names. Defaults to all available datasets.
        similarity_thresholds: Thresholds to sweep.
        top_ks: Values of k to sweep.
        pagerank_alphas: Damping factors to sweep.
        **config_overrides: Any other PipelineConfig fields to override.

    Returns:
        dict mapping dataset name -> sweep output dict.
    """
    if datasets is None:
        datasets = list(LOADERS.keys())

    output_dir = config_overrides.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "full_sweep.json")

    # Resume from partial full_sweep.json if it exists
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        if all_results:
            print(f"  Resuming full sweep - already completed: {list(all_results.keys())}")

    for dataset in datasets:
        if dataset in all_results:
            print(f"\n{'#'*60}")
            print(f"# Dataset: {dataset} (skipped - already complete)")
            print(f"{'#'*60}")
            continue
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

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nFull sweep results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_full_sweep()
