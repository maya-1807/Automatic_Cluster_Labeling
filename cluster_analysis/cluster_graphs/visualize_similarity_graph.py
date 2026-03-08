"""Visualize a document similarity graph from a DataFrame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `pipeline` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


from pipeline.embeddings import embed_documents

LOADERS = {
    "bbc_news": ("data_collection.bbc_news", "load_bbc_news"),
    "banking77": ("data_collection.banking77", "load_banking77"),
    "20newsgroups": ("data_collection.20newsgroups", "load_20newsgroups"),
    "ag_news": ("data_collection.ag_news", "load_ag_news"),
}

def _load_dataset(name: str) -> pd.DataFrame:
    module_path, func_name = LOADERS[name]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()

def visualize_similarity_graph(
    df: pd.DataFrame,
    similarity_threshold: float = 0.5,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str | None = None,
    embeddings: np.ndarray | None = None,
    random_state: int = 42,
    figsize: tuple[int, int] = (14, 10),
    output_path: str | Path | None = None,
    show_plot: bool = True,
) -> tuple[nx.Graph, plt.Figure, plt.Axes]:
    """
    Build and visualize a document graph from a DataFrame with `label` and `text`.

    Nodes are documents. An undirected edge is added between two documents when
    their cosine similarity is at least `similarity_threshold`.

    Args:
        embeddings: Optional pre-computed L2-normalised embeddings. When
            provided, `model_name`, `batch_size`, and `device` are ignored.
    """
    required_columns = {"label", "text"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"DataFrame must contain columns {sorted(required_columns)}; "
            f"missing {sorted(missing_columns)}."
        )

    if df.empty:
        raise ValueError("DataFrame is empty.")

    graph_df = df.loc[:, ["label", "text"]].reset_index(drop=True).copy()
    texts = graph_df["text"].astype(str).tolist()

    if embeddings is None:
        embeddings = embed_documents(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )

    similarity_matrix = embeddings @ embeddings.T
    upper_i, upper_j = np.triu_indices(len(graph_df), k=1)
    edge_mask = similarity_matrix[upper_i, upper_j] >= similarity_threshold

    graph = nx.Graph()
    for idx, row in graph_df.iterrows():
        graph.add_node(idx, label=row["label"], text=row["text"])

    for i, j in zip(upper_i[edge_mask], upper_j[edge_mask]):
        graph.add_edge(i, j, weight=float(similarity_matrix[i, j]))

    unique_labels = sorted(graph_df["label"].astype(str).unique())
    cmap = plt.colormaps.get_cmap("tab20")
    color_lookup = {
        label: cmap(i % cmap.N)
        for i, label in enumerate(unique_labels)
    }
    node_colors = [color_lookup[str(graph.nodes[node]["label"])] for node in graph.nodes]

    fig, ax = plt.subplots(figsize=figsize)
    layout = nx.spring_layout(graph, seed=random_state, weight="weight")

    nx.draw_networkx_edges(
        graph,
        pos=layout,
        ax=ax,
        alpha=0.18,
        width=0.8,
        edge_color="#6b7280",
    )
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        ax=ax,
        node_color=node_colors,
        node_size=80,
        linewidths=0.3,
        edgecolors="white",
    )

    for label in unique_labels:
        ax.scatter([], [], color=color_lookup[label], label=label, s=80)

    ax.set_title(
        f"Document Similarity Graph (threshold={similarity_threshold}, "
        f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()})"
    )
    ax.legend(title="Label", loc="best", frameon=True)
    ax.set_axis_off()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return graph, fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a document similarity graph."
    )
    parser.add_argument("--dataset", default="bbc_news", choices=list(LOADERS))
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for adding graph edges.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional image output path. If omitted, the plot is only displayed.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory for cached embeddings (default: cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached embeddings and always recompute.",
    )
    args = parser.parse_args()

    df = _load_dataset(args.dataset)

    # Load embeddings from the pipeline cache if available, matching the
    # naming convention used by run_pipeline.py: {dataset}_{model}.npy
    embeddings = None
    cache_path = Path(args.cache_dir) / f"{args.dataset}_{args.model}.npy"
    if not args.no_cache and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
    else:
        print("Computing embeddings...")

    visualize_similarity_graph(
        df=df,
        similarity_threshold=args.threshold,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        embeddings=embeddings,
        output_path=f"cluster_analysis/cluster_graphs/{args.dataset}.png",
        show_plot=not args.no_show,
    )

def create_all_graphs() -> None:
    for dataset in LOADERS.keys():
        df = _load_dataset(dataset)
        
        # Load embeddings from the pipeline cache if available, matching the
        # naming convention used by run_pipeline.py: {dataset}_{model}.npy
        embeddings = None
        cache_path = Path("cache") / f"{dataset}_all-MiniLM-L6-v2.npy"
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
        else:
            print("Computing embeddings...")

        visualize_similarity_graph(
            df=df,
            similarity_threshold=0.5,
            model_name="all-MiniLM-L6-v2",
            batch_size=256,
            device=None,
            embeddings=embeddings,
            output_path=f"cluster_analysis/cluster_graphs/{dataset}.png",
            show_plot=False,
        )


if __name__ == "__main__":
    #main()
    create_all_graphs()