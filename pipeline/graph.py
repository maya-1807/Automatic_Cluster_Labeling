"""Stage 1b: Build per-cluster similarity graphs."""

import numpy as np
import networkx as nx
import pandas as pd


def build_cluster_graph(
    embeddings: np.ndarray,
    doc_indices: list[int],
    similarity_threshold: float = 0.5,
    chunk_size: int = 1000,
) -> nx.Graph:
    """
    Build a weighted undirected graph for a single cluster.

    Edges connect documents whose cosine similarity >= threshold.
    Uses chunked computation to handle large clusters efficiently.

    Args:
        embeddings: Pre-normalized embedding vectors for this cluster.
                    Shape (n_docs, embedding_dim).
        doc_indices: Original DataFrame row indices (used as node IDs).
        similarity_threshold: Minimum cosine similarity to create an edge.
        chunk_size: Rows per chunk in similarity computation.

    Returns:
        nx.Graph with doc_indices as nodes and cosine similarity as edge weight.
    """
    G = nx.Graph()
    G.add_nodes_from(doc_indices)
    n = len(embeddings)

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # Dot product of chunk rows against all rows from i_start onward
        # (upper triangle only to avoid duplicate edges)
        sim_block = embeddings[i_start:i_end] @ embeddings[i_start:].T

        rows, cols = np.where(sim_block >= similarity_threshold)
        for r, c in zip(rows, cols):
            abs_i = i_start + r
            abs_j = i_start + c
            if abs_i < abs_j:  # upper triangle, skip self-loops
                G.add_edge(
                    doc_indices[abs_i],
                    doc_indices[abs_j],
                    weight=float(sim_block[r, c]),
                )

    return G


_graph_cache: dict[tuple, dict[str, nx.Graph]] = {}


def build_all_graphs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.5,
    chunk_size: int = 1000,
) -> dict[str, nx.Graph]:
    """
    Build one graph per unique label in the DataFrame.
    Results are cached in memory by (threshold, chunk_size) so repeated
    calls with the same parameters (e.g. during a hyperparameter sweep)
    skip the expensive graph construction.

    Args:
        df: DataFrame with columns [text, label].
        embeddings: Embedding array aligned with df rows.
        similarity_threshold: Minimum cosine similarity for edges.
        chunk_size: Rows per chunk in similarity computation.

    Returns:
        dict mapping label -> nx.Graph.
    """
    cache_key = (id(embeddings), similarity_threshold, chunk_size)
    if cache_key in _graph_cache:
        graphs = _graph_cache[cache_key]
        for label, g in graphs.items():
            print(f"  Cluster '{label}': {g.number_of_nodes()} docs, {g.number_of_edges()} edges (cached)")
        return graphs

    graphs = {}
    for label, group in df.groupby("label"):
        idx = group.index.tolist()
        cluster_embs = embeddings[idx]
        graphs[label] = build_cluster_graph(
            cluster_embs, idx, similarity_threshold, chunk_size
        )
        print(
            f"  Cluster '{label}': {len(idx)} docs, "
            f"{graphs[label].number_of_edges()} edges"
        )

    _graph_cache[cache_key] = graphs
    return graphs
