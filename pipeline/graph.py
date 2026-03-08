"""Stage 1b: Build per-cluster similarity graphs."""

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp

# Clusters with more docs than this are stored as (csr_matrix, doc_indices)
# instead of nx.Graph to avoid ~12 GB of Python dict overhead at 57M edges.
_LARGE_CLUSTER_THRESHOLD = 10_000


def build_cluster_graph(
    embeddings: np.ndarray,
    doc_indices: list[int],
    similarity_threshold: float = 0.5,
    chunk_size: int = 1000,
):
    """
    Build a weighted similarity graph for a single cluster.

    Returns:
        nx.Graph for small clusters.
        (scipy.sparse.csr_matrix, doc_indices) for large clusters — the
        matrix A where A[i,j] = cosine similarity if >= threshold, else 0.
        Matrix indices are positions in doc_indices (0..n-1).
    """
    n = len(embeddings)
    if n > _LARGE_CLUSTER_THRESHOLD:
        return _build_sparse(embeddings, doc_indices, similarity_threshold, chunk_size)

    G = nx.Graph()
    G.add_nodes_from(doc_indices)

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        sim_block = embeddings[i_start:i_end] @ embeddings[i_start:].T

        rows, cols = np.where(sim_block >= similarity_threshold)
        for r, c in zip(rows, cols):
            abs_i = i_start + r
            abs_j = i_start + c
            if abs_i < abs_j:
                G.add_edge(
                    doc_indices[abs_i],
                    doc_indices[abs_j],
                    weight=float(sim_block[r, c]),
                )

    return G


def _build_sparse(
    embeddings: np.ndarray,
    doc_indices: list[int],
    similarity_threshold: float,
    chunk_size: int,
) -> tuple:
    """
    Build a symmetric CSR sparse matrix directly from chunked dot products.
    Skips nx.Graph entirely — edges are collected as numpy arrays and
    assembled into a CSR matrix in one shot.
    """
    n = len(embeddings)
    chunk_rows, chunk_cols, chunk_weights = [], [], []

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # sim_block[r, c] = similarity between embeddings[i_start+r] and embeddings[i_start+c]
        sim_block = embeddings[i_start:i_end] @ embeddings[i_start:].T

        r, c = np.where(sim_block >= similarity_threshold)
        abs_i = i_start + r  # global index into embeddings (0..n-1)
        abs_j = i_start + c  # global index into embeddings (i_start..n-1)
        mask = abs_i < abs_j  # upper triangle only, no self-loops

        chunk_rows.append(abs_i[mask])
        chunk_cols.append(abs_j[mask])
        chunk_weights.append(sim_block[r[mask], c[mask]])

    if chunk_rows:
        rows = np.concatenate(chunk_rows)
        cols = np.concatenate(chunk_cols)
        weights = np.concatenate(chunk_weights)
    else:
        rows = cols = np.array([], dtype=np.int32)
        weights = np.array([], dtype=np.float32)

    # Build symmetric CSR (upper + lower triangle)
    all_r = np.concatenate([rows, cols])
    all_c = np.concatenate([cols, rows])
    all_w = np.concatenate([weights, weights])
    A = sp.csr_matrix((all_w, (all_r, all_c)), shape=(n, n))

    return A, list(doc_indices)


def _graph_edge_count(g) -> int:
    if isinstance(g, tuple):
        return g[0].nnz // 2  # symmetric matrix, each edge stored twice
    return g.number_of_edges()


def _graph_node_count(g) -> int:
    if isinstance(g, tuple):
        return g[0].shape[0]
    return g.number_of_nodes()


_graph_cache: dict[tuple, dict[str, object]] = {}


def build_all_graphs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.5,
    chunk_size: int = 1000,
) -> dict[str, object]:
    """
    Build one graph per unique label in the DataFrame.
    Results are cached in memory by (threshold, chunk_size) so repeated
    calls with the same parameters skip the expensive graph construction.

    Returns:
        dict mapping label -> nx.Graph or (csr_matrix, doc_indices).
    """
    cache_key = (id(embeddings), similarity_threshold, chunk_size)
    if cache_key in _graph_cache:
        graphs = _graph_cache[cache_key]
        for label, g in graphs.items():
            print(
                f"  Cluster '{label}': {_graph_node_count(g)} docs, "
                f"{_graph_edge_count(g)} edges (cached)"
            )
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
            f"{_graph_edge_count(graphs[label])} edges"
        )

    _graph_cache[cache_key] = graphs
    return graphs
