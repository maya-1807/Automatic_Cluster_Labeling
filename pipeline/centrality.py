"""Stage 2: PageRank-based centrality selection of representative documents."""

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _pagerank_on_matrix(
    A: sp.csr_matrix,
    doc_indices: list[int],
    alpha: float = 0.85,
    max_iter: int = 1000,
    tol: float = 1.0e-6,
) -> dict:
    """
    PageRank via power iteration directly on a scipy sparse matrix.

    A[i,j] is the cosine similarity between doc i and doc j (0 if below threshold).
    Computes the column-stochastic transition matrix M = A / col_sums, then
    iterates r = alpha * M @ r + (1-alpha)/n until convergence.

    Returns:
        dict mapping doc_indices[i] -> pagerank score.
    """
    n = A.shape[0]

    # Column-normalize A to get the transition matrix M
    col_sums = np.asarray(A.sum(axis=0)).flatten()
    dangling = col_sums == 0
    col_sums[dangling] = 1.0  # avoid divide-by-zero; handled via dangling term
    M = A @ sp.diags(1.0 / col_sums)  # column-stochastic

    x = np.full(n, 1.0 / n)
    p = np.full(n, 1.0 / n)  # uniform teleportation

    for _ in range(max_iter):
        xlast = x.copy()
        dangling_sum = alpha * xlast[dangling].sum()
        x = alpha * M.dot(xlast) + (dangling_sum + (1.0 - alpha)) * p
        if np.abs(x - xlast).sum() < n * tol:
            break

    return dict(zip(doc_indices, x))


def select_central_documents(
    graph,
    df: pd.DataFrame,
    top_k: int = 10,
    pagerank_alpha: float = 0.85,
) -> list[dict]:
    """
    Run PageRank on a cluster graph and return the top-k most central documents.

    Args:
        graph: Either an nx.Graph (small clusters) or a
               (scipy.sparse.csr_matrix, doc_indices) tuple (large clusters).
        df: Full DataFrame (used to look up text by index).
        top_k: Number of central documents to select.
        pagerank_alpha: Damping factor for PageRank.

    Returns:
        List of dicts with keys: index, text, pagerank_score.
        Sorted by pagerank_score descending.
    """
    if isinstance(graph, tuple):
        sparse_A, doc_indices = graph
        scores = _pagerank_on_matrix(sparse_A, doc_indices, alpha=pagerank_alpha)
    else:
        scores = nx.pagerank(graph, alpha=pagerank_alpha, weight="weight", max_iter=1000)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            "index": doc_idx,
            "text": df.loc[doc_idx, "text"],
            "pagerank_score": score,
        }
        for doc_idx, score in ranked
    ]


def select_all_central_documents(
    graphs: dict,
    df: pd.DataFrame,
    top_k: int = 10,
    pagerank_alpha: float = 0.85,
) -> dict[str, list[dict]]:
    """
    Apply centrality selection across all clusters.

    Returns:
        dict mapping label -> list of top-k document dicts.
    """
    results = {}
    for label, graph in graphs.items():
        results[label] = select_central_documents(
            graph, df, top_k, pagerank_alpha
        )
    return results
