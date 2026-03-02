"""Stage 2: PageRank-based centrality selection of representative documents."""

import networkx as nx
import pandas as pd


def select_central_documents(
    graph: nx.Graph,
    df: pd.DataFrame,
    top_k: int = 10,
    pagerank_alpha: float = 0.85,
) -> list[dict]:
    """
    Run PageRank on a cluster graph and return the top-k most central documents.

    Args:
        graph: Weighted cluster graph from Stage 1.
        df: Full DataFrame (used to look up text by index).
        top_k: Number of central documents to select.
        pagerank_alpha: Damping factor for PageRank.

    Returns:
        List of dicts with keys: index, text, pagerank_score.
        Sorted by pagerank_score descending.
    """
    scores = nx.pagerank(graph, alpha=pagerank_alpha, weight="weight")
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
    graphs: dict[str, nx.Graph],
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
