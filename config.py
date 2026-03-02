"""Central configuration for the Graph Guided RAG pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # Dataset
    dataset: str = "bbc_news"  # 20newsgroups, ag_news, banking77, bbc_news

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    embedding_device: str | None = None  # None = auto-detect GPU

    # Graph construction
    similarity_threshold: float = 0.5
    chunk_size: int = 1000

    # Centrality selection
    top_k: int = 10
    pagerank_alpha: float = 0.85

    # Generative labeling (Groq)
    labeling_model: str = "llama-3.1-8b-instant"
    max_doc_words: int = 300

    # Caching
    cache_dir: str = "cache"
    use_cache: bool = True

    # Output
    output_dir: str = "results"
