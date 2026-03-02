# Automatic Cluster Labeling via Graph-Guided RAG

Automatically generates descriptive labels for document clusters using a graph-based retrieval-augmented generation (RAG) approach. The pipeline builds a similarity graph per cluster, uses PageRank to select the most representative documents, and prompts an LLM to generate a concise label.

## Pipeline Overview

```
Documents ──> Embeddings ──> Similarity Graphs ──> PageRank ──> Top-k Docs ──> LLM Label
                (Stage 1a)      (Stage 1b)         (Stage 2)                  (Stage 3)
```

1. **Embed** documents using `all-MiniLM-L6-v2` (sentence-transformers)
2. **Build graphs** — connect documents within each cluster whose cosine similarity exceeds a threshold
3. **Select central documents** — run weighted PageRank and pick the top-k highest-scoring documents
4. **Generate labels** — prompt Groq (`llama-3.1-8b-instant`) with the top-k documents to produce a short cluster label
5. **Evaluate** — compare generated labels to ground-truth using semantic similarity and token-overlap F1

## Project Structure

```
Automatic_Cluster_Labeling/
├── config.py                     # Pipeline configuration (dataclass)
├── run_pipeline.py               # Main entry point & hyperparameter sweep
├── pipeline/
│   ├── embeddings.py             # Stage 1a: sentence-transformer encoding
│   ├── graph.py                  # Stage 1b: per-cluster similarity graphs
│   ├── centrality.py             # Stage 2: PageRank centrality selection
│   └── labeling.py               # Stage 3: LLM label generation (Groq)
├── evaluation/
│   └── metrics.py                # Semantic similarity & token-overlap F1
├── data_collection/
│   ├── 20newsgroups.py           # 20 Newsgroups loader (sklearn)
│   ├── ag_news.py                # AG News loader (HuggingFace)
│   ├── banking77.py              # Banking77 loader (HuggingFace)
│   ├── bbc_news.py               # BBC News loader (local CSV)
│   └── bbc_news.csv              # Pre-processed BBC News dataset
├── requirements.txt
└── .gitignore
```

## Datasets

| Dataset | Documents | Clusters |
|---------|-----------|----------|
| [BBC News](http://mlg.ucd.ie/datasets/bbc.html) | 2,225 | 5 |
| [Banking77](https://huggingface.co/datasets/mteb/banking77) | 13,083 | 77 |
| [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) | ~18,000 | 20 |
| [AG News](https://huggingface.co/datasets/sh0416/ag_news) | ~120,000 | 4 |

### BBC News

A collection of 2,225 news articles from the BBC News website, published between 2004-2005. Each document is a full news article. The dataset is divided into 5 topic categories: **business**, **entertainment**, **politics**, **sport**, and **tech**, with roughly 445 articles per category. Loaded from a pre-processed local CSV file (originally sourced from raw text files).

### Banking77

A dataset of 13,083 customer service queries from the banking domain, designed for intent detection. Each document is a short user message (e.g., *"I didn't receive my card yet"*, *"Why was I charged a fee?"*). The queries are categorized into 77 fine-grained intent clusters such as **card_arrival**, **card_payment_fee_charged**, **lost_or_stolen_card**, **transfer_fee_charged**, and **activate_my_card**, with roughly 170 queries per intent.

### 20 Newsgroups

A classic text classification benchmark containing approximately 18,000 posts from 20 different Usenet newsgroups. Each document is the body of a newsgroup message (headers, footers, and quotes are stripped). The 20 clusters correspond to newsgroup topics, spanning categories like computer science (**comp.graphics**, **comp.sys.mac.hardware**), science (**sci.med**, **sci.space**), recreation (**rec.sport.hockey**, **rec.autos**), politics (**talk.politics.mideast**, **talk.politics.guns**), and religion (**alt.atheism**, **soc.religion.christian**).

### AG News

A large-scale news classification dataset with approximately 120,000 news articles collected from over 2,000 news sources. Each document is a short news summary (1-2 sentences: title + description). The articles are classified into 4 broad categories: **World**, **Sports**, **Business**, and **Sci/Tech**, with roughly 30,000 articles per category.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Usage

### Run a full hyperparameter sweep (all datasets)

```bash
python run_pipeline.py
```

This sweeps over all combinations of:
- `similarity_threshold`: [0.3, 0.5, 0.7]
- `top_k`: [3, 5, 10]
- `pagerank_alpha`: [0.5, 0.85, 0.95]

across all 4 datasets (108 total runs). Results are saved incrementally to `results/`.

### Run a single dataset

```python
from config import PipelineConfig
from run_pipeline import run

result = run(PipelineConfig(dataset="bbc_news"))
```

### Run a sweep on one dataset

```python
from run_pipeline import run_hyperparameter_sweep

results = run_hyperparameter_sweep(dataset="bbc_news")
```

## Configuration

All parameters are defined in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `"bbc_news"` | Dataset to use |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence-transformer model |
| `embedding_device` | `"cpu"` | Device for embeddings (`"cpu"` or `"cuda"`) |
| `similarity_threshold` | `0.5` | Min cosine similarity for graph edges |
| `chunk_size` | `1000` | Rows per chunk in similarity computation |
| `top_k` | `10` | Number of central documents to select |
| `pagerank_alpha` | `0.85` | PageRank damping factor |
| `labeling_model` | `"llama-3.1-8b-instant"` | Groq model for label generation |
| `max_doc_words` | `300` | Max words per document in the LLM prompt |

## Evaluation Metrics

- **Semantic Similarity**: Cosine similarity between sentence-transformer embeddings of ground-truth and generated labels
- **Token Overlap F1**: Lexical F1 score between tokenized ground-truth and generated labels

## Output

Results are saved as JSON files in `results/`:
- `{dataset}_sweep.json` — per-dataset sweep results
- `full_sweep.json` — combined results across all datasets

Each result includes the configuration used, generated labels, and evaluation metrics.
