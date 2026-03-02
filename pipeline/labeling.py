"""Stage 3: Generate cluster labels using Groq API."""

import os

from groq import Groq

LABEL_PROMPT_TEMPLATE = """You are an expert at identifying the common theme of a group of documents.

Below are {k} representative documents from a single cluster. These documents were selected because they are the most central/representative of the cluster.

{documents_block}

Based on these documents, provide a short, descriptive label for this cluster.

Rules:
- The label should be 1-4 words long
- The label should capture the primary shared topic or theme
- Be specific rather than generic (e.g., "US Foreign Policy" not "Politics")
- Return ONLY the label, nothing else"""


def _format_documents(docs: list[dict], max_words: int = 300) -> str:
    """Format top-k documents into a numbered text block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        words = doc["text"].split()
        truncated = " ".join(words[:max_words])
        if len(words) > max_words:
            truncated += "..."
        parts.append(f"Document {i}:\n{truncated}")
    return "\n\n".join(parts)


def generate_label(
    docs: list[dict],
    model: str = "llama-3.3-70b-versatile",
    max_words: int = 300,
) -> str:
    """
    Generate a cluster label from representative documents using Groq.

    Args:
        docs: List of dicts with at least a "text" key.
        model: Groq model name.
        max_words: Max words per document in the prompt.

    Returns:
        Generated label string.
    """
    documents_block = _format_documents(docs, max_words)
    prompt = LABEL_PROMPT_TEMPLATE.format(
        k=len(docs), documents_block=documents_block
    )

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30,
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        return "[no label generated]"

    text = response.choices[0].message.content
    if text is None:
        print(f"    ERROR: Groq returned empty response")
        return "[no label generated]"
    return text.strip()


def label_all_clusters(
    central_docs: dict[str, list[dict]],
    model: str = "llama-3.3-70b-versatile",
    max_words: int = 300,
) -> dict[str, str]:
    """
    Generate labels for all clusters.

    Args:
        central_docs: dict mapping ground-truth label -> list of top-k doc dicts.
        model: Groq model name.
        max_words: Max words per document in the prompt.

    Returns:
        dict mapping ground-truth label -> generated label.
    """
    labels = {}
    for cluster_label, docs in central_docs.items():
        generated = generate_label(docs, model=model, max_words=max_words)
        labels[cluster_label] = generated
        print(f"  '{cluster_label}' -> '{generated}'")
    return labels
