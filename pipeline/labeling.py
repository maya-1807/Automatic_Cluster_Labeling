"""Stage 3: Generate cluster labels using Groq API."""

import os
import re
import time

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


def _parse_retry_seconds(error_msg: str) -> float:
    """Extract wait time from Groq rate-limit error message."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(error_msg))
    if match:
        return float(match.group(1))
    match = re.search(r"try again in (\d+)m([\d.]+)s", str(error_msg))
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    return 60.0


def generate_label(
    docs: list[dict],
    client: Groq,
    model: str = "llama-3.1-8b-instant",
    max_words: int = 300,
    max_retries: int = 5,
) -> str:
    """
    Generate a cluster label from representative documents using Groq.

    Args:
        docs: List of dicts with at least a "text" key.
        client: Groq client instance.
        model: Groq model name.
        max_words: Max words per document in the prompt.
        max_retries: Max retries on rate-limit errors.

    Returns:
        Generated label string.
    """
    documents_block = _format_documents(docs, max_words)
    prompt = LABEL_PROMPT_TEMPLATE.format(
        k=len(docs), documents_block=documents_block
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30,
            )
            text = response.choices[0].message.content
            if text is None:
                print(f"    ERROR: Groq returned empty response")
                return "[no label generated]"
            return text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait = _parse_retry_seconds(error_str)
                print(f"    Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    ERROR: {e}")
                return "[no label generated]"

    print(f"    ERROR: Max retries exceeded")
    return "[no label generated]"


def label_all_clusters(
    central_docs: dict[str, list[dict]],
    model: str = "llama-3.1-8b-instant",
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
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    labels = {}
    i = 0
    for cluster_label, docs in central_docs.items():
        generated = generate_label(docs, client, model=model, max_words=max_words)
        labels[cluster_label] = generated
        print(f"  '[{i+1}/{len(central_docs)}] {cluster_label}' -> '{generated}'")
        i += 1
    return labels
