"""
Load the AG News dataset from Hugging Face into a standardized DataFrame.

4 clusters (World, Sports, Business, Sci/Tech)
with ~30,000 documents per cluster on average.

Requirements:
    pip install datasets pandas
"""

import pandas as pd
from datasets import load_dataset

# AG News uses integer labels 0-3; map them to readable names
_LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_ag_news(split: str = "train") -> pd.DataFrame:
    """
    Load AG News from Hugging Face and return a DataFrame.

    Args:
        split: Which split to load — "train" or "test".
               Use "train" for the larger portion (~120k rows).

    Returns:
        DataFrame with columns: text, label
    """
    ds = load_dataset("ag_news", split=split)
    df = ds.to_pandas()

    # Map integer labels to human-readable category names
    df["label"] = df["label"].map(_LABEL_MAP)

    # Keep only the columns we need
    df = df[["text", "label"]].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_ag_news()
    print(f"Total documents: {len(df)}")
    print(f"Number of clusters: {df['label'].nunique()}")
    print(f"\nDocuments per cluster:\n{df['label'].value_counts().sort_index()}")
    print(f"\nSample row:\n{df.iloc[0]}")
