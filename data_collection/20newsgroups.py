"""
Load the 20 Newsgroups dataset into a standardized DataFrame.

20 clusters (e.g., sci.med, rec.sport.hockey, comp.graphics, talk.politics.mideast)
with ~942 documents per cluster on average.

Requirements:
    pip install scikit-learn pandas
"""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


def load_20newsgroups(subset: str = "all", remove_headers: bool = True) -> pd.DataFrame:
    """
    Fetch the 20 Newsgroups dataset from sklearn and return a DataFrame.

    Args:
        subset: Which portion to load — "train", "test", or "all".
        remove_headers: If True, strip mail headers/footers/quotes so the model
                        sees only the body text.

    Returns:
        DataFrame with columns: text, label
    """
    remove = ("headers", "footers", "quotes") if remove_headers else ()

    data = fetch_20newsgroups(subset=subset, remove=remove, shuffle=False)

    # Map numeric target ids back to human-readable newsgroup names
    label_names = [data.target_names[i] for i in data.target]

    df = pd.DataFrame({"text": data.data, "label": label_names})

    # Drop rows where the body is empty after header removal
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_20newsgroups()
    print(f"Total documents: {len(df)}")
    print(f"Number of clusters: {df['label'].nunique()}")
    print(f"\nDocuments per cluster:\n{df['label'].value_counts().sort_index()}")
    print(f"\nSample row:\n{df.iloc[0]}")
