"""
Load the Banking77 dataset from Hugging Face into a standardized DataFrame.

77 intent clusters (e.g., lost_or_stolen_card, transfer_fee_charged,
card_payment_fee_charged) with ~130 documents per cluster on average.

Requirements:
    pip install datasets pandas
"""

import os
import sys

import pandas as pd
from datasets import load_dataset

# Remove this script's directory from sys.path so the datasets library
# doesn't mistake this file for a dataset loading script.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)


def load_banking77(split: str = "train") -> pd.DataFrame:
    """
    Load Banking77 from Hugging Face and return a DataFrame.

    Args:
        split: Which split to load — "train" or "test".

    Returns:
        DataFrame with columns: text, label
    """
    ds = load_dataset("mteb/banking77", split=split)
    df = ds.to_pandas()

    df = df[["text", "label_text"]].reset_index(drop=True)
    df = df.rename(columns={"label_text": "label"})

    return df


if __name__ == "__main__":
    df = load_banking77()
    print(f"Total documents: {len(df)}")
    print(f"Number of clusters: {df['label'].nunique()}")
    print(f"\nDocuments per cluster:\n{df['label'].value_counts().sort_index()}")
    print(f"\nSample row:\n{df.iloc[0]}")
