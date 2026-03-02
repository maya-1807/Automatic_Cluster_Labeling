"""
Load the BBC News dataset from a pre-processed CSV file.

5 clusters (business, entertainment, politics, sport, tech)
with ~445 documents per cluster on average.

Source: http://mlg.ucd.ie/datasets/bbc.html

The CSV was generated from the raw text files in bbc.zip.
To regenerate it:
    1. unzip data_collection/bbc.zip -d data_collection/
    2. python data_collection/bbc_news.py --rebuild

Requirements:
    pip install pandas
"""

import argparse
from pathlib import Path

import pandas as pd

_CSV_PATH = Path(__file__).resolve().parent / "bbc_news.csv"
_RAW_DIR = Path(__file__).resolve().parent / "bbc"


def _rebuild_csv() -> pd.DataFrame:
    """Read raw text files from bbc/ and save to bbc_news.csv."""
    if not _RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"{_RAW_DIR} not found. Unzip bbc.zip first:\n"
            f"  unzip data_collection/bbc.zip -d data_collection/"
        )

    rows = []
    for category_dir in sorted(_RAW_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        label = category_dir.name
        for txt_file in sorted(category_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)[["text", "label"]].reset_index(drop=True)
    df.to_csv(_CSV_PATH, index=False)
    print(f"Saved {len(df)} rows to {_CSV_PATH}")
    return df


def load_bbc_news() -> pd.DataFrame:
    """
    Load the BBC News dataset and return a DataFrame with columns: text, label.

    Returns:
        DataFrame with columns: text, label
    """
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"{_CSV_PATH} not found. Rebuild it from the raw data:\n"
            f"  1. unzip data_collection/bbc.zip -d data_collection/\n"
            f"  2. python data_collection/bbc_news.py --rebuild"
        )
    return pd.read_csv(_CSV_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild CSV from raw text files in bbc/")
    args = parser.parse_args()

    if args.rebuild:
        df = _rebuild_csv()
    else:
        df = load_bbc_news()

    print(f"Total documents: {len(df)}")
    print(f"Number of clusters: {df['label'].nunique()}")
    print(f"\nDocuments per cluster:\n{df['label'].value_counts().sort_index()}")
    print(f"\nSample row:\n{df.iloc[0]}")
