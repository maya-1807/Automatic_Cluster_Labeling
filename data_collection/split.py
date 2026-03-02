"""Stratified dev/test split for any dataset DataFrame."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    """Container for a stratified dev/test split."""

    dev: pd.DataFrame
    test: pd.DataFrame
    dataset_name: str


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_name: str = "",
) -> DatasetSplit:
    """
    Split a DataFrame into dev and test sets, stratified by the 'label' column.

    Args:
        df: DataFrame with columns [text, label].
        test_size: Fraction of data to hold out for testing.
        random_state: Seed for reproducibility.
        dataset_name: Name of the dataset (for display purposes).

    Returns:
        DatasetSplit with .dev and .test DataFrames (reset indices).
    """
    dev_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    return DatasetSplit(
        dev=dev_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        dataset_name=dataset_name,
    )
