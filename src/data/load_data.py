"""
Load and preprocess FakeNewsNet dataset.
Reads the raw CSV files and structures them for downstream processing.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def load_fakenewsnet(data_dir: str) -> pd.DataFrame:
    """Load all FakeNewsNet CSVs into a single DataFrame with labels."""
    files = {
        "politifact_fake": ("politifact", "fake"),
        "politifact_real": ("politifact", "real"),
        "gossipcop_fake": ("gossipcop", "fake"),
        "gossipcop_real": ("gossipcop", "real"),
    }

    dfs = []
    dataset_path = os.path.join(data_dir, "dataset")

    for filename, (source, label) in files.items():
        filepath = os.path.join(dataset_path, f"{filename}.csv")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        df = pd.read_csv(filepath)
        df["source"] = source
        df["label"] = label
        df["label_binary"] = 1 if label == "fake" else 0
        dfs.append(df)
        print(f"Loaded {filename}: {len(df)} articles")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal articles loaded: {len(combined)}")
    print(f"  Fake: {(combined['label'] == 'fake').sum()}")
    print(f"  Real: {(combined['label'] == 'real').sum()}")

    return combined


def parse_tweet_ids(tweet_ids_str: str) -> list:
    """Parse tab-separated tweet IDs from the CSV column."""
    if pd.isna(tweet_ids_str):
        return []
    return str(tweet_ids_str).strip().split("\t")


def get_cascade_sizes(df: pd.DataFrame) -> pd.Series:
    """Count number of tweet IDs per article (cascade size)."""
    return df["tweet_ids"].apply(lambda x: len(parse_tweet_ids(x)))


def split_by_source(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split dataset into PolitiFact and GossipCop subsets."""
    return {
        source: group for source, group in df.groupby("source")
    }


def create_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split preserving label distribution."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label_binary"],
        random_state=random_state,
    )
    return train_df, test_df


if __name__ == "__main__":
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "FakeNewsNet"
    )
    data_dir = os.path.abspath(data_dir)

    df = load_fakenewsnet(data_dir)

    # Show cascade size statistics
    df["cascade_size"] = get_cascade_sizes(df)
    print(f"\nCascade size stats:")
    print(df["cascade_size"].describe())

    # Show per-source breakdown
    for source, sub_df in split_by_source(df).items():
        print(f"\n{source}: {len(sub_df)} articles "
              f"(fake={sum(sub_df['label']=='fake')}, "
              f"real={sum(sub_df['label']=='real')})")
