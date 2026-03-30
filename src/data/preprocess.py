"""
Text and data preprocessing pipeline for FakeNewsNet.
Handles text cleaning, normalization, and quality filtering.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional


def clean_text(text: str) -> str:
    """Clean and normalize text while preserving structure."""
    if not isinstance(text, str):
        return ""

    # Normalize URLs to [URL] token
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # Normalize mentions to [USER] token
    text = re.sub(r"@\w+", "[USER]", text)

    # Fix encoding issues
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_quality_filters(
    df: pd.DataFrame,
    min_text_length: int = 10,
    max_text_length: int = 1000,
    min_cascade_size: int = 10,
) -> pd.DataFrame:
    """Apply quality filters to the dataset."""
    initial_count = len(df)

    # Filter by title length
    if "title" in df.columns:
        df = df[df["title"].apply(lambda x: isinstance(x, str) and len(x) >= min_text_length)]

    # Filter by cascade size if available
    if "cascade_size" in df.columns:
        df = df[df["cascade_size"] >= min_cascade_size]

    filtered_count = len(df)
    print(f"Quality filtering: {initial_count} -> {filtered_count} "
          f"(removed {initial_count - filtered_count})")

    return df.reset_index(drop=True)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    # Clean title text
    if "title" in df.columns:
        df["title_clean"] = df["title"].apply(clean_text)

    # Clean news URL
    if "news_url" in df.columns:
        df["domain"] = df["news_url"].apply(extract_domain)

    return df


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not isinstance(url, str):
        return "unknown"
    # Remove protocol
    domain = re.sub(r"^https?://", "", url)
    # Remove path
    domain = domain.split("/")[0]
    # Remove www
    domain = re.sub(r"^www\.", "", domain)
    return domain


if __name__ == "__main__":
    from load_data import load_fakenewsnet, get_cascade_sizes
    import os

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "FakeNewsNet")
    )

    df = load_fakenewsnet(data_dir)
    df["cascade_size"] = get_cascade_sizes(df)
    df = preprocess_dataset(df)
    df = apply_quality_filters(df)

    print(f"\nFinal dataset: {len(df)} articles")
    print(df[["id", "title_clean", "source", "label", "cascade_size"]].head(10))
