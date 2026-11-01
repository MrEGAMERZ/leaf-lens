"""Dataset loading, validation, and stratified splitting for Crop Disease Intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import (
    SUPPORTED_EXTENSIONS,
    clean_label,
    load_image_rgb,
    validate_image_file,
)


def load_canonical_class_names(class_names_path: str | Path) -> list[str]:
    """Load canonical 38 class names list from JSON."""
    p = Path(class_names_path)
    if not p.exists():
        raise FileNotFoundError(f"Class names file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_dataset_directory(
    dataset_dir: str | Path,
    expected_classes: list[str] | None = None,
) -> pd.DataFrame:
    """Scan a dataset directory organized as class subfolders.

    Returns:
        pd.DataFrame with columns: ['filepath', 'label', 'crop', 'condition', 'valid']
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records = []
    class_folders = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for folder in sorted(class_folders, key=lambda x: x.name):
        label = folder.name
        if expected_classes and label not in expected_classes:
            continue
        crop, condition = clean_label(label)

        for img_file in folder.iterdir():
            if img_file.suffix.lower() in [ext.lower() for ext in SUPPORTED_EXTENSIONS]:
                records.append(
                    {
                        "filepath": str(img_file.resolve()),
                        "label": label,
                        "crop": crop,
                        "condition": condition,
                    }
                )

    df = pd.DataFrame(records)
    return df


def audit_dataset_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """Validate image readability, detect corrupt files, and summarize class distribution."""
    total_images = len(df)
    corrupt_files = []
    valid_count = 0

    for path_str in df["filepath"]:
        try:
            validate_image_file(path_str)
            valid_count += 1
        except Exception as e:
            corrupt_files.append({"filepath": path_str, "error": str(e)})

    class_counts = df["label"].value_counts().to_dict()
    crop_counts = df["crop"].value_counts().to_dict()

    return {
        "total_images": total_images,
        "valid_images": valid_count,
        "corrupt_images": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "class_distribution": class_counts,
        "crop_distribution": crop_counts,
        "imbalance_ratio": (
            max(class_counts.values()) / min(class_counts.values())
            if class_counts
            else 1.0
        ),
    }


def create_stratified_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train, validation, and test splits.

    Ensures zero data leakage across splits.
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column for stratified splitting.")

    # First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    # Then split train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
