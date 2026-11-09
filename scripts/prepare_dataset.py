"""Prepare benchmark dataset and sample demonstration images for Crop Disease Intelligence."""

from __future__ import annotations

import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
BENCHMARK_DIR = DATA_DIR / "benchmark_dataset"

# 15 canonical classes from the subset
SUBSET_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
]


def extract_and_prepare(
    parquet_path: Path = DATA_DIR / "test_subset.parquet",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract real PlantVillage images from parquet and construct stratified benchmark splits."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    print(f"Reading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = BENCHMARK_DIR / "train"
    val_dir = BENCHMARK_DIR / "val"
    test_dir = BENCHMARK_DIR / "test"

    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)
        for class_name in SUBSET_CLASS_NAMES:
            (d / class_name).mkdir(parents=True, exist_ok=True)

    records = []
    print("Extracting images and mapping classes...")
    for idx, row in df.iterrows():
        label_idx = int(row["label"])
        if label_idx >= len(SUBSET_CLASS_NAMES):
            continue
        class_name = SUBSET_CLASS_NAMES[label_idx]

        img_data = row["image"]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Save sample demonstration images (first 3 of each class)
        class_sample_dir = SAMPLE_DIR / class_name
        class_sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = class_sample_dir / f"sample_{idx:04d}.jpg"
        if len(list(class_sample_dir.glob("*.jpg"))) < 3:
            img.save(sample_path, "JPEG", quality=95)

        records.append(
            {
                "id": idx,
                "label": class_name,
                "class_index": label_idx,
                "img_bytes": img_bytes,
            }
        )

    records_df = pd.DataFrame(records)
    print(f"Total extracted samples: {len(records_df)}")

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    train_val, test = train_test_split(
        records_df,
        test_size=0.15,
        stratify=records_df["class_index"],
        random_state=seed,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.1765,  # 0.15 / 0.85 ~ 0.1765
        stratify=train_val["class_index"],
        random_state=seed,
    )

    print(f"Splits created: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Save images to disk
    def save_split_images(split_df: pd.DataFrame, target_dir: Path) -> None:
        for _, r in split_df.iterrows():
            img_path = target_dir / r["label"] / f"img_{r['id']:04d}.jpg"
            with open(img_path, "wb") as f:
                f.write(r["img_bytes"])

    print("Saving train split images...")
    save_split_images(train, train_dir)
    print("Saving validation split images...")
    save_split_images(val, val_dir)
    print("Saving held-out test split images...")
    save_split_images(test, test_dir)

    print("Dataset preparation complete.")
    return train, val, test


if __name__ == "__main__":
    extract_and_prepare()
