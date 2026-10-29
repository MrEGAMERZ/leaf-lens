"""Utility functions and backward-compatible helpers for Crop Disease Intelligence."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.preprocessing import clean_label, load_image_rgb, resize_image


def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(dir_path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML configuration file."""
    p = Path(config_path)
    if not p.is_absolute():
        p = get_project_root() / p

    with open(p, "r", encoding="utf-8") as file:
        if p.suffix.lower() in [".yaml", ".yml"]:
            import yaml
            return yaml.safe_load(file)
        return json.load(file)


def save_json(data: Any, save_path: str | Path) -> None:
    """Save data structure as JSON file."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def ensure_best_model(config: dict[str, Any] | None = None) -> str:
    """Ensure best_model.keras exists at models/best_model.keras."""
    root = get_project_root()
    best_model_path = root / "models" / "best_model.keras"

    if best_model_path.exists():
        return str(best_model_path)

    if config and "source_model_path" in config:
        src = root / config["source_model_path"]
        if src.exists():
            shutil.copyfile(src, best_model_path)
            return str(best_model_path)

    raise FileNotFoundError(f"Missing model file at '{best_model_path}'.")


def load_class_names(class_names_path: str | Path | None = None) -> list[str]:
    """Load canonical list of class names."""
    if class_names_path is None:
        p = get_project_root() / "results" / "class_names.json"
    else:
        p = Path(class_names_path)
        if not p.is_absolute():
            p = get_project_root() / p

    with open(p, "r", encoding="utf-8") as file:
        return json.load(file)


def load_image_for_prediction(image: Any, image_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and resize image for prediction returning [1, H, W, 3] float32 array."""
    img_rgb = load_image_rgb(image)
    resized = resize_image(img_rgb, image_size)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def get_top_predictions(
    predictions: np.ndarray,
    class_names: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Extract top-k ranked predictions from output probability vector."""
    scores = predictions[0] if predictions.ndim > 1 else predictions
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for rank, index in enumerate(top_indices, start=1):
        idx = int(index)
        label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        crop, condition = clean_label(label)
        confidence = float(scores[idx])
        results.append(
            {
                "rank": rank,
                "label": label,
                "crop": crop,
                "condition": condition,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100.0, 2),
            }
        )
    return results


def build_efficientnetb0_model(
    image_size: tuple[int, int] = (224, 224),
    num_classes: int = 38,
    base_trainable: bool = False,
) -> Any:
    """Backward-compatible helper to build EfficientNetB0 classification model."""
    from src.models.efficientnet import build_efficientnetb0
    model = build_efficientnetb0(num_classes=num_classes, image_size=image_size)
    if base_trainable:
        for layer in model.layers:
            layer.trainable = True
    return model


def load_image_datasets(
    dataset_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    use_valid_as_eval: bool = True,
) -> tuple[Any, Any, list[str]]:
    """Backward-compatible helper to load image datasets from train and valid directories."""
    import tensorflow as tf
    root = get_project_root()
    p = Path(dataset_path) if Path(dataset_path).is_absolute() else root / dataset_path

    train_ds = tf.keras.utils.image_dataset_from_directory(
        p / "train",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    class_names = train_ds.class_names

    eval_folder = "valid" if use_valid_as_eval else "test"
    val_ds = tf.keras.utils.image_dataset_from_directory(
        p / eval_folder,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_ds, val_ds, class_names
