"""Centralized Configuration Manager for Crop Disease Intelligence Platform."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the absolute path of the project root."""
    return Path(__file__).resolve().parent.parent


@dataclass
class DatasetConfig:
    num_classes: int = 38
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    channels: int = 3
    supported_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    )
    validation_split_ratio: float = 0.15
    test_split_ratio: float = 0.15


@dataclass
class PathsConfig:
    dataset_root: str = "data/new_plant_diseases_dataset"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"
    models_dir: str = "models"
    results_dir: str = "results"
    metrics_dir: str = "results/metrics"
    figures_dir: str = "results/figures"
    reports_dir: str = "results/reports"
    class_names_path: str = "results/class_names.json"
    best_model_path: str = "models/best_model.keras"


@dataclass
class InferenceConfig:
    confidence_threshold_high: float = 0.85
    confidence_threshold_medium: float = 0.60
    uncertainty_warning_threshold: float = 0.60
    top_k: int = 3


@dataclass
class ExplainabilityConfig:
    alpha: float = 0.35
    colormap: str = "jet"


@dataclass
class AppConfig:
    project_name: str = "Crop Disease Intelligence"
    version: str = "2.0.0"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    raw_config: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, relative_or_absolute_path: str | Path) -> Path:
        """Resolve a path against the project root if it is relative."""
        p = Path(relative_or_absolute_path)
        if p.is_absolute():
            return p
        return get_project_root() / p


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML file or defaults."""
    root = get_project_root()
    if config_path is None:
        default_yaml = root / "configs" / "config.yaml"
        if default_yaml.exists():
            config_path = default_yaml

    raw: dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    paths_data = raw.get("paths", {})
    paths = PathsConfig(
        dataset_root=paths_data.get("dataset_root", "data/new_plant_diseases_dataset"),
        train_split=paths_data.get("train_split", "train"),
        val_split=paths_data.get("val_split", "valid"),
        test_split=paths_data.get("test_split", "test"),
        models_dir=paths_data.get("models_dir", "models"),
        results_dir=paths_data.get("results_dir", "results"),
        metrics_dir=paths_data.get("metrics_dir", "results/metrics"),
        figures_dir=paths_data.get("figures_dir", "results/figures"),
        reports_dir=paths_data.get("reports_dir", "results/reports"),
        class_names_path=paths_data.get("class_names_path", "results/class_names.json"),
        best_model_path=paths_data.get("best_model_path", "models/best_model.keras"),
    )

    dataset_data = raw.get("dataset", {})
    img_size_raw = dataset_data.get("image_size", [224, 224])
    image_size = (int(img_size_raw[0]), int(img_size_raw[1]))
    dataset = DatasetConfig(
        num_classes=int(dataset_data.get("num_classes", 38)),
        image_size=image_size,
        batch_size=int(dataset_data.get("batch_size", 32)),
        channels=int(dataset_data.get("channels", 3)),
        supported_extensions=dataset_data.get(
            "supported_extensions",
            [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
        ),
        validation_split_ratio=float(dataset_data.get("validation_split_ratio", 0.15)),
        test_split_ratio=float(dataset_data.get("test_split_ratio", 0.15)),
    )

    inf_data = raw.get("inference", {})
    inference = InferenceConfig(
        confidence_threshold_high=float(inf_data.get("confidence_threshold_high", 0.85)),
        confidence_threshold_medium=float(inf_data.get("confidence_threshold_medium", 0.60)),
        uncertainty_warning_threshold=float(
            os.getenv("UNCERTAINTY_THRESHOLD", inf_data.get("uncertainty_warning_threshold", 0.60))
        ),
        top_k=int(inf_data.get("top_k", 3)),
    )

    exp_data = raw.get("explainability", {}).get("gradcam", {})
    explainability = ExplainabilityConfig(
        alpha=float(exp_data.get("alpha", 0.35)),
        colormap=str(exp_data.get("colormap", "jet")),
    )

    return AppConfig(
        project_name=raw.get("project_name", "Crop Disease Intelligence"),
        version=raw.get("version", "2.0.0"),
        seed=int(raw.get("seed", 42)),
        paths=paths,
        dataset=dataset,
        inference=inference,
        explainability=explainability,
        raw_config=raw,
    )
