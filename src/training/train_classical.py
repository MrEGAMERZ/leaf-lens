"""Classical Machine Learning Training and Benchmarking Pipeline.

Trains and evaluates:
1. Support Vector Machine (SVM)
2. Random Forest Classifier
3. XGBoost Classifier

Uses unified feature extraction (Color Histograms, HOG, GLCM) and fits scalers
strictly on training data to prevent data leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import scan_dataset_directory
from src.data.preprocessing import load_image_rgb
from src.evaluation.benchmark import (
    compute_evaluation_metrics,
    measure_inference_latency,
    plot_and_save_confusion_matrix,
    save_benchmark_results,
)
from src.features.feature_pipeline import (
    create_classical_feature_pipeline,
    save_feature_pipeline,
)
from src.models.classical_ml import (
    build_random_forest_classifier,
    build_svm_classifier,
    build_xgboost_classifier,
    save_classical_model,
)

ROOT = Path(__file__).resolve().parents[2]


def load_images_and_labels(
    split_dir: Path,
    class_names: list[str],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load all images as RGB numpy arrays and numeric labels."""
    df = scan_dataset_directory(split_dir, expected_classes=class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    images: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        try:
            img = load_image_rgb(row["filepath"])
            images.append(img)
            labels.append(class_to_idx[row["label"]])
        except Exception:
            continue

    return images, np.array(labels, dtype=int)


def train_and_evaluate_classical_models(
    data_dir: Path = ROOT / "data" / "benchmark_dataset",
    output_models_dir: Path = ROOT / "models",
    output_results_dir: Path = ROOT / "results",
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Train SVM, Random Forest, and XGBoost on train split, evaluate on held-out test split."""
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    output_models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_results_dir / "metrics"
    figures_dir = output_results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Class names from train directory subfolders
    class_folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_names = class_folders
    print(f"Benchmarking on {len(class_names)} classes.")

    print("Loading training images...")
    X_train_raw, y_train = load_images_and_labels(train_dir, class_names)
    print(f"Loaded {len(X_train_raw)} training images.")

    print("Loading test images (held-out evaluation set)...")
    X_test_raw, y_test = load_images_and_labels(test_dir, class_names)
    print(f"Loaded {len(X_test_raw)} held-out test images.")

    # 1. Fit Feature Pipeline strictly on training data
    print("Extracting features and fitting feature pipeline (train split only)...")
    feature_pipeline = create_classical_feature_pipeline()
    X_train_features = feature_pipeline.fit_transform(X_train_raw)
    pipeline_path = output_models_dir / "classical_feature_pipeline.joblib"
    save_feature_pipeline(feature_pipeline, pipeline_path)
    print(f"Feature dimension: {X_train_features.shape[1]}. Saved pipeline to {pipeline_path}")

    # Transform test set using fitted pipeline
    print("Transforming test set using fitted pipeline...")
    X_test_features = feature_pipeline.transform(X_test_raw)

    results: dict[str, dict[str, Any]] = {}

    # Define model configurations
    models_to_train = [
        ("SVM", build_svm_classifier(kernel="rbf", C=10.0, probability=True, random_state=seed), "svm_model.joblib"),
        ("Random Forest", build_random_forest_classifier(n_estimators=150, max_depth=25, random_state=seed), "random_forest_model.joblib"),
        ("XGBoost", build_xgboost_classifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=seed), "xgboost_model.joblib"),
    ]

    for model_name, model, artifact_filename in models_to_train:
        print(f"\n--- Training {model_name} ---")
        model.fit(X_train_features, y_train)

        artifact_path = output_models_dir / artifact_filename
        save_classical_model(model, artifact_path)
        print(f"Saved {model_name} artifact to {artifact_path}")

        print(f"Evaluating {model_name} on held-out test set...")
        y_pred = model.predict(X_test_features)

        # Measure latency on test sample
        sample_img = [X_test_raw[0]]
        latency = measure_inference_latency(
            model,
            sample_img,
            is_cnn=False,
            feature_pipeline=feature_pipeline,
            num_warmup=5,
            num_runs=30,
        )

        metrics = compute_evaluation_metrics(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_name=model_name,
            model_path=artifact_path,
            latency_stats=latency,
        )

        # Save metrics JSON & classification report CSV
        save_benchmark_results(metrics, metrics_dir)

        # Plot & save confusion matrix
        cm_path = figures_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plot_and_save_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            class_names,
            cm_path,
            title=f"{model_name} Confusion Matrix",
        )

        results[model_name] = metrics
        print(
            f"Results for {model_name}: Accuracy={metrics['accuracy']:.4f}, "
            f"Macro F1={metrics['macro_f1']:.4f}, Latency={metrics['latency']['mean_latency_ms']:.2f}ms"
        )

    return results


if __name__ == "__main__":
    train_and_evaluate_classical_models()
