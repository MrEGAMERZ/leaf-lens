"""Comprehensive Model Benchmarking and Evaluation Suite.

Computes:
- Overall Accuracy
- Macro & Weighted Precision
- Macro & Weighted Recall
- Macro & Weighted F1-Score
- Per-Class Classification Report
- Confusion Matrix
- Empirical Inference Latency (Mean, Std, P95)
- Serialized Model Disk Size (MB)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_model_size_mb(model_path: str | Path) -> float:
    """Calculate the file size of a saved model artifact in megabytes (MB)."""
    p = Path(model_path)
    if not p.exists():
        return 0.0
    return round(p.stat().st_size / (1024.0 * 1024.0), 2)


def measure_inference_latency(
    model: Any,
    sample_input: np.ndarray,
    is_cnn: bool = True,
    feature_pipeline: Any = None,
    num_warmup: int = 5,
    num_runs: int = 30,
) -> dict[str, float]:
    """Measure inference latency across warm-up and repeated benchmark runs.

    Returns:
        dict: mean_ms, std_ms, p95_ms, min_ms, max_ms
    """
    # Warm-up runs
    for _ in range(num_warmup):
        if is_cnn:
            _ = model.predict(sample_input, verbose=0)
        else:
            feat = feature_pipeline.transform(sample_input) if feature_pipeline else sample_input
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(feat)
            else:
                _ = model.predict(feat)

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        if is_cnn:
            _ = model.predict(sample_input, verbose=0)
        else:
            feat = feature_pipeline.transform(sample_input) if feature_pipeline else sample_input
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(feat)
            else:
                _ = model.predict(feat)
        latencies.append((time.perf_counter() - start) * 1000.0)

    lat_array = np.array(latencies)
    return {
        "mean_latency_ms": round(float(np.mean(lat_array)), 2),
        "std_latency_ms": round(float(np.std(lat_array)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_array, 95)), 2),
        "min_latency_ms": round(float(np.min(lat_array)), 2),
        "max_latency_ms": round(float(np.max(lat_array)), 2),
    }


def compute_evaluation_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: list[str],
    model_name: str,
    model_path: str | Path | None = None,
    latency_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics across 38 classes."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)
    model_size_mb = get_model_size_mb(model_path) if model_path else 0.0

    return {
        "model_name": model_name,
        "model_path": str(model_path) if model_path else "",
        "model_size_mb": model_size_mb,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "num_classes": len(class_names),
        "num_samples": len(y_true),
        "latency": latency_stats or {},
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (18, 16),
) -> None:
    """Render and save normalized confusion matrix heatmap."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=[c.split("___")[-1][:12] for c in class_names],
        yticklabels=[c.split("___")[-1][:12] for c in class_names],
    )
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()


def save_benchmark_results(
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Save structured evaluation outputs to results/ directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_slug = metrics["model_name"].lower().replace(" ", "_").replace("-", "_")

    # 1. Summary JSON
    summary_path = out / f"{model_slug}_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Classification Report CSV
    report_dict = metrics.get("classification_report", {})
    if report_dict:
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = out / f"{model_slug}_classification_report.csv"
        report_df.to_csv(report_path)
