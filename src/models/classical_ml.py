"""Classical Machine Learning Models (SVM, Random Forest, XGBoost) for Crop Disease Intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_svm_classifier(
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str | float = "scale",
    probability: bool = True,
    random_state: int = 42,
    **kwargs: Any,
) -> SVC:
    """Build a Support Vector Classifier (SVM) with probability calibration."""
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        class_weight="balanced",
        random_state=random_state,
        **kwargs,
    )


def build_random_forest_classifier(
    n_estimators: int = 150,
    max_depth: int | None = 25,
    min_samples_split: int = 4,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Build a Random Forest Classifier baseline."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
        **kwargs,
    )


def build_xgboost_classifier(
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    **kwargs: Any,
) -> XGBClassifier:
    """Build an XGBoost Gradient Boosted Decision Tree classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",
        **kwargs,
    )


def get_classical_model(model_name: str, **kwargs: Any) -> Any:
    """Factory function to build a classical ML model by name."""
    norm = model_name.strip().lower().replace(" ", "_").replace("-", "_")
    if "svm" in norm or "svc" in norm:
        return build_svm_classifier(**kwargs)
    elif "random_forest" in norm or "rf" in norm:
        return build_random_forest_classifier(**kwargs)
    elif "xgboost" in norm or "xgb" in norm:
        return build_xgboost_classifier(**kwargs)
    else:
        raise ValueError(
            f"Unsupported classical model: '{model_name}'. Choose from: SVM, Random Forest, XGBoost."
        )


def save_classical_model(model: Any, save_path: str | Path) -> None:
    """Serialize trained classical ML model using joblib."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_classical_model(model_path: str | Path) -> Any:
    """Load serialized classical ML model from disk."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Classical model artifact not found: {p}")
    return joblib.load(p)
