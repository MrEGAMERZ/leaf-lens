"""End-to-end Feature Pipeline for classical machine learning models.

Implements a scikit-learn compatible transformer that extracts image features
(Color Histograms + HOG + GLCM texture) and fits a StandardScaler strictly on
training data to prevent data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_extraction import extract_combined_features


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer that extracts normalized computer vision features from RGB images."""

    def __init__(
        self,
        color_bins: tuple[int, int, int] = (8, 8, 8),
        use_hog: bool = True,
        use_glcm: bool = True,
    ) -> None:
        self.color_bins = color_bins
        self.use_hog = use_hog
        self.use_glcm = use_glcm
        self.feature_dim_: int | None = None

    def fit(self, X: Sequence[np.ndarray], y: Any = None) -> ImageFeatureExtractor:
        """Fit transformer (determines feature dimension on first sample)."""
        if len(X) > 0:
            sample_feat = extract_combined_features(
                X[0],
                color_bins=self.color_bins,
                use_hog=self.use_hog,
                use_glcm=self.use_glcm,
            )
            self.feature_dim_ = len(sample_feat)
        return self

    def transform(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """Extract features for each image in X."""
        features_list = [
            extract_combined_features(
                img,
                color_bins=self.color_bins,
                use_hog=self.use_hog,
                use_glcm=self.use_glcm,
            )
            for img in X
        ]
        return np.vstack(features_list).astype(np.float32)


def create_classical_feature_pipeline() -> Pipeline:
    """Create a reproducible feature extraction and scaling pipeline.

    Returns:
        Pipeline: ImageFeatureExtractor followed by StandardScaler.
    """
    return Pipeline(
        [
            ("extractor", ImageFeatureExtractor(color_bins=(8, 8, 8), use_hog=True, use_glcm=True)),
            ("scaler", StandardScaler()),
        ]
    )


def save_feature_pipeline(pipeline: Pipeline, save_path: str | Path) -> None:
    """Serialize fitted feature pipeline to disk using joblib."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, p)


def load_feature_pipeline(pipeline_path: str | Path) -> Pipeline:
    """Load serialized feature pipeline from disk."""
    p = Path(pipeline_path)
    if not p.exists():
        raise FileNotFoundError(f"Feature pipeline file not found: {p}")
    return joblib.load(p)
