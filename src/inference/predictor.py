"""Unified Inference Engine for Crop Disease Intelligence Platform.

Supports both deep learning CNN models (.keras) and classical ML models (.joblib),
providing standardized prediction outputs, confidence assessments, uncertainty warnings,
and optional Grad-CAM explainability.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from src.data.preprocessing import (
    clean_label,
    load_image_rgb,
    preprocess_for_model,
    validate_image_file,
)
from src.explainability.gradcam import (
    compute_gradcam_heatmap,
    create_side_by_side_comparison,
    is_gradcam_supported,
    overlay_gradcam,
)
from src.features.feature_pipeline import load_feature_pipeline


def get_confidence_tier(confidence: float) -> str:
    """Classify confidence score into human-interpretable tier."""
    percent = confidence * 100.0
    if percent >= 85.0:
        return "High confidence"
    elif percent >= 60.0:
        return "Medium confidence"
    else:
        return "Low confidence"


class CropDiseasePredictor:
    """Unified predictor supporting CNNs and Classical ML pipelines."""

    def __init__(
        self,
        model: Any,
        class_names: list[str],
        model_name: str = "EfficientNetB0",
        feature_pipeline: Any = None,
        uncertainty_threshold: float = 0.60,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.model_name = model_name
        self.feature_pipeline = feature_pipeline
        self.uncertainty_threshold = uncertainty_threshold
        self.is_cnn = hasattr(model, "layers") and not hasattr(model, "predict_proba")

    def predict(
        self,
        image_input: str | Path | BinaryIO | bytes | np.ndarray,
        top_k: int = 3,
        include_gradcam: bool = False,
        gradcam_alpha: float = 0.35,
    ) -> dict[str, Any]:
        """Run inference on an input image and return structured prediction results."""
        # 1. Load and validate RGB image
        if isinstance(image_input, np.ndarray):
            image_rgb = image_input
        else:
            image_rgb = load_image_rgb(image_input)

        start_time = time.perf_counter()

        # 2. Model inference
        if self.is_cnn:
            # Preprocess 4D tensor [1, 224, 224, 3]
            tensor = preprocess_for_model(image_rgb, model_name=self.model_name)
            raw_predictions = self.model.predict(tensor, verbose=0)
            probabilities = raw_predictions[0] if raw_predictions.ndim > 1 else raw_predictions
        else:
            # Classical ML inference
            if self.feature_pipeline is None:
                raise ValueError(
                    f"Feature pipeline is required for classical model '{self.model_name}'."
                )
            features = self.feature_pipeline.transform([image_rgb])
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
            else:
                # Fallback for decision function
                df = self.model.decision_function(features)[0]
                exp_df = np.exp(df - np.max(df))
                probabilities = exp_df / np.sum(exp_df)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Extract top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            class_idx = int(idx)
            label = (
                self.class_names[class_idx]
                if class_idx < len(self.class_names)
                else f"Class_{class_idx}"
            )
            crop, condition = clean_label(label)
            conf = float(probabilities[class_idx])
            top_predictions.append(
                {
                    "rank": rank,
                    "class_index": class_idx,
                    "label": label,
                    "crop": crop,
                    "condition": condition,
                    "confidence": conf,
                    "confidence_percent": round(conf * 100.0, 2),
                    "confidence_tier": get_confidence_tier(conf),
                }
            )

        top_pred = top_predictions[0]
        is_uncertain = top_pred["confidence"] < self.uncertainty_threshold

        uncertainty_warning = None
        if is_uncertain:
            uncertainty_warning = (
                f"Prediction uncertainty warning: Top confidence ({top_pred['confidence_percent']}%) "
                f"is below the reliability threshold ({round(self.uncertainty_threshold * 100, 1)}%). "
                "Visible symptoms may be ambiguous, unrepresented in training data, or multiple "
                "stress factors may be present. Visual inspection by an agricultural expert is recommended."
            )

        # 4. Optional Grad-CAM explainability
        gradcam_result = None
        if include_gradcam and self.is_cnn and is_gradcam_supported(self.model):
            try:
                heatmap = compute_gradcam_heatmap(
                    self.model,
                    tensor,
                    target_class_idx=top_pred["class_index"],
                )
                overlay = overlay_gradcam(image_rgb, heatmap, alpha=gradcam_alpha)
                side_by_side = create_side_by_side_comparison(image_rgb, overlay)
                gradcam_result = {
                    "supported": True,
                    "heatmap": heatmap,
                    "overlay_image": overlay,
                    "side_by_side_image": side_by_side,
                    "target_class": top_pred["label"],
                    "explanation": (
                        "Highlighted regions indicate spatial visual patterns in the leaf that most strongly "
                        f"influenced the model's diagnosis of {top_pred['crop']} - {top_pred['condition']}."
                    ),
                }
            except Exception as e:
                gradcam_result = {
                    "supported": False,
                    "error": str(e),
                    "explanation": "Grad-CAM could not be computed for this layer.",
                }
        elif include_gradcam:
            gradcam_result = {
                "supported": False,
                "explanation": (
                    f"Model '{self.model_name}' does not support Grad-CAM. "
                    "Grad-CAM requires a differentiable Convolutional Neural Network."
                ),
            }

        return {
            "model_name": self.model_name,
            "inference_latency_ms": round(elapsed_ms, 2),
            "top_prediction": top_pred,
            "top_k_predictions": top_predictions,
            "is_uncertain": is_uncertain,
            "uncertainty_warning": uncertainty_warning,
            "gradcam": gradcam_result,
            "image_shape": image_rgb.shape,
        }
