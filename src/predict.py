"""CLI script for single-image crop disease prediction with optional Grad-CAM explainability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import tensorflow as tf

from src.features.feature_pipeline import load_feature_pipeline
from src.inference.predictor import CropDiseasePredictor
from src.models.classical_ml import load_classical_model
from src.utils import load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crop disease class for an image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--model",
        default="models/best_model.keras",
        help="Path to trained model (.keras or .joblib).",
    )
    parser.add_argument(
        "--class-names",
        default="results/class_names.json",
        help="Path to class names JSON.",
    )
    parser.add_argument(
        "--pipeline",
        default="models/classical_feature_pipeline.joblib",
        help="Path to feature pipeline (required for classical ML models).",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate and save Grad-CAM heatmap overlay (CNN models only).",
    )
    parser.add_argument(
        "--gradcam-out",
        default="results/gradcam_prediction.png",
        help="Output path for Grad-CAM overlay image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    class_names_path = Path(args.class_names)

    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not class_names_path.is_absolute():
        class_names_path = ROOT / class_names_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names not found: {class_names_path}")

    class_names = load_class_names(class_names_path)

    # Determine model type
    is_joblib = model_path.suffix.lower() == ".joblib"
    feature_pipeline = None

    if is_joblib:
        model = load_classical_model(model_path)
        pipe_path = Path(args.pipeline)
        if not pipe_path.is_absolute():
            pipe_path = ROOT / pipe_path
        if pipe_path.exists():
            feature_pipeline = load_feature_pipeline(pipe_path)
        model_name = model_path.stem.replace("_", " ").title()
    else:
        model = tf.keras.models.load_model(model_path)
        model_name = "EfficientNetB0"

    predictor = CropDiseasePredictor(
        model=model,
        class_names=class_names,
        model_name=model_name,
        feature_pipeline=feature_pipeline,
    )

    result = predictor.predict(image_path, top_k=3, include_gradcam=args.gradcam)

    top = result["top_prediction"]
    print("=" * 50)
    print("Crop Disease Intelligence - Inference Report")
    print("=" * 50)
    print(f"Model Used:          {result['model_name']}")
    print(f"Inference Latency:   {result['inference_latency_ms']:.2f} ms")
    print(f"Predicted Crop:      {top['crop']}")
    print(f"Condition:           {top['condition']}")
    print(f"Confidence:          {top['confidence_percent']}% ({top['confidence_tier']})")
    print(f"Full Label:          {top['label']}")

    if result.get("is_uncertain"):
        print("\n[WARNING]")
        print(result["uncertainty_warning"])

    print("\nTop 3 Predictions:")
    for item in result["top_k_predictions"]:
        print(f"  {item['rank']}. {item['crop']} - {item['condition']}: {item['confidence_percent']}%")

    if args.gradcam and result["gradcam"] and result["gradcam"]["supported"]:
        out_p = Path(args.gradcam_out)
        if not out_p.is_absolute():
            out_p = ROOT / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        overlay_rgb = result["gradcam"]["overlay_image"]
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_p), overlay_bgr)
        print(f"\nGrad-CAM overlay saved to: {out_p}")


if __name__ == "__main__":
    main()
