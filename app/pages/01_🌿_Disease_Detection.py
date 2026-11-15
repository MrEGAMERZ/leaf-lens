"""Leaf Lens — Disease Detection Page."""

import io
import json
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Detection | Leaf Lens",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .page-title {
            font-size: 2.4rem; font-weight: 800;
            background: linear-gradient(135deg, #2E8B57, #4682B4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 4px;
        }
        .conf-badge-green  { background:#d4edda; color:#155724; border:1px solid #c3e6cb;
                             border-radius:20px; padding:6px 16px; font-weight:700; font-size:1.1rem; }
        .conf-badge-orange { background:#fff3cd; color:#856404; border:1px solid #ffeeba;
                             border-radius:20px; padding:6px 16px; font-weight:700; font-size:1.1rem; }
        .conf-badge-red    { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb;
                             border-radius:20px; padding:6px 16px; font-weight:700; font-size:1.1rem; }
        .result-card {
            background: linear-gradient(135deg, #f0fff4, #e8f5e9);
            border: 1px solid #c3e6cb; border-radius: 14px; padding: 22px 26px;
        }
        .section-header {
            font-size: 1.3rem; font-weight: 700; color: #2E8B57;
            border-bottom: 2px solid #2E8B57; padding-bottom: 4px; margin: 20px 0 12px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data & model helpers ──────────────────────────────────────────────────────
@st.cache_data
def load_class_names() -> list:
    path = os.path.join(ROOT, "results", "class_names.json")
    with open(path, "r") as fh:
        return json.load(fh)


@st.cache_resource
def load_cnn_model(model_path: str):
    """Load a Keras CNN model with optional config sanitisation."""
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception:
        # Fallback: sanitise config.json if SavedModel format causes issues
        import zipfile, shutil, tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                zf.extractall(tmpdir)
            cfg_path = os.path.join(tmpdir, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg_text = f.read()
                cfg_text = cfg_text.replace('"groups": 1,', "")
                with open(cfg_path, "w") as f:
                    f.write(cfg_text)
            model = tf.keras.models.load_model(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return model


@st.cache_resource
def load_classical_models():
    """Load joblib classical ML models and feature pipeline."""
    import joblib
    models_dir = os.path.join(ROOT, "models")
    pipeline = joblib.load(os.path.join(models_dir, "classical_feature_pipeline.joblib"))
    svm   = joblib.load(os.path.join(models_dir, "svm_model.joblib"))
    rf    = joblib.load(os.path.join(models_dir, "random_forest_model.joblib"))
    xgb   = joblib.load(os.path.join(models_dir, "xgboost_model.joblib"))
    return pipeline, svm, rf, xgb


CLASS_NAMES = load_class_names()

MODEL_OPTIONS = [
    "EfficientNetB0 (Best, 97.0%)",
    "MobileNetV2 (Fast, 96.0%)",
    "XGBoost (Classical, 94.3%)",
    "Random Forest (Classical, 90.6%)",
    "SVM (Classical, 89.5%)",
]

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="page-title">🌿 Disease Detection</h1>', unsafe_allow_html=True)
st.markdown(
    "Upload a leaf image to receive an AI-powered diagnosis. "
    "Choose between deep learning and classical ML models."
)
st.divider()

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<div class="section-header">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded leaf image", use_container_width=True)

    st.markdown('<div class="section-header">🤖 Select Model</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Model", MODEL_OPTIONS, index=0, label_visibility="collapsed"
    )

    is_cnn = selected_model_name.startswith("EfficientNetB0") or selected_model_name.startswith("MobileNetV2")

    if is_cnn:
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
    else:
        show_gradcam = False
        st.info("Grad-CAM is only available for CNN models.")

    run_btn = st.button("🔍 Run Prediction", type="primary", use_container_width=True, disabled=not uploaded_file)

with right_col:
    if uploaded_file and run_btn:
        try:
            img_bytes = uploaded_file.getvalue()
            image_array = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            with st.spinner("Running inference…"):
                from src.inference.predictor import CropDiseasePredictor

                if is_cnn:
                    models_dir = os.path.join(ROOT, "models")
                    if selected_model_name.startswith("EfficientNetB0"):
                        model_path = os.path.join(models_dir, "best_model.keras")
                        m_name = "EfficientNetB0"
                    else:
                        # MobileNetV2 — look for mobilenetv2 keras file
                        model_path = os.path.join(models_dir, "mobilenetv2_model.keras")
                        if not os.path.exists(model_path):
                            # fall back to best model
                            model_path = os.path.join(models_dir, "best_model.keras")
                        m_name = "MobileNetV2"

                    cnn_model = load_cnn_model(model_path)
                    predictor = CropDiseasePredictor(
                        model=cnn_model,
                        class_names=CLASS_NAMES,
                        model_name=m_name,
                    )
                    result = predictor.predict(
                        image_array,
                        top_k=3,
                        include_gradcam=show_gradcam,
                    )
                else:
                    pipeline, svm_m, rf_m, xgb_m = load_classical_models()
                    if "XGBoost" in selected_model_name:
                        clf, m_name = xgb_m, "XGBoost"
                    elif "Random Forest" in selected_model_name:
                        clf, m_name = rf_m, "Random Forest"
                    else:
                        clf, m_name = svm_m, "SVM"

                    predictor = CropDiseasePredictor(
                        model=clf,
                        class_names=CLASS_NAMES,
                        model_name=m_name,
                        feature_pipeline=pipeline,
                    )
                    result = predictor.predict(image_array, top_k=3, include_gradcam=False)

            # ── Results ───────────────────────────────────────────────────────
            top = result["top_prediction"]
            conf_pct = top["confidence_percent"]

            st.markdown('<div class="section-header">🎯 Diagnosis Result</div>', unsafe_allow_html=True)

            if conf_pct >= 85:
                badge_cls = "conf-badge-green"
                tier_icon = "✅"
            elif conf_pct >= 60:
                badge_cls = "conf-badge-orange"
                tier_icon = "⚠️"
            else:
                badge_cls = "conf-badge-red"
                tier_icon = "🔴"

            r1, r2, r3 = st.columns(3)
            r1.metric("🌾 Crop", top["crop"])
            r2.metric("🦠 Condition", top["condition"])
            r3.markdown(
                f'<br><span class="{badge_cls}">{tier_icon} {conf_pct:.1f}% — {top["confidence_tier"]}</span>',
                unsafe_allow_html=True,
            )

            st.caption(f"Inference latency: **{result['inference_latency_ms']:.1f} ms** | Model: **{result['model_name']}**")

            if result["is_uncertain"]:
                st.warning(
                    "⚠️ **Low confidence** — please consult an agricultural expert. "
                    "The model is not sufficiently certain about this prediction."
                )

            # ── Top 3 predictions ─────────────────────────────────────────────
            st.markdown('<div class="section-header">📋 Top 3 Predictions</div>', unsafe_allow_html=True)
            import pandas as pd
            top3_rows = []
            for pred in result["top_k_predictions"]:
                top3_rows.append({
                    "Rank": f"#{pred['rank']}",
                    "Crop": pred["crop"],
                    "Condition": pred["condition"],
                    "Confidence": f"{pred['confidence_percent']:.2f}%",
                    "Tier": pred["confidence_tier"],
                })
            st.dataframe(pd.DataFrame(top3_rows), use_container_width=True, hide_index=True)

            # ── Grad-CAM ──────────────────────────────────────────────────────
            if show_gradcam and result.get("gradcam"):
                gcam = result["gradcam"]
                st.markdown('<div class="section-header">🔬 Grad-CAM Explanation</div>', unsafe_allow_html=True)
                if gcam.get("supported") and gcam.get("overlay_image") is not None:
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.image(image_array, caption="Original Image", use_container_width=True)
                    with gc2:
                        st.image(gcam["overlay_image"], caption="Grad-CAM Heatmap", use_container_width=True)
                    st.caption(gcam.get("explanation", ""))
                else:
                    st.info(gcam.get("explanation", "Grad-CAM not available for this prediction."))

        except Exception as exc:
            st.error(
                f"**Prediction failed:** {exc}\n\n"
                "Possible causes:\n"
                "- Model file not found (check `models/` directory)\n"
                "- Incompatible image format\n"
                "- Missing dependency (TensorFlow, joblib, OpenCV)\n\n"
                "Please verify model files exist and try again."
            )
    elif not uploaded_file:
        st.markdown(
            """
            <div style="text-align:center; padding:60px 20px; color:#888;">
                <h2>📤 Upload a leaf image to begin</h2>
                <p>Supported formats: JPG, JPEG, PNG</p>
                <p>Select your preferred model, then click <strong>Run Prediction</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
