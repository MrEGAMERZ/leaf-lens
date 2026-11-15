"""Leaf Lens — Explainability (Grad-CAM) Page."""

from __future__ import annotations

import io
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
    page_title="Explainability | Leaf Lens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .page-title {
            font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(135deg, #2E8B57, #9C27B0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .theory-box {
            background: #f8f9fa; border-radius: 12px; padding: 1.2rem 1.5rem;
            border-left: 5px solid #9C27B0; margin: 1rem 0;
        }
        .warning-box {
            background: #fff8e1; border-radius: 10px; padding: 1rem 1.2rem;
            border-left: 4px solid #FF9800; margin: 1rem 0;
        }
        .section-header {
            font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
            margin: 1.5rem 0 0.6rem 0; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🔬 Explainable AI — Grad-CAM</p>', unsafe_allow_html=True)
st.markdown(
    "Understand *why* the model makes a prediction by visualising which pixels drove it.",
)
st.divider()

# ── Theory ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">How Grad-CAM Works</p>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="theory-box">
        <b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> produces a coarse
        localisation map that highlights the image regions most influential for a given
        prediction class.

        <ol style="margin-top:0.8rem;">
            <li>A forward pass is run to get a class prediction.</li>
            <li>Gradients of the predicted class score are computed with respect to the
                activations of the last convolutional layer.</li>
            <li>These gradients are global-average-pooled to produce channel importance weights.</li>
            <li>The weighted sum of activation maps is ReLU'd and resized to match the input image.</li>
            <li>The heatmap is overlaid on the original image at a configurable opacity.</li>
        </ol>

        <b>What it shows:</b> Regions the network attends to — typically lesion boundaries,
        discolouration patches, or texture anomalies on the leaf.<br>
        <b>What it does not show:</b> Proof of correctness. A model can attend to the right region
        for the wrong reason.
    </div>
    """,
    unsafe_allow_html=True,
)

col_arch, col_eq = st.columns([3, 2])
with col_arch:
    st.markdown(
        """
        **Supported architectures in this platform:**
        | Architecture | Grad-CAM Support | Target Layer |
        |---|---|---|
        | EfficientNetB0 | ✅ Full | Last conv in nested backbone |
        | MobileNetV2 | ✅ Full | Last depthwise conv |
        | XGBoost | ❌ Not applicable | — |
        | Random Forest | ❌ Not applicable | — |
        | SVM | ❌ Not applicable | — |

        Classical ML models do not use spatial feature maps, so Grad-CAM cannot be applied.
        """
    )
with col_eq:
    st.markdown(
        r"""
        **The importance weight for channel $k$:**

        $$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}$$

        **The heatmap:**

        $$L^c = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$
        """
    )

# ── Worked example ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Worked Example</p>', unsafe_allow_html=True)
gradcam_path = os.path.join(ROOT, "results", "gradcam_prediction.png")
if os.path.exists(gradcam_path):
    st.image(gradcam_path, caption="Grad-CAM overlay — Grape Black Rot (99.49% confidence). Left: original leaf. Right: heatmap overlay highlighting lesion regions.", use_container_width=True)
    st.caption("🔴 Hot colours (red/yellow) indicate high influence; 🔵 cool colours (blue) indicate low influence.")
else:
    st.info("Example Grad-CAM image not found at results/gradcam_prediction.png. Run `python src/predict.py --image <leaf.jpg> --gradcam` to generate one.")

# ── Interactive demo ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Interactive Demo</p>', unsafe_allow_html=True)
st.markdown("Upload a leaf image to generate a Grad-CAM heatmap using **EfficientNetB0**.")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], key="gradcam_upload")

if uploaded is not None:
    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    col_orig, col_cam = st.columns(2)
    with col_orig:
        st.markdown("**Original Image**")
        st.image(pil_img, use_container_width=True)

    with col_cam:
        st.markdown("**Grad-CAM Heatmap**")
        with st.spinner("Generating Grad-CAM heatmap…"):
            try:
                import tensorflow as tf
                from src.explainability.gradcam import compute_gradcam_heatmap, overlay_gradcam
                from src.utils import ensure_best_model, load_config

                CONFIG_PATH = os.path.join(ROOT, "configs", "efficientnetb0_config.json")
                config = load_config(CONFIG_PATH)
                model_path = ensure_best_model(config)

                @st.cache_resource
                def _load_model(path: str):
                    return tf.keras.models.load_model(path)

                model = _load_model(model_path)
                img_arr = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

                heatmap = compute_gradcam_heatmap(model, img_arr, target_class_idx=None)
                overlay = overlay_gradcam(np.array(pil_img), heatmap, alpha=0.40)
                st.image(overlay, use_container_width=True)
                st.caption("🔴 Red/yellow = high influence. 🔵 Blue = low influence.")
            except Exception as exc:
                st.error(f"Grad-CAM failed: {exc}")
                st.info("Ensure the model is loaded correctly and src/explainability/gradcam.py is accessible.")

# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Limitations & Caveats</p>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="warning-box">
        ⚠️ <b>Grad-CAM is an explanation tool, not a verification tool.</b>
        <ul style="margin-top:0.5rem;">
            <li>A heatmap centred on a lesion does <em>not</em> guarantee the prediction is correct.</li>
            <li>The model may focus on spurious correlations (e.g. background texture) that happen to co-occur with a disease class in the training dataset.</li>
            <li>Grad-CAM resolution is limited by the spatial size of the last convolutional feature map (typically 7×7 for EfficientNetB0 on 224×224 input).</li>
            <li>Always corroborate model predictions with agronomic expertise before taking field action.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
