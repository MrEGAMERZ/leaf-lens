"""Leaf Lens — Dataset & Methodology Page."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dataset & Methodology | Leaf Lens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .page-title {
            font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(135deg, #E65100, #2E8B57);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .info-box {
            background: #e8f5e9; border-radius: 10px; padding: 1rem 1.4rem;
            border-left: 4px solid #2E8B57; margin: 0.6rem 0;
        }
        .caution-box {
            background: #fbe9e7; border-radius: 10px; padding: 1rem 1.4rem;
            border-left: 4px solid #E65100; margin: 0.6rem 0;
        }
        .section-header {
            font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
            margin: 1.5rem 0 0.6rem 0; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
        .step-box {
            background: #f3f4f6; border-radius: 8px; padding: 0.7rem 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">📚 Dataset & Methodology</p>', unsafe_allow_html=True)
st.markdown("How the data was sourced, processed, split, and used to train and evaluate the models.")
st.divider()

# ── Dataset overview ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">PlantVillage Dataset</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Images", "87,000+")
c2.metric("Disease Classes", "38")
c3.metric("Crop Species", "14")
c4.metric("Image Resolution", "256×256 px")

st.markdown(
    """
    <div class="info-box">
        PlantVillage is a publicly available benchmark dataset of leaf images, originally published by
        Hughes & Salathé (2015) and subsequently extended. Each image shows a single leaf on a uniform
        background, labelled with crop species and disease (or healthy). It is widely used for plant
        pathology computer vision research.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 38 Classes table ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Supported Classes (38)</p>', unsafe_allow_html=True)

CLASS_NAMES_PATH = os.path.join(ROOT, "results", "class_names.json")


@st.cache_data
def load_classes() -> list[str]:
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


class_names = load_classes()

# Parse into a readable table
rows = []
for raw in class_names:
    parts = raw.split("___", 1)
    crop = parts[0].replace("_", " ") if len(parts) > 0 else raw
    condition = parts[1].replace("_", " ").replace("  ", " ") if len(parts) > 1 else "—"
    is_healthy = "healthy" in condition.lower()
    rows.append(
        {
            "Class ID": raw,
            "Crop": crop,
            "Condition": condition,
            "Status": "✅ Healthy" if is_healthy else "🔴 Disease",
        }
    )

classes_df = pd.DataFrame(rows)

# Crop filter
crops_available = sorted(classes_df["Crop"].unique())
selected_crop = st.selectbox("Filter by crop", ["All"] + crops_available)
if selected_crop != "All":
    display_df = classes_df[classes_df["Crop"] == selected_crop]
else:
    display_df = classes_df

st.dataframe(display_df[["Crop", "Condition", "Status"]].reset_index(drop=True), use_container_width=True, hide_index=True)
st.caption(f"Showing {len(display_df)} of 38 classes.")

# ── Class distribution chart ───────────────────────────────────────────────────
st.markdown('<p class="section-header">Class Distribution</p>', unsafe_allow_html=True)
st.markdown("The PlantVillage dataset is approximately balanced across classes (~2,000–2,500 images per class).")

crop_counts = classes_df.groupby("Crop")["Status"].count().reset_index()
crop_counts.columns = ["Crop", "Number of Disease Classes"]
fig_dist = px.bar(
    crop_counts.sort_values("Number of Disease Classes", ascending=True),
    x="Number of Disease Classes",
    y="Crop",
    orientation="h",
    color="Number of Disease Classes",
    color_continuous_scale="Greens",
    text="Number of Disease Classes",
)
fig_dist.update_layout(
    height=420,
    coloraxis_showscale=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="Number of Labelled Conditions",
    yaxis_title="",
    margin=dict(l=10, r=10, t=20, b=10),
)
fig_dist.update_traces(textposition="outside")
fig_dist.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig_dist, use_container_width=True)

# ── Preprocessing pipeline ────────────────────────────────────────────────────
st.markdown('<p class="section-header">Preprocessing Pipeline</p>', unsafe_allow_html=True)
pre_col, aug_col = st.columns(2)

with pre_col:
    st.markdown("**Image Preprocessing (all splits)**")
    for step in [
        "1. Validate image file (check readability, dimensions)",
        "2. Decode and load as RGB (OpenCV / Pillow)",
        "3. Resize to 224 × 224 pixels (bicubic for DL; area interp for classical)",
        "4. Normalise pixel values to [0, 1]",
        "5. Convert to float32 tensor",
    ]:
        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

with aug_col:
    st.markdown("**Data Augmentation (training split only)**")
    for step in [
        "• Random horizontal flip",
        "• Random rotation ±10°",
        "• Random zoom ±10%",
        "• Applied inline via Keras preprocessing layers",
        "• Never applied to validation or test images",
    ]:
        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

# ── Classical ML features ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">Classical ML Feature Pipeline</p>', unsafe_allow_html=True)
st.markdown("Each image is converted to a **2,364-dimensional feature vector** before training SVM, Random Forest, or XGBoost.")

feat_col1, feat_col2, feat_col3 = st.columns(3)
with feat_col1:
    st.markdown(
        """
        **Color Histogram**
        - HSV: 8×8×8 = 512 bins
        - RGB: 4×4×4 = 64 bins
        - Total: **576 dims**
        - Captures colour distribution per channel
        """
    )
with feat_col2:
    st.markdown(
        """
        **HOG (Gradient Orientation)**
        - 9 orientation bins
        - 16×16 pixels per cell
        - 2×2 cells per block
        - Implemented via NumPy + Sobel
        - Total: **~1,764 dims**
        - Captures texture and edge structure
        """
    )
with feat_col3:
    st.markdown(
        """
        **GLCM (Haralick Texture)**
        - 4 directions × 2 distances
        - Contrast, energy, homogeneity, correlation
        - Total: **24 dims**
        - Captures co-occurrence texture statistics
        """
    )

st.markdown("All features are concatenated and **z-score standardised** using a `StandardScaler` fit only on training data.")

# ── Train / val / test split ──────────────────────────────────────────────────
st.markdown('<p class="section-header">Data Splitting Strategy</p>', unsafe_allow_html=True)
split_col1, split_col2 = st.columns([2, 1])

with split_col1:
    st.markdown(
        """
        | Split | Fraction | Purpose |
        |---|---|---|
        | Train | 70% | Model fitting, weight updates, feature scaling |
        | Validation | 15% | Hyperparameter tuning, early stopping decisions |
        | Test | 15% | Final held-out evaluation — **never used for model selection** |

        - **Stratified split** ensures equal class representation in every partition.
        - **Random seed: 42** for full reproducibility.
        - **No data leakage**: StandardScaler is fit on training images only, then applied to val and test.
        - Deep learning models use validation loss for early stopping; test results are computed once after training is complete.
        """
    )

with split_col2:
    split_data = pd.DataFrame({"Split": ["Train", "Val", "Test"], "Fraction (%)": [70, 15, 15]})
    fig_split = px.pie(
        split_data,
        values="Fraction (%)",
        names="Split",
        color_discrete_map={"Train": "#2E8B57", "Val": "#4682B4", "Test": "#E65100"},
        hole=0.4,
    )
    fig_split.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
    st.plotly_chart(fig_split, use_container_width=True)

# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Dataset Limitations</p>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="caution-box">
        ⚠️ <b>Important Limitations — Read Before Deployment</b>
        <ul style="margin-top:0.6rem;">
            <li><b>Controlled conditions:</b> PlantVillage images are photographed against plain backgrounds under
                uniform lighting. Real farm photographs include soil backgrounds, partial occlusion, varied lighting,
                and mixed foliage — conditions the model was not trained on.</li>
            <li><b>Generalisation gap:</b> High test accuracy on PlantVillage (<em>in-distribution</em>) does not
                translate directly to high accuracy on real-world photographs (<em>out-of-distribution</em>).</li>
            <li><b>Class coverage:</b> Only 38 disease/healthy combinations from 14 crop species are covered.
                Many diseases, regional strains, and crop species are absent.</li>
            <li><b>Not a replacement for experts:</b> This platform is a research and educational tool.
                Always consult a qualified agricultural expert or plant pathologist before taking field action.</li>
            <li><b>Confidence ≠ correctness:</b> A high softmax confidence score indicates the model's
                certainty given its training distribution — it is not a calibrated probability of correctness.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
