"""Leaf Lens — About & Technology Stack Page."""

from __future__ import annotations

import os
import sys

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="About | Leaf Lens",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .page-title {
            font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(135deg, #1565C0, #2E8B57);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .tech-card {
            background: #f8f9fa; border-radius: 10px; padding: 0.9rem 1.1rem;
            border-left: 4px solid #2E8B57; margin-bottom: 0.6rem;
        }
        .tech-card b { font-size: 1rem; }
        .tech-card span { font-size: 0.85rem; color: #555; }
        .section-header {
            font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
            margin: 1.5rem 0 0.6rem 0; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
        .credit-box {
            background: #e8f5e9; border-radius: 10px; padding: 1rem 1.4rem;
            margin: 0.5rem 0; border-left: 3px solid #4CAF50;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">ℹ️ About Leaf Lens</p>', unsafe_allow_html=True)
st.markdown(
    """
    **Leaf Lens — Crop Disease Intelligence Platform** is an open-source machine learning project
    that demonstrates end-to-end applied AI for agricultural image analysis.
    It is designed as a research reference implementation — not a production diagnostic tool.
    """
)
st.divider()

# ── Project mission ───────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Project Mission</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        **🎓 Educational**

        Demonstrates comparative ML — from handcrafted classical features (HOG, GLCM, color histograms)
        to deep transfer learning — on a real computer vision problem.
        """
    )
with c2:
    st.markdown(
        """
        **🔬 Reproducible Science**

        Every benchmark metric is generated from real model evaluation runs using a held-out test set.
        Seeds, splits, and feature pipelines are documented and versioned.
        """
    )
with c3:
    st.markdown(
        """
        **🌱 Real Impact Context**

        The PlantVillage dataset is laboratory-quality. We explicitly document the generalisation gap
        to real farm conditions rather than overstating applicability.
        """
    )

# ── Technology stack ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Technology Stack</p>', unsafe_allow_html=True)

STACK = [
    ("Python 3.10", "Core development language. All training, evaluation, and inference pipelines.", "#2E8B57"),
    ("TensorFlow 2.21 / Keras 3", "CNN training, transfer learning (EfficientNetB0, MobileNetV2), model serialisation.", "#FF6F00"),
    ("Scikit-learn 1.7", "SVM, Random Forest, StandardScaler, stratified splits, classification reports.", "#F7931E"),
    ("XGBoost 3.2", "Gradient boosting on handcrafted features. Best classical ML baseline at 94.3% accuracy.", "#189AB4"),
    ("OpenCV 5 / Pillow", "Image loading, resizing, colour space conversion, HOG gradient computation.", "#5C2D91"),
    ("Streamlit 1.65", "Interactive multi-page web application — uploaded image prediction, benchmarking dashboard.", "#FF4B4B"),
    ("Plotly", "Interactive charts: accuracy/F1 bar charts, latency scatter plots, pie charts.", "#3F4F75"),
    ("NumPy / Pandas / SciPy", "Feature vectors, data manipulation, GLCM Haralick statistics.", "#013243"),
    ("Grad-CAM (custom)", "Gradient-weighted Class Activation Maps for CNN prediction explainability.", "#9C27B0"),
    ("Joblib", "Serialisation and loading of trained classical ML pipelines (feature extractor + model).", "#4682B4"),
    ("Docker", "Containerised deployment: `docker build . && docker run -p 8501:8501 leaf-lens`", "#2496ED"),
    ("GitHub Actions", "CI pipeline: syntax check and import validation on every push to main.", "#2088FF"),
]

col_left, col_right = st.columns(2)
for i, (name, purpose, color) in enumerate(STACK):
    col = col_left if i % 2 == 0 else col_right
    with col:
        st.markdown(
            f"""
            <div class="tech-card" style="border-left-color:{color}">
                <b>{name}</b><br><span>{purpose}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Architecture overview ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">System Architecture</p>', unsafe_allow_html=True)
st.markdown(
    """
    ```
    Leaf image (224×224 RGB)
         │
         ├─── Deep Learning path ──────────────────────────────────────────────┐
         │    Augmentation → Pretrained Backbone → GlobalAvgPool → Dense(38)   │
         │    Supported: EfficientNetB0 (97%), MobileNetV2 (96%)               │
         │    Training: 2-stage (frozen backbone → fine-tune upper layers)     │
         │    Explainability: Grad-CAM heatmap on last conv layer              │
         │                                                                     │
         └─── Classical ML path ───────────────────────────────────────────────┤
              Colour Hist (576) + HOG (1764) + GLCM (24) = 2,364-dim vector   │
              StandardScaler → SVM / Random Forest / XGBoost                  │
              Best: XGBoost (94.3%)                                            │
                                                                               │
                        Unified CropDiseasePredictor ◄──────────────────────── ┘
                              │
                              ├── Class label (crop + disease)
                              ├── Confidence score + tier (High/Medium/Low)
                              ├── Top-3 predictions
                              ├── Uncertainty warning (< 60%)
                              └── Grad-CAM overlay (CNN only)
    ```
    """
)

# ── Benchmark summary ─────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Benchmark Summary</p>', unsafe_allow_html=True)
st.markdown(
    """
    | Model | Accuracy | Macro F1 | Latency | Size |
    |---|---|---|---|---|
    | EfficientNetB0 | **97.02%** | **97.02%** | 14.85 ms | 16.82 MB |
    | MobileNetV2 | 95.97% | 95.82% | **7.32 ms** | 9.45 MB |
    | XGBoost | 94.29% | 92.86% | 4.54 ms | **2.15 MB** |
    | Random Forest | 90.64% | 87.92% | 16.98 ms | 11.68 MB |
    | SVM | 89.50% | 86.58% | 8.98 ms | 35.56 MB |

    *Evaluated on 2,916 held-out test images from 15 PlantVillage classes. Seed: 42.*
    """
)

# ── Credits ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Credits & Acknowledgements</p>', unsafe_allow_html=True)
for credit in [
    ("PlantVillage Dataset", "Hughes, D.P. & Salathé, M. (2015). An open access repository of images of diseased and healthy plants. arXiv:1511.08060."),
    ("TensorFlow / Keras Team", "Open-source deep learning framework. tensorflow.org"),
    ("Streamlit", "Open-source Python app framework for ML/data science. streamlit.io"),
    ("Scikit-learn", "Machine learning in Python. Pedregosa et al. (2011). JMLR 12, 2825–2830."),
    ("Selvaraju et al. (2017)", "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. arXiv:1610.02391."),
]:
    st.markdown(
        f"""
        <div class="credit-box">
            <b>{credit[0]}</b><br><span style="font-size:0.85rem;color:#555;">{credit[1]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── GitHub / links ────────────────────────────────────────────────────────────
st.divider()
col_gh, col_disc = st.columns(2)
with col_gh:
    st.markdown(
        """
        **📦 GitHub Repository**

        [github.com/MrEGAMERZ/leaf-lens](https://github.com/MrEGAMERZ/leaf-lens)

        Issues, PRs, and contributions welcome.
        """
    )
with col_disc:
    st.markdown(
        """
        **⚠️ Disclaimer**

        This platform is for research and educational purposes only.
        It is not a certified agricultural diagnostic tool.
        Always consult qualified plant pathologists before taking farm action.
        Model accuracy figures are measured on the PlantVillage dataset and may
        not generalise to real-world field conditions.
        """
    )
