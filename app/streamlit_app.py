"""Leaf Lens | Crop Disease Intelligence Platform — Home Page."""

from __future__ import annotations

import json
import os
import sys

import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Leaf Lens | Crop Disease Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Hero gradient title */
        .hero-title {
            font-size: 3rem; font-weight: 900; line-height: 1.1;
            background: linear-gradient(135deg, #1b5e20 0%, #2E8B57 40%, #4CAF50 70%, #81C784 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.4rem;
        }
        .hero-sub {
            font-size: 1.15rem; color: #444; max-width: 700px; line-height: 1.6;
        }
        /* Metric card */
        .metric-card {
            background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border-top: 4px solid var(--accent);
            text-align: center;
        }
        .metric-card .metric-val {
            font-size: 2rem; font-weight: 800; color: var(--accent);
        }
        .metric-card .metric-label {
            font-size: 0.8rem; color: #666; margin-top: 2px;
        }
        /* Section header */
        .section-header {
            font-size: 1.4rem; font-weight: 700; color: #1a1a2e;
            margin: 1.6rem 0 0.6rem 0; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
        /* Nav card */
        .nav-card {
            background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            border-left: 5px solid #2E8B57; margin-bottom: 0.8rem;
            transition: box-shadow 0.2s;
        }
        .nav-card:hover { box-shadow: 0 4px 20px rgba(46,139,87,0.2); }
        .nav-card .nav-title { font-size: 1rem; font-weight: 700; }
        .nav-card .nav-desc { font-size: 0.85rem; color: #555; margin-top: 0.3rem; }
        /* Footer */
        .footer {
            text-align: center; color: #888; font-size: 0.8rem;
            margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;
        }
        /* Tech badge */
        .tech-pill {
            display: inline-block; background: #e8f5e9; color: #1b5e20;
            border-radius: 20px; padding: 3px 12px; font-size: 0.8rem;
            font-weight: 600; margin: 3px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(ROOT, "results", "model_comparison.json")
CLASS_NAMES_PATH = os.path.join(ROOT, "results", "class_names.json")


@st.cache_data
def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_class_names() -> list[str]:
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


results = load_results()
class_names = load_class_names()

# Parse unique crops
crops = sorted({n.split("___")[0].replace("_", " ") for n in class_names})

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌿 Leaf Lens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Crop Disease Intelligence Platform — '
    "comparative ML benchmarking (SVM → EfficientNetB0), Grad-CAM explainability, "
    "and interactive disease detection on PlantVillage's 38-class dataset.</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Top-level stats ───────────────────────────────────────────────────────────
h_col1, h_col2, h_col3, h_col4 = st.columns(4)
with h_col1:
    st.markdown(
        '<div class="metric-card" style="--accent:#2E8B57;">'
        '<div class="metric-val">97.0%</div>'
        '<div class="metric-label">Best model accuracy<br>(EfficientNetB0)</div></div>',
        unsafe_allow_html=True,
    )
with h_col2:
    st.markdown(
        '<div class="metric-card" style="--accent:#4682B4;">'
        '<div class="metric-val">5</div>'
        '<div class="metric-label">Models compared<br>(2 DL + 3 Classical)</div></div>',
        unsafe_allow_html=True,
    )
with h_col3:
    st.markdown(
        '<div class="metric-card" style="--accent:#E65100;">'
        f'<div class="metric-val">{len(class_names)}</div>'
        '<div class="metric-label">Disease / healthy<br>classes supported</div></div>',
        unsafe_allow_html=True,
    )
with h_col4:
    st.markdown(
        '<div class="metric-card" style="--accent:#9C27B0;">'
        '<div class="metric-val">2,916</div>'
        '<div class="metric-label">Held-out test images<br>for benchmarking</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Performance overview chart ────────────────────────────────────────────────
st.markdown('<p class="section-header">Model Performance Overview</p>', unsafe_allow_html=True)

chart_col, legend_col = st.columns([3, 1])
with chart_col:
    display_names = {
        "EfficientNetB0 (38 Classes)": "EfficientNetB0",
        "MobileNetV2 (38 Classes)": "MobileNetV2",
        "XGBoost": "XGBoost",
        "Random Forest": "Random Forest",
        "SVM": "SVM",
    }
    model_colors_map = {
        "EfficientNetB0": "#2E8B57",
        "MobileNetV2": "#3CB371",
        "XGBoost": "#4682B4",
        "Random Forest": "#6495ED",
        "SVM": "#87CEEB",
    }

    models, accs, f1s, colors = [], [], [], []
    for key, vals in results.items():
        name = display_names.get(key, key)
        models.append(name)
        accs.append(round(vals["accuracy"] * 100, 2))
        f1s.append(round(vals["macro_f1"] * 100, 2))
        colors.append(model_colors_map.get(name, "#888"))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Accuracy (%)",
            x=models,
            y=accs,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in accs],
            textposition="outside",
            offsetgroup=0,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Macro F1 (%)",
            x=models,
            y=f1s,
            marker_color=colors,
            marker_pattern_shape="/",
            text=[f"{v:.1f}%" for v in f1s],
            textposition="outside",
            offsetgroup=1,
            opacity=0.72,
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (%)", range=[82, 100]),
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, b=10, l=0, r=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

with legend_col:
    st.markdown("**Benchmark details**")
    st.markdown(
        """
        - **Test set:** 2,916 images
        - **Classes:** 15 PlantVillage
        - **Seed:** 42
        - **Split:** Stratified 70/15/15
        - **Latency:** CPU median 100 runs

        🟢 Deep Learning  
        🔵 Classical ML
        """
    )

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Explore the Platform</p>', unsafe_allow_html=True)
nav_col1, nav_col2 = st.columns(2)

NAV = [
    ("🌿 Disease Detection", "Upload a leaf image and get an instant diagnosis with confidence score and Grad-CAM heatmap.", "#2E8B57"),
    ("📊 Model Benchmarking", "Compare all 5 models on accuracy, macro F1, latency, and model size with interactive Plotly charts.", "#4682B4"),
    ("🔬 Explainability", "Understand why the CNN reaches its predictions using Grad-CAM activation heatmaps.", "#9C27B0"),
    ("📚 Dataset & Methodology", "Explore the PlantVillage dataset, 38 disease classes, preprocessing pipeline, and experimental methodology.", "#E65100"),
]
for i, (title, desc, color) in enumerate(NAV):
    col = nav_col1 if i % 2 == 0 else nav_col2
    with col:
        st.markdown(
            f"""
            <div class="nav-card" style="border-left-color:{color};">
                <div class="nav-title">{title}</div>
                <div class="nav-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Supported crops ───────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Supported Crops</p>', unsafe_allow_html=True)
pills_html = " ".join(f'<span class="tech-pill">{crop}</span>' for crop in crops)
st.markdown(pills_html, unsafe_allow_html=True)

# ── Technology stack ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Built With</p>', unsafe_allow_html=True)
tech_items = [
    "Python 3.10", "TensorFlow 2.21", "Keras 3", "Scikit-learn", "XGBoost",
    "OpenCV", "Pillow", "Streamlit", "Plotly", "NumPy", "Pandas", "Joblib",
    "Grad-CAM", "Docker", "GitHub Actions",
]
tech_html = " ".join(f'<span class="tech-pill">{t}</span>' for t in tech_items)
st.markdown(tech_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        ⚠️ <b>Disclaimer:</b> Leaf Lens is a research and educational platform. Predictions are based on the
        PlantVillage dataset (laboratory-quality images) and should <em>not</em> replace expert agronomic advice.
        Always consult a qualified plant pathologist before taking field action.<br><br>
        Dataset: PlantVillage (Hughes & Salathé, 2015) · License: MIT ·
        <a href="https://github.com/MrEGAMERZ/leaf-lens" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
