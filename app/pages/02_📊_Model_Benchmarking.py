"""Leaf Lens — Model Benchmarking Page."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Benchmarking | Leaf Lens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .page-title {
            font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(135deg, #2E8B57, #4682B4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .metric-card {
            background: #f8f9fa; border-radius: 12px; padding: 1rem 1.2rem;
            border-left: 4px solid #2E8B57; margin-bottom: 0.6rem;
        }
        .insight-box {
            background: linear-gradient(135deg, #e8f5e9, #e3f2fd);
            border-radius: 10px; padding: 1rem 1.4rem;
            border-left: 4px solid #2E8B57; margin: 1rem 0;
        }
        .section-header {
            font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
            margin: 1.4rem 0 0.6rem 0; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(ROOT, "results", "model_comparison.json")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")


@st.cache_data
def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


results = load_results()

# Build a tidy DataFrame
rows = []
display_names = {
    "EfficientNetB0 (38 Classes)": "EfficientNetB0",
    "MobileNetV2 (38 Classes)": "MobileNetV2",
    "XGBoost": "XGBoost",
    "Random Forest": "Random Forest",
    "SVM": "SVM",
}
for key, vals in results.items():
    rows.append(
        {
            "Model": display_names.get(key, key),
            "Type": vals["type"],
            "Accuracy (%)": round(vals["accuracy"] * 100, 2),
            "Macro F1 (%)": round(vals["macro_f1"] * 100, 2),
            "Precision (%)": round(vals["macro_precision"] * 100, 2),
            "Recall (%)": round(vals["macro_recall"] * 100, 2),
            "Latency (ms)": vals["latency_ms"],
            "Size (MB)": vals["model_size_mb"],
        }
    )
df = pd.DataFrame(rows)
dl_df = df[df["Type"] == "Deep Learning"]
cl_df = df[df["Type"] == "Classical ML"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">📊 Model Benchmarking</p>', unsafe_allow_html=True)
st.markdown(
    "Comparative evaluation of **5 models** (deep learning + classical ML) on a held-out test set "
    "of **2,916 images** from 15 PlantVillage classes. No model selection used test data.",
)
st.divider()

# ── Top metric cards ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cols = [c1, c2, c3, c4, c5]
colors = ["#2E8B57", "#3CB371", "#4682B4", "#6495ED", "#87CEEB"]
for col, row, color in zip(cols, df.itertuples(), colors):
    with col:
        st.markdown(
            f"""
            <div style="background:#f8f9fa;border-radius:10px;padding:0.8rem;
                        border-top:4px solid {color};text-align:center;">
                <div style="font-size:0.8rem;color:#666;font-weight:600;">{row.Model}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{color};">{row._3:.1f}%</div>
                <div style="font-size:0.75rem;color:#888;">accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Accuracy & F1 bar chart ───────────────────────────────────────────────────
st.markdown('<p class="section-header">Accuracy vs Macro F1 Score</p>', unsafe_allow_html=True)

fig_bar = go.Figure()
model_colors = {
    "EfficientNetB0": "#2E8B57",
    "MobileNetV2": "#3CB371",
    "XGBoost": "#4682B4",
    "Random Forest": "#6495ED",
    "SVM": "#87CEEB",
}
fig_bar.add_trace(
    go.Bar(
        name="Accuracy (%)",
        x=df["Model"],
        y=df["Accuracy (%)"],
        marker_color=[model_colors[m] for m in df["Model"]],
        text=df["Accuracy (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        offsetgroup=0,
    )
)
fig_bar.add_trace(
    go.Bar(
        name="Macro F1 (%)",
        x=df["Model"],
        y=df["Macro F1 (%)"],
        marker_color=[model_colors[m] for m in df["Model"]],
        marker_pattern_shape="x",
        text=df["Macro F1 (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        offsetgroup=1,
        opacity=0.75,
    )
)
fig_bar.update_layout(
    barmode="group",
    yaxis=dict(title="Score (%)", range=[80, 100]),
    xaxis_title="Model",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=40, b=20),
)
fig_bar.update_xaxes(showgrid=False)
fig_bar.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig_bar, use_container_width=True)

# ── Latency vs Accuracy scatter ───────────────────────────────────────────────
st.markdown('<p class="section-header">Latency vs Accuracy Tradeoff</p>', unsafe_allow_html=True)
st.caption("Bubble size = model disk size (MB). Hover for details.")

fig_scatter = px.scatter(
    df,
    x="Latency (ms)",
    y="Accuracy (%)",
    size="Size (MB)",
    color="Type",
    text="Model",
    color_discrete_map={"Deep Learning": "#2E8B57", "Classical ML": "#4682B4"},
    hover_data=["Macro F1 (%)", "Size (MB)", "Latency (ms)"],
    size_max=50,
)
fig_scatter.update_traces(textposition="top center", marker=dict(opacity=0.85))
fig_scatter.update_layout(
    height=420,
    yaxis=dict(title="Accuracy (%)", range=[87, 100]),
    xaxis_title="Inference Latency (ms)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=20, b=20),
)
fig_scatter.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
fig_scatter.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Full metrics table ────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Full Metrics Table</p>', unsafe_allow_html=True)
display_df = df[["Model", "Type", "Accuracy (%)", "Macro F1 (%)", "Precision (%)", "Recall (%)", "Latency (ms)", "Size (MB)"]].copy()

def _highlight_type(row: pd.Series) -> list[str]:
    if row["Type"] == "Deep Learning":
        return ["background-color: #e8f5e9"] * len(row)
    return ["background-color: #e3f2fd"] * len(row)

st.dataframe(
    display_df.style.apply(_highlight_type, axis=1).format(
        {
            "Accuracy (%)": "{:.2f}",
            "Macro F1 (%)": "{:.2f}",
            "Precision (%)": "{:.2f}",
            "Recall (%)": "{:.2f}",
            "Latency (ms)": "{:.2f}",
            "Size (MB)": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ── Key insights ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        <div class="insight-box">
            <b>🏆 Best Overall: EfficientNetB0</b><br>
            97.02% accuracy and 97.02% macro F1 on 15-class test set.
            Highest across all metrics but requires 14.85 ms per image.
        </div>
        <div class="insight-box">
            <b>⚡ Fastest Deep Learning: MobileNetV2</b><br>
            96% accuracy at only 7.3 ms per image — 2× faster than EfficientNetB0,
            with minimal accuracy loss. Ideal for edge deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
        <div class="insight-box">
            <b>🎯 Best Classical ML: XGBoost</b><br>
            94.3% accuracy with the fastest inference (4.5 ms) and smallest disk footprint (2.15 MB).
            Gradient boosting on handcrafted features remains highly competitive.
        </div>
        <div class="insight-box">
            <b>📐 Feature Engineering Gap</b><br>
            SVM and Random Forest on 2,364-dim hand-crafted features (Color Hist + HOG + GLCM)
            reach 90–91%, only 6–7% below the best deep learning model.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Confusion matrices ────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Confusion Matrices — Classical ML</p>', unsafe_allow_html=True)
st.caption("Evaluated on the same held-out test set for all models.")

cm_col1, cm_col2, cm_col3 = st.columns(3)
cm_files = {
    "SVM": "svm_confusion_matrix.png",
    "Random Forest": "random_forest_confusion_matrix.png",
    "XGBoost": "xgboost_confusion_matrix.png",
}
for col, (name, fname) in zip([cm_col1, cm_col2, cm_col3], cm_files.items()):
    path = os.path.join(FIGURES_DIR, fname)
    with col:
        st.markdown(f"**{name}**")
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info(f"Confusion matrix for {name} not available.")

# ── Methodology note ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    **📋 Methodology Note**

    All metrics were computed on a **held-out test set** of 2,916 images from 15 PlantVillage classes
    sourced from [GVJahnavi/Plant_village_subset](https://huggingface.co/datasets/GVJahnavi/Plant_village_subset)
    on Hugging Face. The test set was **never used for model selection, feature scaling, or hyperparameter tuning**.
    Feature preprocessing (StandardScaler) was fit on training data only.
    Random seed: 42. Stratified train/val/test split.

    Latency is measured as median inference time over 100 warmup-then-timed runs on CPU.
    """,
    unsafe_allow_html=False,
)
