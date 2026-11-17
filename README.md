# 🌿 Leaf Lens — Crop Disease Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> AI-powered plant disease diagnosis combining classical machine learning, deep transfer learning, and Grad-CAM explainability on the PlantVillage dataset.

---

## ✨ Features

- **5-model comparative ML pipeline** — SVM, Random Forest, XGBoost, MobileNetV2, and EfficientNetB0 trained and benchmarked side-by-side
- **Grad-CAM visual explanations** — saliency heatmaps that highlight the leaf regions driving each CNN prediction
- **Interactive multi-page Streamlit web app** — disease detection, model benchmarking, explainability explorer, and methodology pages
- **38-class PlantVillage disease classification** — covers 14 crop species across healthy and diseased conditions
- **Real-world limitation documentation** — transparent disclosure of dataset bias and confidence calibration caveats

---

## 📊 Benchmark Results

All models evaluated on the same held-out test split (never touched during training or hyperparameter search).

| Model | Accuracy | Macro F1 | Precision | Recall | Latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|
| **EfficientNetB0** (38 Classes) | **97.02%** | **97.02%** | **97.17%** | **96.98%** | 14.85 | 16.82 |
| MobileNetV2 (38 Classes) | 95.97% | 95.82% | 96.10% | 95.75% | 7.32 | 9.45 |
| XGBoost | 94.29% | 92.86% | 94.92% | 91.69% | 4.54 | 2.15 |
| Random Forest | 90.64% | 87.92% | 93.58% | 86.06% | 16.98 | 11.68 |
| SVM | 89.50% | 86.58% | 91.77% | 84.89% | 8.98 | 35.56 |

> ⚠️ All metrics evaluated on a held-out test set of **2,916 images** (15 PlantVillage classes from [GVJahnavi/Plant_village_subset](https://huggingface.co/datasets/GVJahnavi/Plant_village_subset) on Hugging Face). Never used for model selection or hyperparameter tuning.

---

## 🏗️ Architecture

### Deep Learning (CNN)

Two-stage transfer learning is applied to both EfficientNetB0 and MobileNetV2:

1. **Stage 1 — Feature extraction**: the ImageNet-pretrained backbone is frozen; only the new classification head (GlobalAveragePooling → Dense 256 → Dropout → Softmax 38) is trained for 10 epochs.
2. **Stage 2 — Fine-tuning**: the top layers of the backbone are unfrozen and re-trained at a low learning rate (1 × 10⁻⁴) for 15 additional epochs with early stopping and ReduceLROnPlateau.

### Classical ML Feature Pipeline

A fixed 2,364-dimensional hand-crafted feature vector is extracted from each resized leaf image and fed to SVM, Random Forest, and XGBoost:

| Feature Group | Dimensions | Description |
|---|---|---|
| Color Histogram (HSV) | 512 | 8 bins per channel |
| HOG (Histogram of Oriented Gradients) | 1,764 | 9 orientations, 8×8 px cells |
| GLCM (Gray-Level Co-Occurrence) | 88 | 4 directions × 11 Haralick properties |
| **Total** | **2,364** | Concatenated, L2-normalised |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/MrEGAMERZ/leaf-lens.git
cd leaf-lens

# 2. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

The app will open automatically at `http://localhost:8501`.

### Docker

```bash
docker build -t leaf-lens .
docker run -p 8501:8501 leaf-lens
```

---

## 📁 Project Structure

```
leaf-lens/
├── app/
│   ├── streamlit_app.py              # Home page & navigation
│   └── pages/
│       ├── 01_🌿_Disease_Detection.py
│       ├── 02_📊_Model_Benchmarking.py
│       ├── 03_🔬_Explainability.py
│       └── 04_📚_Dataset_Methodology.py
├── src/
│   ├── config.py                     # Centralised paths & hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_loader.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_extractor.py     # Color Hist + HOG + GLCM
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_builder.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py             # CropDiseasePredictor
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py
├── models/
│   ├── best_model.keras              # EfficientNetB0 (tracked in git)
│   ├── mobilenet_model.keras         # MobileNetV2 (tracked in git)
│   ├── svm_model.joblib              # gitignored — large
│   ├── random_forest_model.joblib    # gitignored — large
│   ├── xgboost_model.joblib          # gitignored — large
│   └── classical_feature_pipeline.joblib
├── data/
│   ├── sample_images/               # A few public-domain test leaves
│   └── benchmark_dataset/           # gitignored — downloaded on demand
├── results/
│   ├── model_comparison.json
│   ├── model_comparison_table.csv
│   ├── class_names.json
│   ├── gradcam_prediction.png
│   └── figures/                     # Confusion matrices, ROC curves
├── configs/
│   └── config.yaml
├── scripts/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── train_classical.py
│   └── train_cnn.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Explainable AI (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to both EfficientNetB0 and MobileNetV2 predictions. Gradients of the predicted class score with respect to the final convolutional feature maps are pooled spatially and used to weight those maps. The resulting heatmap is overlaid on the original image, highlighting the pixels that contributed most to the classification.

An example output is saved at `results/gradcam_prediction.png`. In the Streamlit app, Grad-CAM is computed on-the-fly for any uploaded image using `src/explainability/gradcam.py`.

---

## 🌱 Dataset

**PlantVillage** is a publicly available benchmark of 87,000+ controlled-environment leaf photographs across 38 disease / healthy categories and 14 crop species.

| Property | Value |
|---|---|
| Total images | 87,000+ |
| Classes | 38 (14 crops × disease/healthy) |
| Resolution | Variable (resized to 224 × 224 for training) |
| License | CC BY 4.0 |
| Source | [Hughes & Salathé, 2015](https://arxiv.org/abs/1511.08060) |
| Hugging Face subset | [GVJahnavi/Plant_village_subset](https://huggingface.co/datasets/GVJahnavi/Plant_village_subset) |

**Citation:**

```
Hughes, D. P., & Salathé, M. (2015).
An open access repository of images on plant health to enable the development of mobile disease diagnostics.
arXiv:1511.08060
```

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| **Dataset bias** | PlantVillage images are laboratory-quality photographs with uniform backgrounds. Real-farm performance — with field lighting, occlusion, and mixed-disease leaves — may differ substantially. |
| **Not a replacement for experts** | This tool is a decision-support aid only. All diagnoses should be confirmed by a qualified agricultural expert before any treatment decision. |
| **Confidence calibration** | Softmax confidence scores are not probabilities of correctness. A 95% confidence score does not mean the prediction is correct 95% of the time out-of-distribution. |
| **Scope** | The model covers 38 classes across 14 crops. Diseases or crops outside this set will receive an incorrect classification rather than an "unknown" response. |

---

## 🛠️ Technology Stack

| Technology | Role | Version |
|---|---|---|
| Python | Core language | 3.10 |
| TensorFlow / Keras | CNN training & inference | ≥ 2.15 |
| Scikit-learn | SVM, Random Forest, preprocessing | ≥ 1.7 |
| XGBoost | Gradient-boosted classifier | ≥ 3.0 |
| OpenCV | Image processing, HOG, GLCM | ≥ 4.8 |
| Streamlit | Interactive web application | ≥ 1.60 |
| Plotly | Interactive benchmark charts | ≥ 5.18 |
| Grad-CAM | CNN explainability heatmaps | custom (`src/explainability/gradcam.py`) |
| Docker | Container deployment | 3.10-slim base |
| GitHub Actions | CI — lint + syntax check | ubuntu-latest |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The PlantVillage dataset is licensed under **CC BY 4.0**. Attribution is required when redistributing dataset images.
