# Contributing to Leaf Lens

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/MrEGAMERZ/leaf-lens.git
cd leaf-lens
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
leaf-lens/
├── app/                  # Streamlit multi-page app
│   ├── streamlit_app.py  # Home page
│   └── pages/            # Detection, Benchmarking, Explainability, Dataset, About
├── src/                  # Core ML library
│   ├── data/             # Preprocessing, augmentation, dataset utilities
│   ├── features/         # Classical ML feature extraction (HOG, GLCM, color hist)
│   ├── models/           # CNN and classical model builders
│   ├── training/         # Training pipelines
│   ├── evaluation/       # Metrics and benchmarking
│   ├── explainability/   # Grad-CAM implementation
│   └── inference/        # Unified predictor
├── scripts/              # Data preparation and evaluation scripts
├── configs/              # YAML configuration
├── models/               # Saved model artifacts
└── results/              # Evaluation outputs, figures, metrics
```

## Running the App

```bash
streamlit run app/streamlit_app.py
```

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings on all modules and classes
- Imports at the top of every file (no inline imports)

## Pull Requests

1. Fork the repo and create a feature branch.
2. Keep PRs focused — one logical change per PR.
3. Run `python -m py_compile` on any files you modify.
4. Update `results/` or `README.md` if you change model training or evaluation logic.
