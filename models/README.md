# Models

This directory contains trained model artifacts for the Leaf Lens platform.

## Files

| File | Description | Size | Accuracy |
|---|---|---|---|
| `best_model.keras` | EfficientNetB0 — two-stage transfer learning on PlantVillage 38-class | ~17 MB | 97.02% |
| `mobilenet_model.keras` | MobileNetV2 — two-stage transfer learning *(if present)* | ~9 MB | 95.97% |
| `svm_model.joblib` | SVM with RBF kernel on 2,364-dim feature vectors *(excluded from git — large)* | ~36 MB | 89.50% |
| `random_forest_model.joblib` | Random Forest 150 estimators *(excluded from git)* | ~12 MB | 90.64% |
| `xgboost_model.joblib` | XGBoost 150 estimators *(excluded from git)* | ~2 MB | 94.29% |
| `classical_feature_pipeline.joblib` | StandardScaler fit on training features *(excluded from git)* | <1 MB | — |

## Regenerating Models

To retrain classical ML models from scratch:

```bash
PYTHONPATH=. python scripts/prepare_dataset.py   # download & extract dataset
PYTHONPATH=. python src/training/train_classical.py
```

To retrain CNN models (requires full PlantVillage dataset):

```bash
PYTHONPATH=. python src/training/train_cnn.py --arch efficientnetb0
PYTHONPATH=. python src/training/train_cnn.py --arch mobilenetv2
```

## Notes

- All models are trained with **seed = 42** for reproducibility.
- The `.gitignore` excludes large `.joblib` files. Download them by running the training scripts.
- `best_model.keras` is committed to the repository for immediate demo use.
