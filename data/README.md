# Data

This directory holds dataset artifacts used by Leaf Lens.

## What's Here

| Path | Description |
|---|---|
| `sample_images/` | 3 sample images per class (15 classes × 3 = 45 images) for demo use |
| `benchmark_dataset/` | Full extracted train/val/test split *(git-ignored — regenerate below)* |
| `test_subset.parquet` | Raw Hugging Face parquet download *(git-ignored — 45 MB)* |

## Regenerating the Benchmark Dataset

```bash
PYTHONPATH=. python scripts/prepare_dataset.py
```

The script downloads from `GVJahnavi/Plant_village_subset` on Hugging Face (15 PlantVillage classes, 2,916 test images).

## Full PlantVillage Dataset

The full 38-class PlantVillage dataset (~1.5 GB) is available at:
- [Kaggle — New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

Place it at `data/new_plant_diseases_dataset/` to train CNN models from scratch.
