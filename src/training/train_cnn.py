"""Two-Stage Transfer Learning and Fine-Tuning Pipeline for CNN Architectures.

Supports:
- EfficientNetB0
- MobileNetV2
- ResNet50

Implementation:
Stage 1 — Feature Extraction:
- Backbone frozen with ImageNet weights.
- Train classification head with Adam optimizer (learning rate ~1e-3).

Stage 2 — Fine-Tuning:
- Unfreeze top layers (keeping BatchNorm layers frozen).
- Lower learning rate (~5e-5) with EarlyStopping and ModelCheckpoint.
- Save best model checkpoint based on validation accuracy/loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark import (
    compute_evaluation_metrics,
    measure_inference_latency,
    plot_and_save_confusion_matrix,
    save_benchmark_results,
)
from src.models.efficientnet import build_efficientnetb0, unfreeze_efficientnet_for_finetuning
from src.models.mobilenet import build_mobilenetv2, unfreeze_mobilenet_for_finetuning
from src.models.resnet import build_resnet50, unfreeze_resnet_for_finetuning


def get_model_builders(model_name: str) -> tuple[Any, Any]:
    """Retrieve builder and unfreezing functions for a given architecture."""
    name = model_name.lower().replace("-", "").replace("_", "")
    if "efficientnet" in name:
        return build_efficientnetb0, unfreeze_efficientnet_for_finetuning
    elif "mobilenet" in name:
        return build_mobilenetv2, unfreeze_mobilenet_for_finetuning
    elif "resnet" in name:
        return build_resnet50, unfreeze_resnet_for_finetuning
    else:
        raise ValueError(f"Unknown architecture '{model_name}'. Choose EfficientNetB0, MobileNetV2, or ResNet50.")


def create_dataset_pipeline(
    directory: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = False,
    model_name: str = "EfficientNetB0",
    seed: int = 42,
) -> tf.data.Dataset:
    """Create tf.data pipeline with augmentation strictly on training images."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=is_training,
        seed=seed,
    )

    name_lower = model_name.lower()
    if is_training:
        data_augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
                keras.layers.RandomRotation(0.1, seed=seed),
                keras.layers.RandomZoom(0.1, seed=seed),
            ],
            name="augmentation",
        )
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocessing normalization based on architecture
    if "mobilenet" in name_lower or "resnet" in name_lower:
        ds = ds.map(
            lambda x, y: ((tf.cast(x, tf.float32) / 127.5) - 1.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    # EfficientNet has internal rescaling so accepts raw [0, 255]

    return ds.prefetch(tf.data.AUTOTUNE)


def train_cnn(
    model_name: str = "EfficientNetB0",
    data_dir: Path = ROOT / "data" / "benchmark_dataset",
    output_models_dir: Path = ROOT / "models",
    output_results_dir: Path = ROOT / "results",
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    stage1_epochs: int = 5,
    stage2_epochs: int = 8,
    unfreeze_layers: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute two-stage training strategy and evaluate on held-out test split."""
    tf.keras.utils.set_random_seed(seed)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    output_models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_results_dir / "metrics"
    figures_dir = output_results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    num_classes = len(class_names)
    print(f"Training {model_name} on {num_classes} classes...")

    # Load datasets
    train_ds = create_dataset_pipeline(
        train_dir, image_size=image_size, batch_size=batch_size, is_training=True, model_name=model_name, seed=seed
    )
    val_ds = create_dataset_pipeline(
        val_dir, image_size=image_size, batch_size=batch_size, is_training=False, model_name=model_name, seed=seed
    )
    test_ds = create_dataset_pipeline(
        test_dir, image_size=image_size, batch_size=batch_size, is_training=False, model_name=model_name, seed=seed
    )

    build_fn, unfreeze_fn = get_model_builders(model_name)
    model = build_fn(num_classes=num_classes, image_size=image_size, weights="imagenet")

    # STAGE 1: Train Head with Frozen Backbone
    print(f"\n[Stage 1] Training classification head for {model_name} (Backbone Frozen)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        verbose=1,
    )

    # STAGE 2: Fine-Tuning Top Layers
    print(f"\n[Stage 2] Fine-tuning top {unfreeze_layers} layers of {model_name} with low learning rate...")
    model = unfreeze_fn(model, unfreeze_layers=unfreeze_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    slug = model_name.lower().replace("-", "_").replace(" ", "_")
    checkpoint_path = output_models_dir / f"{slug}_best.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model if checkpoint wasn't triggered
    if not checkpoint_path.exists():
        model.save(checkpoint_path)

    print(f"\nEvaluating {model_name} on held-out test split...")
    best_model = keras.models.load_model(checkpoint_path)

    # Predict test set
    y_true_list = []
    y_pred_list = []
    for images, labels in test_ds:
        preds = best_model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(np.argmax(preds, axis=-1))

    sample_batch = next(iter(test_ds))[0][:1].numpy()
    latency = measure_inference_latency(
        best_model,
        sample_batch,
        is_cnn=True,
        num_warmup=5,
        num_runs=30,
    )

    metrics = compute_evaluation_metrics(
        y_true=y_true_list,
        y_pred=y_pred_list,
        class_names=class_names,
        model_name=model_name,
        model_path=checkpoint_path,
        latency_stats=latency,
    )

    save_benchmark_results(metrics, metrics_dir)

    cm_path = figures_dir / f"{slug}_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names,
        cm_path,
        title=f"{model_name} Confusion Matrix",
    )

    print(
        f"Completed {model_name} benchmark: Accuracy={metrics['accuracy']:.4f}, "
        f"Macro F1={metrics['macro_f1']:.4f}, Latency={metrics['latency']['mean_latency_ms']:.2f}ms"
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MobileNetV2", choices=["MobileNetV2", "EfficientNetB0", "ResNet50"])
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    args = parser.parse_args()
    train_cnn(
        model_name=args.model,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )
