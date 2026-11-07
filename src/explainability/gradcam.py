"""Explainable AI (XAI): Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Supports CNN architectures (EfficientNetB0, MobileNetV2, ResNet50) with nested
or flat backbones, computes activation heatmaps with respect to the predicted class,
and generates blended overlays for intuitive visualization.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def is_gradcam_supported(model: Any) -> bool:
    """Check if the provided model supports Grad-CAM (must be a differentiable CNN)."""
    if model is None:
        return False
    if hasattr(model, "predict_proba") and not hasattr(model, "layers"):
        return False
    return hasattr(model, "layers") and len(getattr(model, "layers", [])) > 0


def locate_conv_layer_and_base(model: Any) -> tuple[int | None, str | None]:
    """Dynamically locate the target convolutional layer and whether it is inside a nested base model."""
    import tensorflow as tf
    from tensorflow.keras import layers

    conv_types = (layers.Conv2D, layers.DepthwiseConv2D, layers.SeparableConv2D)

    # Check if nested (e.g. model.layers has a tf.keras.Model like EfficientNetB0 or MobileNetV2)
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, conv_types):
                    return i, sublayer.name

    # Flat model
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return None, layer.name

    return None, None


def compute_gradcam_heatmap(
    model: Any,
    image_tensor: np.ndarray,
    target_class_idx: int | None = None,
) -> np.ndarray:
    """Compute Grad-CAM activation heatmap for a given input tensor and target class.

    Args:
        model: Trained Keras CNN model.
        image_tensor: 4D tensor [1, H, W, 3] preprocessed for the model.
        target_class_idx: Target class index. If None, uses top predicted class.

    Returns:
        2D float32 array [H, W] normalized between 0.0 and 1.0.
    """
    import tensorflow as tf

    base_model_idx, conv_layer_name = locate_conv_layer_and_base(model)
    if conv_layer_name is None:
        raise ValueError(
            f"Could not locate a Convolutional layer in model '{model.name}'. Grad-CAM is unsupported."
        )

    tensor = tf.cast(image_tensor, tf.float32)

    if base_model_idx is not None:
        base_model = model.layers[base_model_idx]
        target_conv = base_model.get_layer(conv_layer_name)
        conv_model = tf.keras.Model(base_model.input, target_conv.output)

        conv_idx = next(i for i, l in enumerate(base_model.layers) if l.name == conv_layer_name)
        post_base_layers = base_model.layers[conv_idx + 1:]
        head_layers = model.layers[base_model_idx + 1:]
        pre_base_layers = model.layers[:base_model_idx]

        with tf.GradientTape() as tape:
            x = tensor
            for l in pre_base_layers:
                x = l(x, training=False)

            conv_outputs = conv_model(x, training=False)
            tape.watch(conv_outputs)

            x = conv_outputs
            for l in post_base_layers:
                x = l(x, training=False)

            for l in head_layers:
                x = l(x, training=False)

            preds = x
            if target_class_idx is None:
                target_class_idx = int(tf.argmax(preds[0]))

            loss = preds[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return np.zeros((tensor.shape[1], tensor.shape[2]), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_features = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_features * pooled_grads, axis=-1)

    else:
        # Flat model architecture
        target_conv = model.get_layer(conv_layer_name)
        conv_model = tf.keras.Model(model.input, target_conv.output)
        conv_idx = next(i for i, l in enumerate(model.layers) if l.name == conv_layer_name)
        head_layers = model.layers[conv_idx + 1:]

        with tf.GradientTape() as tape:
            conv_outputs = conv_model(tensor, training=False)
            tape.watch(conv_outputs)

            x = conv_outputs
            for l in head_layers:
                x = l(x, training=False)

            preds = x
            if target_class_idx is None:
                target_class_idx = int(tf.argmax(preds[0]))

            loss = preds[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return np.zeros((tensor.shape[1], tensor.shape[2]), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_features = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_features * pooled_grads, axis=-1)

    # ReLU on heatmap
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)

    if float(max_val) > 0.0:
        heatmap = heatmap / max_val

    return heatmap.numpy().astype(np.float32)


def overlay_gradcam(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend Grad-CAM heatmap onto the original RGB image.

    Args:
        image_rgb: Original RGB image [H, W, 3] uint8.
        heatmap: 2D float32 heatmap [h, w] normalized to [0, 1].
        alpha: Weight for heatmap overlay (0.0 to 1.0).
        colormap: OpenCV colormap constant (default: COLORMAP_JET).

    Returns:
        RGB uint8 image [H, W, 3] with heatmap overlay.
    """
    h, w = image_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))

    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(colored_rgb, alpha, image_rgb, 1.0 - alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_side_by_side_comparison(
    image_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    spacing_px: int = 10,
    divider_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Combine original image and Grad-CAM overlay side-by-side with a separator."""
    h, w = image_rgb.shape[:2]
    separator = np.full((h, spacing_px, 3), divider_color, dtype=np.uint8)
    return np.hstack([image_rgb, separator, overlay_rgb])
