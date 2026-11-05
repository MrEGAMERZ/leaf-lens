"""ResNet50 architecture with two-stage transfer learning support."""

from __future__ import annotations

from typing import Any


def build_resnet50(
    num_classes: int = 38,
    image_size: tuple[int, int] = (224, 224),
    dropout_rate: float = 0.3,
    weights: str | None = "imagenet",
) -> Any:
    """Build ResNet50 classification model with frozen backbone for Stage 1."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50

    inputs = layers.Input(shape=(image_size[1], image_size[0], 3), name="input_image")

    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
    )
    base_model.trainable = False  # Freeze for Stage 1

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropDisease_ResNet50")
    return model


def unfreeze_resnet_for_finetuning(model: Any, unfreeze_layers: int = 20) -> Any:
    """Unfreeze top layers for Stage 2 fine-tuning while keeping BatchNorm frozen."""
    import tensorflow as tf
    from tensorflow.keras import layers

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    target = base_model if base_model is not None else model
    target.trainable = True

    layers_to_freeze = max(0, len(target.layers) - unfreeze_layers)
    for layer in target.layers[:layers_to_freeze]:
        layer.trainable = False

    for layer in target.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    return model


def get_resnet_last_conv_layer(model: Any) -> str:
    """Return the name of the final convolutional layer for Grad-CAM."""
    from tensorflow.keras import layers

    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            return layer.name
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, (layers.Conv2D, layers.DepthwiseConv2D)):
                    return sublayer.name

    return "conv5_block3_out"
