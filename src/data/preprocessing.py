"""Image preprocessing and validation utilities for Crop Disease Intelligence."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when an uploaded file or image path is corrupted, unreadable, or not a valid image."""
    pass


class UnsupportedFormatError(ValueError):
    """Raised when the image format is not among supported formats."""
    pass


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".JPG", ".JPEG", ".PNG"}


def clean_label(label: str) -> tuple[str, str]:
    """Parse raw dataset label into human-readable Crop and Condition strings.

    Example:
        'Tomato___Early_blight' -> ('Tomato', 'Early blight')
        'Apple___healthy' -> ('Apple', 'healthy')
        'Corn_(maize)___Common_rust_' -> ('Corn (maize)', 'Common rust')
    """
    if "___" in label:
        crop_raw, condition_raw = label.split("___", 1)
    else:
        crop_raw, condition_raw = label, "Unknown"

    crop = crop_raw.replace("_", " ").strip()
    condition = condition_raw.replace("_", " ").strip()

    # Clean multi-space artifacts and trailing underscores
    crop = " ".join(crop.split())
    condition = " ".join(condition.split())

    return crop, condition


def validate_image_file(file_or_path: str | Path | BinaryIO | bytes) -> bool:
    """Validate that the input is a readable, non-corrupted image.

    Raises:
        InvalidImageError: If the image is corrupt, unreadable, or empty.
        UnsupportedFormatError: If the format is unsupported.
    """
    try:
        if isinstance(file_or_path, (str, Path)):
            path = Path(file_or_path)
            if not path.exists():
                raise InvalidImageError(f"File not found: {path}")
            if path.suffix.lower() not in [ext.lower() for ext in SUPPORTED_EXTENSIONS]:
                raise UnsupportedFormatError(
                    f"Unsupported extension '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
            with Image.open(path) as img:
                img.verify()
            # Second check: actual read
            with Image.open(path) as img:
                img.load()
                if img.size[0] <= 0 or img.size[1] <= 0:
                    raise InvalidImageError("Image has invalid dimensions (width or height <= 0).")
        elif isinstance(file_or_path, bytes):
            if len(file_or_path) == 0:
                raise InvalidImageError("Image byte buffer is empty.")
            with Image.open(io.BytesIO(file_or_path)) as img:
                img.verify()
            with Image.open(io.BytesIO(file_or_path)) as img:
                img.load()
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            pos = file_or_path.tell() if hasattr(file_or_path, "tell") else None
            data = file_or_path.read()
            if hasattr(file_or_path, "seek") and pos is not None:
                file_or_path.seek(pos)
            if len(data) == 0:
                raise InvalidImageError("Uploaded file is empty.")
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
            with Image.open(io.BytesIO(data)) as img:
                img.load()
        return True
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise InvalidImageError(f"Corrupted or invalid image file: {e}") from e


def load_image_rgb(file_or_path: str | Path | BinaryIO | bytes) -> np.ndarray:
    """Load image as an RGB uint8 NumPy array [H, W, 3]."""
    validate_image_file(file_or_path)
    if isinstance(file_or_path, (str, Path)):
        img = Image.open(file_or_path).convert("RGB")
    elif isinstance(file_or_path, bytes):
        img = Image.open(io.BytesIO(file_or_path)).convert("RGB")
    else:
        pos = file_or_path.tell() if hasattr(file_or_path, "tell") else None
        data = file_or_path.read()
        if hasattr(file_or_path, "seek") and pos is not None:
            file_or_path.seek(pos)
        img = Image.open(io.BytesIO(data)).convert("RGB")

    return np.array(img, dtype=np.uint8)


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize RGB image to target size (width, height) using bilinear interpolation."""
    if image.shape[0] != target_size[1] or image.shape[1] != target_size[0]:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return image


def preprocess_for_model(
    image: np.ndarray,
    model_name: str = "EfficientNetB0",
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Preprocess RGB image into a 4D batch tensor [1, H, W, 3] for deep learning models.

    EfficientNetB0:
        Accepts raw [0, 255] float32 image tensors because EfficientNet incorporates
        its own internal rescaling layer.
    MobileNetV2:
        Requires normalization to [-1, 1]: (image / 127.5) - 1.0.
    ResNet50:
        Requires zero-centering with ImageNet mean subtraction in BGR or standard [-1, 1].
    """
    resized = resize_image(image, target_size)
    img_float = resized.astype(np.float32)

    name_lower = model_name.lower()
    if "mobilenet" in name_lower:
        processed = (img_float / 127.5) - 1.0
    elif "resnet" in name_lower:
        # Standard ResNet50 caffe-style BGR mean subtraction or tf-style [-1, 1]
        processed = (img_float / 127.5) - 1.0
    elif "efficientnet" in name_lower:
        # EfficientNet has internal rescaling
        processed = img_float
    else:
        # Default fallback: [0, 1] normalization
        processed = img_float / 255.0

    return np.expand_dims(processed, axis=0)
