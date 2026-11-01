"""Data Augmentation module for training images.

Ensures augmentations are applied strictly to training images and never
to validation or test splits, preserving evaluation validity and preventing data leakage.
"""

from __future__ import annotations

import random
import cv2
import numpy as np


class TrainingImageAugmenter:
    """Configurable image augmentation pipeline using OpenCV and NumPy."""

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_range: float = 20.0,
        zoom_range: tuple[float, float] = (0.9, 1.1),
        brightness_range: tuple[float, float] = (0.85, 1.15),
        seed: int | None = None,
    ) -> None:
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def augment(self, image_rgb: np.ndarray) -> np.ndarray:
        """Apply random geometric and photometric augmentations to an RGB image."""
        img = image_rgb.copy()
        h, w = img.shape[:2]

        # 1. Random Horizontal Flip
        if self.horizontal_flip and random.random() > 0.5:
            img = cv2.flip(img, 1)

        # 2. Random Vertical Flip
        if self.vertical_flip and random.random() > 0.5:
            img = cv2.flip(img, 0)

        # 3. Random Rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        # 4. Random Zoom / Scale
        if self.zoom_range[0] != 1.0 or self.zoom_range[1] != 1.0:
            scale = random.uniform(self.zoom_range[0], self.zoom_range[1])
            if scale != 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if scale > 1.0:
                    # Crop center
                    x_start = (new_w - w) // 2
                    y_start = (new_h - h) // 2
                    img = resized[y_start:y_start + h, x_start:x_start + w]
                else:
                    # Pad border
                    pad_w = (w - new_w) // 2
                    pad_h = (h - new_h) // 2
                    img = cv2.copyMakeBorder(
                        resized,
                        pad_h,
                        h - new_h - pad_h,
                        pad_w,
                        w - new_w - pad_w,
                        borderType=cv2.BORDER_REFLECT_101,
                    )

        # 5. Random Brightness & Contrast
        if self.brightness_range[0] != 1.0 or self.brightness_range[1] != 1.0:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.augment(image_rgb)
