"""Feature extraction module for classical machine learning baselines.

Extracts:
1. Color Histograms (HSV and RGB color spaces, normalized)
2. Histogram of Oriented Gradients (HOG)
3. Gray-Level Co-occurrence Matrix (GLCM) Haralick Texture Features
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_color_histogram(
    image_rgb: np.ndarray,
    bins: tuple[int, int, int] = (8, 8, 8),
    color_space: str = "HSV",
) -> np.ndarray:
    """Extract normalized 3D color histogram feature vector.

    Args:
        image_rgb: Input RGB image [H, W, 3] uint8.
        bins: Number of bins for each of the three channels.
        color_space: "HSV" or "RGB".

    Returns:
        1D float32 array of shape (bins[0] * bins[1] * bins[2],).
    """
    if color_space.upper() == "HSV":
        converted = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        ranges = [0, 180, 0, 256, 0, 256]
    else:
        converted = image_rgb
        ranges = [0, 256, 0, 256, 0, 256]

    hist = cv2.calcHist(
        [converted],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(bins),
        ranges=ranges,
    )
    hist_norm = cv2.normalize(hist, hist).flatten()
    return hist_norm.astype(np.float32)


def extract_hog_features(
    image_rgb: np.ndarray,
    orientations: int = 9,
    cell_size: tuple[int, int] = (16, 16),
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Extract Histogram of Oriented Gradients (HOG) feature vector.

    Computes spatial gradient magnitude and orientation, accumulates histograms
    over spatial cells, and applies L2-Hys block normalization (Dalal & Triggs).
    """
    resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0  # Unsigned gradients [0, 180)

    n_cells_y = target_size[1] // cell_size[1]
    n_cells_x = target_size[0] // cell_size[0]
    hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    bin_width = 180.0 / orientations
    bin_idx = (ang / bin_width).astype(int) % orientations

    for y in range(n_cells_y):
        y0, y1 = y * cell_size[1], (y + 1) * cell_size[1]
        for x in range(n_cells_x):
            x0, x1 = x * cell_size[0], (x + 1) * cell_size[0]
            c_mag = mag[y0:y1, x0:x1]
            c_bin = bin_idx[y0:y1, x0:x1]
            for o in range(orientations):
                hist[y, x, o] = np.sum(c_mag[c_bin == o])

    # 2x2 Block L2 normalization
    eps = 1e-5
    blocks = []
    for y in range(n_cells_y - 1):
        for x in range(n_cells_x - 1):
            block = hist[y:y + 2, x:x + 2].flatten()
            norm = np.sqrt(np.sum(block ** 2) + eps ** 2)
            blocks.append(block / norm)

    if not blocks:
        return hist.flatten()
    return np.concatenate(blocks).astype(np.float32)


def compute_fast_glcm(
    gray: np.ndarray,
    levels: int = 16,
    dx: int = 1,
    dy: int = 0,
) -> np.ndarray:
    """Compute normalized 2D Gray-Level Co-occurrence Matrix for quantized grayscale image.

    Args:
        gray: Grayscale 2D image [H, W] uint8.
        levels: Number of quantized gray levels (e.g. 16 for speed and robustness).
        dx: Horizontal pixel displacement.
        dy: Vertical pixel displacement.

    Returns:
        2D float32 array of shape (levels, levels) summing to 1.0.
    """
    # Quantize gray image from [0, 255] to [0, levels - 1]
    quantized = (gray // (256 // levels)).astype(np.int32)
    h, w = quantized.shape

    # Source and shifted windows
    if dy >= 0:
        y1_src, y2_src = 0, h - dy
        y1_dst, y2_dst = dy, h
    else:
        y1_src, y2_src = -dy, h
        y1_dst, y2_dst = 0, h + dy

    if dx >= 0:
        x1_src, x2_src = 0, w - dx
        x1_dst, x2_dst = dx, w
    else:
        x1_src, x2_src = -dx, w
        x1_dst, x2_dst = 0, w + dx

    src_patch = quantized[y1_src:y2_src, x1_src:x2_src].flatten()
    dst_patch = quantized[y1_dst:y2_dst, x1_dst:x2_dst].flatten()

    indices = src_patch * levels + dst_patch
    glcm = np.bincount(indices, minlength=levels * levels).reshape((levels, levels)).astype(np.float32)

    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def extract_glcm_features(
    image_rgb: np.ndarray,
    target_size: tuple[int, int] = (128, 128),
    levels: int = 16,
) -> np.ndarray:
    """Extract standard Haralick texture features from GLCM across 4 directions (0, 45, 90, 135 deg).

    Features computed per direction:
    - Contrast
    - Dissimilarity
    - Homogeneity
    - Energy (Angular Second Moment)
    - Correlation
    - Entropy

    Returns:
        1D float32 array of length 24 (6 features * 4 directions).
    """
    resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]  # 0, 45, 90, 135 degrees
    features: list[float] = []

    i_indices, j_indices = np.indices((levels, levels))
    diff_abs = np.abs(i_indices - j_indices).astype(np.float32)
    diff_sq = (diff_abs ** 2).astype(np.float32)
    inv_diff = 1.0 / (1.0 + diff_sq)

    for dx, dy in directions:
        glcm = compute_fast_glcm(gray, levels=levels, dx=dx, dy=dy)
        glcm_sym = (glcm + glcm.T) / 2.0  # Symmetrical GLCM

        # Contrast: sum(P * (i - j)^2)
        contrast = float(np.sum(glcm_sym * diff_sq))

        # Dissimilarity: sum(P * |i - j|)
        dissimilarity = float(np.sum(glcm_sym * diff_abs))

        # Homogeneity: sum(P / (1 + (i - j)^2))
        homogeneity = float(np.sum(glcm_sym * inv_diff))

        # Energy / ASM: sum(P^2)
        energy = float(np.sum(glcm_sym ** 2))

        # Mean and std dev for correlation
        mu_i = float(np.sum(i_indices * glcm_sym))
        mu_j = float(np.sum(j_indices * glcm_sym))
        sigma_i = float(np.sqrt(np.sum(((i_indices - mu_i) ** 2) * glcm_sym)) + 1e-8)
        sigma_j = float(np.sqrt(np.sum(((j_indices - mu_j) ** 2) * glcm_sym)) + 1e-8)
        correlation = float(np.sum((i_indices - mu_i) * (j_indices - mu_j) * glcm_sym) / (sigma_i * sigma_j))

        # Entropy: -sum(P * log2(P))
        non_zero = glcm_sym[glcm_sym > 0]
        entropy = float(-np.sum(non_zero * np.log2(non_zero))) if len(non_zero) > 0 else 0.0

        features.extend([contrast, dissimilarity, homogeneity, energy, correlation, entropy])

    return np.array(features, dtype=np.float32)


def extract_combined_features(
    image_rgb: np.ndarray,
    color_bins: tuple[int, int, int] = (8, 8, 8),
    use_hog: bool = True,
    use_glcm: bool = True,
) -> np.ndarray:
    """Extract combined feature vector: Color Histogram + HOG + GLCM Texture features.

    Args:
        image_rgb: Input RGB image [H, W, 3] uint8.
        color_bins: Bin counts for HSV histogram.
        use_hog: Whether to include HOG descriptors.
        use_glcm: Whether to include GLCM texture features.

    Returns:
        1D float32 array concatenated feature representation.
    """
    parts = []

    # 1. Color Histogram (HSV)
    hist_hsv = extract_color_histogram(image_rgb, bins=color_bins, color_space="HSV")
    parts.append(hist_hsv)

    # 2. Color Histogram (RGB)
    hist_rgb = extract_color_histogram(image_rgb, bins=(4, 4, 4), color_space="RGB")
    parts.append(hist_rgb)

    # 3. HOG Features
    if use_hog:
        hog = extract_hog_features(image_rgb)
        parts.append(hog)

    # 4. GLCM Texture Features
    if use_glcm:
        glcm = extract_glcm_features(image_rgb)
        parts.append(glcm)

    return np.concatenate(parts).astype(np.float32)
