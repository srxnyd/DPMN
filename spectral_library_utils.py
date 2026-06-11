from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


def normalize_numpy_minmax(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array_min = float(array.min())
    array_max = float(array.max())
    return (array - array_min) / (array_max - array_min + eps)


def _reshape_library_to_row_major(array: np.ndarray, bands: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D spectral library array, got shape {array.shape}")

    if array.shape[1] == bands:
        return array.astype(np.float32)
    if array.shape[0] == bands:
        return array.transpose(1, 0).astype(np.float32)
    raise ValueError(f"Cannot align spectral library shape {array.shape} with band count {bands}")


def try_extract_original_spectral_library(mat: Dict[str, np.ndarray], bands: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    candidates = [
        "A",
        "endmembers",
        "Endmembers",
        "endmember",
        "Endmember",
        "E",
    ]
    for key in candidates:
        if key not in mat:
            continue
        array = np.asarray(mat[key], dtype=np.float32)
        try:
            library = _reshape_library_to_row_major(array, bands)
        except ValueError:
            continue
        return library, key
    return None, None


def estimate_spectral_library_full_image(
    data: np.ndarray,
    total_clusters: int,
    random_state: int = 0,
    sample_limit: int = 50000,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, got shape {data.shape}")

    height, width, bands = data.shape
    pixels = data.reshape(height * width, bands)
    if pixels.shape[0] == 0:
        raise ValueError("Input image contains no pixels.")

    total_clusters = int(max(1, min(total_clusters, pixels.shape[0])))
    if sample_limit is not None and pixels.shape[0] > sample_limit:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(pixels.shape[0], size=sample_limit, replace=False)
        fit_pixels = pixels[indices]
    else:
        fit_pixels = pixels

    kmeans = KMeans(n_clusters=total_clusters, n_init=10, random_state=random_state)
    centers = kmeans.fit(fit_pixels).cluster_centers_
    return centers.astype(np.float32)


def resolve_spectral_library(
    mat: Dict[str, np.ndarray],
    data: np.ndarray,
    target_clusters: int,
    background_clusters: int,
    random_state: int = 0,
    sample_limit: int = 50000,
) -> Tuple[np.ndarray, str]:
    bands = int(data.shape[2])
    original_library, source_key = try_extract_original_spectral_library(mat, bands=bands)
    if original_library is not None:
        return original_library, f"original:{source_key}"

    total_clusters = int(max(1, target_clusters + background_clusters))
    estimated_library = estimate_spectral_library_full_image(
        data=data,
        total_clusters=total_clusters,
        random_state=random_state,
        sample_limit=sample_limit,
    )
    return estimated_library, f"estimated:kmeans_full_image_{total_clusters}"
