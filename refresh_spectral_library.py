from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io

from spectral_library_utils import estimate_spectral_library_full_image


def load_training_cube(mat_path: Path) -> tuple[np.ndarray, dict]:
    mat = scipy.io.loadmat(mat_path)
    if "image" not in mat:
        raise KeyError(f"{mat_path} does not contain 'image'")
    stored_image = np.asarray(mat["image"], dtype=np.float32)
    if stored_image.ndim != 3:
        raise ValueError(f"{mat_path} image shape is invalid: {stored_image.shape}")
    data_hwb = np.transpose(stored_image, (2, 1, 0))
    return data_hwb, mat


def rewrite_spectral_library(
    mat_path: Path,
    target_clusters: int,
    background_clusters: int,
    random_state: int,
) -> dict:
    data_hwb, mat = load_training_cube(mat_path)
    A = estimate_spectral_library_full_image(
        data=data_hwb,
        total_clusters=target_clusters + background_clusters,
        random_state=random_state,
    )
    mat["A"] = A.astype(np.float32)
    mat["A_source_info"] = np.array(
        [f"estimated:kmeans_full_image_{target_clusters + background_clusters}"],
        dtype=object,
    )
    scipy.io.savemat(str(mat_path), mat, do_compression=True)
    return {
        "path": str(mat_path),
        "image_shape": tuple(data_hwb.shape),
        "A_shape": tuple(A.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-estimate A for existing DPMN training .mat files without using labels.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing DPMN-style .mat files with image/mask/A.")
    parser.add_argument("--pattern", default="urban_abu_*.mat", help="Glob pattern for files to rewrite.")
    parser.add_argument("--target-clusters", type=int, default=3, help="Used with background_clusters to define the total KMeans cluster count.")
    parser.add_argument("--background-clusters", type=int, default=5, help="Used with target_clusters to define the total KMeans cluster count.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for KMeans.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    files = sorted(dataset_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{args.pattern}' in {dataset_dir}")

    for mat_path in files:
        result = rewrite_spectral_library(
            mat_path=mat_path,
            target_clusters=args.target_clusters,
            background_clusters=args.background_clusters,
            random_state=args.random_state,
        )
        print(f"REWROTE {result['path']} image={result['image_shape']} A={result['A_shape']}")


if __name__ == "__main__":
    main()
