from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.io

from spectral_library_utils import resolve_spectral_library


def convert_hydice_mat_file(
    src_path: str,
    dst_path: str,
    target_clusters: int = 3,
    background_clusters: int = 5,
    random_state: int = 0,
) -> dict:
    mat = scipy.io.loadmat(src_path)
    data = np.asarray(mat["data"], dtype=np.float32)
    anomaly_map = np.asarray(mat["map"])

    if data.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, got shape {data.shape}")
    if anomaly_map.shape != data.shape[:2]:
        raise ValueError(f"Mask shape {anomaly_map.shape} does not match spatial shape {data.shape[:2]}")

    image = np.transpose(data, (2, 1, 0)).astype(np.float32)
    mask = np.transpose(anomaly_map, (1, 0)).astype(np.uint8)
    A, a_source = resolve_spectral_library(
        mat=mat,
        data=data,
        target_clusters=target_clusters,
        background_clusters=background_clusters,
        random_state=random_state,
    )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    scipy.io.savemat(dst_path, {"image": image, "mask": mask, "A": A}, do_compression=True)
    return {
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "image_shape": tuple(image.shape),
        "mask_shape": tuple(mask.shape),
        "A_shape": tuple(A.shape),
        "A_source": a_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HYDICE anomaly dataset into DPMN urban format without label leakage.")
    parser.add_argument("--src-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HYDICE-urban.mat"))
    parser.add_argument("--dst-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "urban2.mat"))
    parser.add_argument("--target-clusters", type=int, default=3, help="Target count used when estimating A if raw A is absent.")
    parser.add_argument("--background-clusters", type=int, default=5, help="Background count used when estimating A if raw A is absent.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for KMeans fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_hydice_mat_file(
        src_path=args.src_path,
        dst_path=args.dst_path,
        target_clusters=args.target_clusters,
        background_clusters=args.background_clusters,
        random_state=args.random_state,
    )
    print(f"Saved: {result['dst_path']}")
    print(f"image shape: {result['image_shape']}")
    print(f"mask shape: {result['mask_shape']}")
    print(f"A shape: {result['A_shape']}")
    print(f"A source: {result['A_source']}")


if __name__ == "__main__":
    main()
