import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io

from spectral_library_utils import normalize_numpy_minmax, resolve_spectral_library


def convert_abu_mat_file(
    src_path,
    dst_path,
    target_clusters=3,
    background_clusters=5,
    normalize_data=True,
    random_state=0,
):
    mat = scipy.io.loadmat(src_path)
    data = np.asarray(mat["data"], dtype=np.float32)
    anomaly_map = np.asarray(mat["map"])

    if data.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, got shape {data.shape}")
    if anomaly_map.shape != data.shape[:2]:
        raise ValueError(f"Mask shape {anomaly_map.shape} does not match spatial shape {data.shape[:2]}")

    if normalize_data:
        data = normalize_numpy_minmax(data)

    A, a_source = resolve_spectral_library(
        mat=mat,
        data=data,
        target_clusters=target_clusters,
        background_clusters=background_clusters,
        random_state=random_state,
    )

    # Match the layout expected by train.py:
    # image: (bands, width, height)
    # mask: (width, height)
    # A: (num_endmembers, bands)
    image = np.transpose(data, (2, 1, 0)).astype(np.float32)
    mask = np.transpose(anomaly_map, (1, 0)).astype(np.uint8)

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


def sanitize_source_name(file_name):
    stem = Path(file_name).stem
    sanitized = []
    for char in stem:
        if char.isalnum():
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_")


def convert_abu_directory(
    source_dir,
    dataset_dir,
    file_pattern="abu-*.mat",
    output_prefix="urban_",
    target_clusters=3,
    background_clusters=5,
    normalize_data=True,
    overwrite=False,
    random_state=0,
):
    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    results = []
    for src_path in sorted(source_dir.glob(file_pattern)):
        dst_name = f"{output_prefix}{sanitize_source_name(src_path.name)}.mat"
        dst_path = dataset_dir / dst_name
        if dst_path.exists() and not overwrite:
            results.append(
                {
                    "src_path": str(src_path),
                    "dst_path": str(dst_path),
                    "skipped": True,
                }
            )
            continue

        result = convert_abu_mat_file(
            src_path=src_path,
            dst_path=dst_path,
            target_clusters=target_clusters,
            background_clusters=background_clusters,
            normalize_data=normalize_data,
            random_state=random_state,
        )
        result["skipped"] = False
        results.append(result)

    if not results:
        raise FileNotFoundError(f"No files matched pattern '{file_pattern}' in {source_dir}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ABU hyperspectral anomaly .mat files into DPMN urban format without label leakage.")
    parser.add_argument("--source-dir", required=True, help="Directory containing ABU .mat files with data/map keys.")
    parser.add_argument("--dataset-dir", required=True, help="Target dataset directory.")
    parser.add_argument("--file-pattern", default="abu-*.mat", help="Glob pattern for source files.")
    parser.add_argument("--output-prefix", default="urban_", help="Prefix for converted dataset files.")
    parser.add_argument("--target-clusters", type=int, default=3, help="Used with background_clusters as the fallback total-cluster estimate when raw A is absent.")
    parser.add_argument("--background-clusters", type=int, default=5, help="Used with target_clusters as the fallback total-cluster estimate when raw A is absent.")
    parser.add_argument("--disable-normalize", action="store_true", help="Do not min-max normalize source cubes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already converted dataset files.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for KMeans.")
    return parser.parse_args()


def main():
    args = parse_args()
    results = convert_abu_directory(
        source_dir=args.source_dir,
        dataset_dir=args.dataset_dir,
        file_pattern=args.file_pattern,
        output_prefix=args.output_prefix,
        target_clusters=args.target_clusters,
        background_clusters=args.background_clusters,
        normalize_data=not args.disable_normalize,
        overwrite=args.overwrite,
        random_state=args.random_state,
    )
    converted_count = sum(0 if item.get("skipped") else 1 for item in results)
    skipped_count = sum(1 if item.get("skipped") else 0 for item in results)
    print(f"Converted {converted_count} file(s), skipped {skipped_count} file(s).")
    for item in results:
        if item.get("skipped"):
            print(f"SKIPPED {item['src_path']} -> {item['dst_path']}")
        else:
            print(
                f"CREATED {item['src_path']} -> {item['dst_path']} "
                f"image={item['image_shape']} mask={item['mask_shape']} "
                f"A={item['A_shape']} source={item['A_source']}"
            )


if __name__ == "__main__":
    main()
