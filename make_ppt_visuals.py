from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import scipy.io
from scipy import ndimage

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_mat_file(file_path: str) -> Dict[str, np.ndarray]:
    if h5py is not None:
        try:
            with h5py.File(file_path, "r") as mat:
                return {key: np.array(mat[key]) for key in mat.keys()}
        except Exception:
            pass
    mat = scipy.io.loadmat(file_path)
    return {key: value for key, value in mat.items() if not key.startswith("__")}


def normalize_numpy_minmax(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array_min = float(array.min())
    array_max = float(array.max())
    return (array - array_min) / (array_max - array_min + eps)


def build_display_image_from_hsi(image_np: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image_np = np.asarray(image_np, dtype=np.float32)
    if image_np.ndim != 3:
        raise ValueError(f"Expected 3D image cube, got {image_np.shape}")

    bands = image_np.shape[0]
    if bands >= 3:
        band_indices = np.linspace(0, bands - 1, 3).astype(int)
        display_img = np.transpose(image_np[band_indices], (1, 2, 0))
    else:
        single_band = image_np.mean(axis=0)
        display_img = np.stack([single_band, single_band, single_band], axis=-1)

    lower = float(np.percentile(display_img, 2.0))
    upper = float(np.percentile(display_img, 98.0))
    display_img = np.clip(display_img, lower, upper)
    display_min = float(display_img.min())
    display_max = float(display_img.max())
    return (display_img - display_min) / (display_max - display_min + eps)


def dilate_binary_mask(mask_np: np.ndarray, iterations: int = 2) -> np.ndarray:
    mask_np = np.asarray(mask_np, dtype=np.float32)
    binary = mask_np > 0.0
    dilated = ndimage.binary_dilation(binary, iterations=iterations)
    return dilated.astype(np.float32)


def find_focus_bbox(mask_np: np.ndarray, residual_np: np.ndarray, padding: int = 12) -> tuple[int, int, int, int]:
    mask_binary = np.asarray(mask_np) > 0
    if mask_binary.any():
        ys, xs = np.where(mask_binary)
    else:
        peak_index = int(np.argmax(residual_np))
        height, width = residual_np.shape
        y = peak_index // width
        x = peak_index % width
        ys = np.array([y])
        xs = np.array([x])

    y0 = max(0, int(ys.min()) - padding)
    y1 = min(residual_np.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(residual_np.shape[1], int(xs.max()) + padding + 1)
    return y0, y1, x0, x1


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "inferno") -> np.ndarray:
    if plt is None:
        return image_rgb
    color_map = plt.get_cmap(cmap)
    colored_heatmap = color_map(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * image_rgb + alpha * colored_heatmap, 0.0, 1.0)


def binarize_by_topk(score_map: np.ndarray, positive_count: int) -> np.ndarray:
    score_map = np.asarray(score_map, dtype=np.float32)
    flat = score_map.reshape(-1)
    positive_count = int(max(0, min(positive_count, flat.size)))
    binary = np.zeros_like(flat, dtype=np.uint8)
    if positive_count > 0:
        topk_indices = np.argpartition(flat, -positive_count)[-positive_count:]
        binary[topk_indices] = 1
    return binary.reshape(score_map.shape)


def compute_localization_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_mask, dtype=np.uint8) > 0
    gt = np.asarray(gt_mask, dtype=np.uint8) > 0

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
    }


def build_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_mask, dtype=np.uint8) > 0
    gt = np.asarray(gt_mask, dtype=np.uint8) > 0
    error_map = np.zeros(gt.shape + (3,), dtype=np.float32)
    error_map[np.logical_and(pred, gt)] = np.array([0.2, 0.8, 0.2], dtype=np.float32)
    error_map[np.logical_and(pred, ~gt)] = np.array([0.9, 0.2, 0.2], dtype=np.float32)
    error_map[np.logical_and(~pred, gt)] = np.array([0.2, 0.4, 0.95], dtype=np.float32)
    return error_map


def resolve_dataset_file(dataset_dir: str, sample_id: str, dataset_prefix: str) -> str:
    candidates = [
        os.path.join(dataset_dir, f"{dataset_prefix}{sample_id}.mat"),
        os.path.join(dataset_dir, f"{dataset_prefix}_{sample_id}.mat"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Dataset file not found for sample_id={sample_id} under {dataset_dir}")


def resolve_detection_file(detection_dir: str, sample_id: str) -> str:
    candidate = os.path.join(detection_dir, f"urban_detection_{sample_id}.mat")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Detection file not found: {candidate}")
    return candidate


def load_sample_assets(
    dataset_dir: str,
    detection_dir: str,
    sample_id: str,
    dataset_prefix: str,
) -> Dict[str, np.ndarray]:
    dataset_path = resolve_dataset_file(dataset_dir, sample_id, dataset_prefix)
    detection_path = resolve_detection_file(detection_dir, sample_id)

    dataset_mat = load_mat_file(dataset_path)
    detection_mat = load_mat_file(detection_path)

    image_np = np.array(dataset_mat["image"]).transpose(0, 2, 1)
    mask_np = np.array(dataset_mat["mask"]).transpose(1, 0)
    residual_np = np.asarray(detection_mat["detection"], dtype=np.float32)
    return {
        "image": image_np,
        "mask": mask_np,
        "residual": residual_np,
    }


def merge_results(
    baseline_rows: List[Dict[str, str]],
    improved_rows: List[Dict[str, str]],
    exclude_samples: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    baseline_map = {row["sample_id"]: row for row in baseline_rows}
    improved_map = {row["sample_id"]: row for row in improved_rows}
    exclude_set = set(exclude_samples or [])
    sample_ids = [
        sample_id
        for sample_id in baseline_map.keys()
        if sample_id in improved_map and sample_id not in exclude_set
    ]

    merged = []
    for sample_id in sample_ids:
        baseline_row = baseline_map[sample_id]
        improved_row = improved_map[sample_id]
        baseline_auc = float(baseline_row["roc_auc"])
        improved_auc = float(improved_row["roc_auc"])
        merged.append(
            {
                "sample_id": sample_id,
                "baseline_auc": baseline_auc,
                "improved_auc": improved_auc,
                "delta_auc": improved_auc - baseline_auc,
                "baseline_iter": float(baseline_row["stop_iteration"]),
                "improved_iter": float(improved_row["stop_iteration"]),
                "baseline_time": float(baseline_row["elapsed_seconds"]),
                "improved_time": float(improved_row["elapsed_seconds"]),
            }
        )
    return merged


def compute_localization_rows(
    merged_rows: List[Dict[str, float]],
    baseline_dataset_dir: str,
    baseline_detection_dir: str,
    improved_dataset_dir: str,
    improved_detection_dir: str,
    dataset_prefix: str,
) -> List[Dict[str, float]]:
    localization_rows: List[Dict[str, float]] = []
    for row in merged_rows:
        sample_id = row["sample_id"]
        baseline_assets = load_sample_assets(baseline_dataset_dir, baseline_detection_dir, sample_id, dataset_prefix)
        improved_assets = load_sample_assets(improved_dataset_dir, improved_detection_dir, sample_id, dataset_prefix)

        gt_mask = np.asarray(improved_assets["mask"], dtype=np.uint8) > 0
        positive_count = int(gt_mask.sum())
        baseline_score = normalize_numpy_minmax(baseline_assets["residual"])
        improved_score = normalize_numpy_minmax(improved_assets["residual"])
        baseline_binary = binarize_by_topk(baseline_score, positive_count)
        improved_binary = binarize_by_topk(improved_score, positive_count)
        baseline_metrics = compute_localization_metrics(baseline_binary, gt_mask)
        improved_metrics = compute_localization_metrics(improved_binary, gt_mask)

        localization_rows.append(
            {
                "sample_id": sample_id,
                "positive_count": float(positive_count),
                "baseline_iou": baseline_metrics["iou"],
                "improved_iou": improved_metrics["iou"],
                "delta_iou": improved_metrics["iou"] - baseline_metrics["iou"],
                "baseline_dice": baseline_metrics["dice"],
                "improved_dice": improved_metrics["dice"],
                "delta_dice": improved_metrics["dice"] - baseline_metrics["dice"],
                "baseline_precision": baseline_metrics["precision"],
                "improved_precision": improved_metrics["precision"],
                "delta_precision": improved_metrics["precision"] - baseline_metrics["precision"],
                "baseline_recall": baseline_metrics["recall"],
                "improved_recall": improved_metrics["recall"],
                "delta_recall": improved_metrics["recall"] - baseline_metrics["recall"],
            }
        )
    return localization_rows


def save_comparison_csv(output_dir: str, merged_rows: List[Dict[str, float]]) -> str:
    csv_path = os.path.join(output_dir, "baseline_vs_improved.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "baseline_auc",
                "improved_auc",
                "delta_auc",
                "baseline_iter",
                "improved_iter",
                "baseline_time",
                "improved_time",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)
    return csv_path


def save_case_manifest(output_dir: str, merged_rows: List[Dict[str, float]]) -> str:
    manifest_path = os.path.join(output_dir, "case_manifest.csv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "baseline_auc",
                "improved_auc",
                "delta_auc",
                "case_figure",
                "overlay_figure",
                "zoom_figure",
                "binary_figure",
                "error_figure",
            ],
        )
        writer.writeheader()
        for row in merged_rows:
            sample_id = row["sample_id"]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "baseline_auc": row["baseline_auc"],
                    "improved_auc": row["improved_auc"],
                    "delta_auc": row["delta_auc"],
                    "case_figure": f"sample_{sample_id}_case.png",
                    "overlay_figure": f"sample_{sample_id}_overlay.png",
                    "zoom_figure": f"sample_{sample_id}_zoom.png",
                    "binary_figure": f"sample_{sample_id}_binary.png",
                    "error_figure": f"sample_{sample_id}_error.png",
                }
            )
    return manifest_path


def save_localization_csv(output_dir: str, localization_rows: List[Dict[str, float]]) -> str:
    csv_path = os.path.join(output_dir, "metrics_localization.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "positive_count",
                "baseline_iou",
                "improved_iou",
                "delta_iou",
                "baseline_dice",
                "improved_dice",
                "delta_dice",
                "baseline_precision",
                "improved_precision",
                "delta_precision",
                "baseline_recall",
                "improved_recall",
                "delta_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(localization_rows)
    return csv_path


def save_summary_plots(output_dir: str, merged_rows: List[Dict[str, float]]) -> List[str]:
    if plt is None:
        return []

    sample_ids = [row["sample_id"] for row in merged_rows]
    baseline_auc = [row["baseline_auc"] for row in merged_rows]
    improved_auc = [row["improved_auc"] for row in merged_rows]
    delta_auc = [row["delta_auc"] for row in merged_rows]
    baseline_time = [row["baseline_time"] for row in merged_rows]
    improved_time = [row["improved_time"] for row in merged_rows]

    paths: List[str] = []
    x = np.arange(len(sample_ids))
    width = 0.38

    figure = plt.figure(figsize=(16, 6))
    plt.bar(x - width / 2, baseline_auc, width=width, label="Baseline")
    plt.bar(x + width / 2, improved_auc, width=width, label="Improved")
    plt.xticks(x, sample_ids, rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.title("Per-Sample AUC Comparison")
    plt.legend()
    plt.tight_layout()
    auc_path = os.path.join(output_dir, "auc_comparison.png")
    figure.savefig(auc_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    paths.append(auc_path)

    figure = plt.figure(figsize=(14, 6))
    colors = ["#2E8B57" if value >= 0 else "#C0392B" for value in delta_auc]
    plt.bar(x, delta_auc, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(x, sample_ids, rotation=45, ha="right")
    plt.ylabel("Improved AUC - Baseline AUC")
    plt.title("AUC Gain per Sample")
    plt.tight_layout()
    delta_path = os.path.join(output_dir, "auc_delta.png")
    figure.savefig(delta_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    paths.append(delta_path)

    figure = plt.figure(figsize=(8, 5))
    mean_baseline_auc = float(np.mean(baseline_auc))
    mean_improved_auc = float(np.mean(improved_auc))
    plt.bar(["Baseline", "Improved"], [mean_baseline_auc, mean_improved_auc], color=["#7F8C8D", "#2980B9"])
    plt.ylabel("Mean AUC")
    plt.title("Mean AUC Comparison")
    plt.tight_layout()
    mean_auc_path = os.path.join(output_dir, "mean_auc_comparison.png")
    figure.savefig(mean_auc_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    paths.append(mean_auc_path)

    figure = plt.figure(figsize=(8, 5))
    plt.bar(
        ["Baseline", "Improved"],
        [float(np.mean(baseline_time)), float(np.mean(improved_time))],
        color=["#95A5A6", "#8E44AD"],
    )
    plt.ylabel("Mean Elapsed Seconds")
    plt.title("Mean Runtime Comparison")
    plt.tight_layout()
    mean_time_path = os.path.join(output_dir, "mean_runtime_comparison.png")
    figure.savefig(mean_time_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    paths.append(mean_time_path)
    return paths


def save_localization_summary_plots(output_dir: str, localization_rows: List[Dict[str, float]]) -> List[str]:
    if plt is None:
        return []

    sample_ids = [row["sample_id"] for row in localization_rows]
    x = np.arange(len(sample_ids))
    width = 0.38
    paths: List[str] = []

    metric_specs = [
        ("iou", "IoU Comparison", "iou_comparison.png"),
        ("dice", "Dice Comparison", "dice_comparison.png"),
        ("precision", "Precision Comparison", "precision_comparison.png"),
        ("recall", "Recall Comparison", "recall_comparison.png"),
    ]

    for metric_name, title, filename in metric_specs:
        baseline_values = [row[f"baseline_{metric_name}"] for row in localization_rows]
        improved_values = [row[f"improved_{metric_name}"] for row in localization_rows]
        figure = plt.figure(figsize=(16, 6))
        plt.bar(x - width / 2, baseline_values, width=width, label="Baseline")
        plt.bar(x + width / 2, improved_values, width=width, label="Improved")
        plt.xticks(x, sample_ids, rotation=45, ha="right")
        plt.ylabel(metric_name.upper())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths.append(output_path)

    return paths


def save_case_visualization(
    output_dir: str,
    row: Dict[str, float],
    baseline_assets: Dict[str, np.ndarray],
    improved_assets: Dict[str, np.ndarray],
) -> str:
    if plt is None:
        return os.path.join(output_dir, f"sample_{row['sample_id']}_case.png")

    sample_id = row["sample_id"]
    original_img = build_display_image_from_hsi(improved_assets["image"])
    gt_mask = np.asarray(improved_assets["mask"], dtype=np.float32)
    gt_mask_display = dilate_binary_mask(gt_mask, iterations=2)
    baseline_residual = normalize_numpy_minmax(baseline_assets["residual"])
    improved_residual = normalize_numpy_minmax(improved_assets["residual"])

    figure, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(original_img, interpolation="nearest")
    axes[0].set_title("Original")
    axes[1].imshow(gt_mask_display, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(baseline_residual, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Baseline Detection\nAUC={row['baseline_auc']:.4f}")
    axes[3].imshow(improved_residual, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[3].set_title(f"Improved Detection\nAUC={row['improved_auc']:.4f}")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(f"Sample {sample_id} | Delta AUC={row['delta_auc']:+.4f}", fontsize=16)
    figure.tight_layout()
    output_path = os.path.join(output_dir, f"sample_{sample_id}_case.png")
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_overlay_visualization(
    output_dir: str,
    row: Dict[str, float],
    baseline_assets: Dict[str, np.ndarray],
    improved_assets: Dict[str, np.ndarray],
) -> str:
    if plt is None:
        return os.path.join(output_dir, f"sample_{row['sample_id']}_overlay.png")

    sample_id = row["sample_id"]
    original_img = build_display_image_from_hsi(improved_assets["image"])
    gt_mask = dilate_binary_mask(improved_assets["mask"], iterations=2)
    baseline_residual = normalize_numpy_minmax(baseline_assets["residual"])
    improved_residual = normalize_numpy_minmax(improved_assets["residual"])
    baseline_overlay = overlay_heatmap_on_image(original_img, baseline_residual, cmap="inferno")
    improved_overlay = overlay_heatmap_on_image(original_img, improved_residual, cmap="inferno")

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_img, interpolation="nearest")
    axes[0].imshow(gt_mask, cmap="Greens", alpha=0.35, interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original + GT")
    axes[1].imshow(baseline_overlay, interpolation="nearest")
    axes[1].set_title(f"Baseline Overlay\nAUC={row['baseline_auc']:.4f}")
    axes[2].imshow(improved_overlay, interpolation="nearest")
    axes[2].set_title(f"Improved Overlay\nAUC={row['improved_auc']:.4f}")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(f"Sample {sample_id} Overlay View", fontsize=16)
    figure.tight_layout()
    output_path = os.path.join(output_dir, f"sample_{sample_id}_overlay.png")
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_zoom_visualization(
    output_dir: str,
    row: Dict[str, float],
    baseline_assets: Dict[str, np.ndarray],
    improved_assets: Dict[str, np.ndarray],
) -> str:
    if plt is None:
        return os.path.join(output_dir, f"sample_{row['sample_id']}_zoom.png")

    sample_id = row["sample_id"]
    original_img = build_display_image_from_hsi(improved_assets["image"])
    gt_mask = dilate_binary_mask(improved_assets["mask"], iterations=2)
    baseline_residual = normalize_numpy_minmax(baseline_assets["residual"])
    improved_residual = normalize_numpy_minmax(improved_assets["residual"])
    y0, y1, x0, x1 = find_focus_bbox(gt_mask, improved_residual)

    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_img[y0:y1, x0:x1], interpolation="nearest")
    axes[0].set_title("Original Zoom")
    axes[1].imshow(gt_mask[y0:y1, x0:x1], cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("GT Zoom")
    axes[2].imshow(baseline_residual[y0:y1, x0:x1], cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[2].set_title("Baseline Zoom")
    axes[3].imshow(improved_residual[y0:y1, x0:x1], cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[3].set_title("Improved Zoom")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(f"Sample {sample_id} Local Zoom", fontsize=16)
    figure.tight_layout()
    output_path = os.path.join(output_dir, f"sample_{sample_id}_zoom.png")
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_binary_visualization(
    output_dir: str,
    row: Dict[str, float],
    baseline_assets: Dict[str, np.ndarray],
    improved_assets: Dict[str, np.ndarray],
) -> str:
    if plt is None:
        return os.path.join(output_dir, f"sample_{row['sample_id']}_binary.png")

    sample_id = row["sample_id"]
    gt_mask = np.asarray(improved_assets["mask"], dtype=np.uint8) > 0
    positive_count = int(gt_mask.sum())
    baseline_binary = binarize_by_topk(normalize_numpy_minmax(baseline_assets["residual"]), positive_count)
    improved_binary = binarize_by_topk(normalize_numpy_minmax(improved_assets["residual"]), positive_count)
    baseline_metrics = compute_localization_metrics(baseline_binary, gt_mask)
    improved_metrics = compute_localization_metrics(improved_binary, gt_mask)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dilate_binary_mask(gt_mask, iterations=2), cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(dilate_binary_mask(baseline_binary, iterations=2), cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Baseline Binary\nIoU={baseline_metrics['iou']:.4f} Dice={baseline_metrics['dice']:.4f}")
    axes[2].imshow(dilate_binary_mask(improved_binary, iterations=2), cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Improved Binary\nIoU={improved_metrics['iou']:.4f} Dice={improved_metrics['dice']:.4f}")
    for axis in axes:
        axis.axis("off")

    figure.suptitle(f"Sample {sample_id} Binary Localization", fontsize=16)
    figure.tight_layout()
    output_path = os.path.join(output_dir, f"sample_{sample_id}_binary.png")
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_error_visualization(
    output_dir: str,
    row: Dict[str, float],
    baseline_assets: Dict[str, np.ndarray],
    improved_assets: Dict[str, np.ndarray],
) -> str:
    if plt is None:
        return os.path.join(output_dir, f"sample_{row['sample_id']}_error.png")

    sample_id = row["sample_id"]
    gt_mask = np.asarray(improved_assets["mask"], dtype=np.uint8) > 0
    positive_count = int(gt_mask.sum())
    baseline_binary = binarize_by_topk(normalize_numpy_minmax(baseline_assets["residual"]), positive_count)
    improved_binary = binarize_by_topk(normalize_numpy_minmax(improved_assets["residual"]), positive_count)
    baseline_metrics = compute_localization_metrics(baseline_binary, gt_mask)
    improved_metrics = compute_localization_metrics(improved_binary, gt_mask)
    baseline_error = build_error_map(baseline_binary, gt_mask)
    improved_error = build_error_map(improved_binary, gt_mask)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(baseline_error, interpolation="nearest")
    axes[0].set_title(
        f"Baseline Error\nP={baseline_metrics['precision']:.4f} R={baseline_metrics['recall']:.4f}"
    )
    axes[1].imshow(improved_error, interpolation="nearest")
    axes[1].set_title(
        f"Improved Error\nP={improved_metrics['precision']:.4f} R={improved_metrics['recall']:.4f}"
    )
    for axis in axes:
        axis.axis("off")

    figure.suptitle(f"Sample {sample_id} Error Map | TP=Green FP=Red FN=Blue", fontsize=16)
    figure.tight_layout()
    output_path = os.path.join(output_dir, f"sample_{sample_id}_error.png")
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_selected_case_visualizations(
    output_dir: str,
    merged_rows: List[Dict[str, float]],
    baseline_dataset_dir: str,
    baseline_detection_dir: str,
    improved_dataset_dir: str,
    improved_detection_dir: str,
    dataset_prefix: str,
    selected_samples: Optional[List[str]],
) -> List[str]:
    row_map = {row["sample_id"]: row for row in merged_rows}

    if selected_samples:
        sample_ids = selected_samples
    else:
        sample_ids = [row["sample_id"] for row in merged_rows]

    output_paths = []
    for sample_id in sample_ids:
        if sample_id not in row_map:
            continue
        baseline_assets = load_sample_assets(baseline_dataset_dir, baseline_detection_dir, sample_id, dataset_prefix)
        improved_assets = load_sample_assets(improved_dataset_dir, improved_detection_dir, sample_id, dataset_prefix)
        row = row_map[sample_id]
        output_paths.append(save_case_visualization(output_dir, row, baseline_assets, improved_assets))
        output_paths.append(save_overlay_visualization(output_dir, row, baseline_assets, improved_assets))
        output_paths.append(save_zoom_visualization(output_dir, row, baseline_assets, improved_assets))
        output_paths.append(save_binary_visualization(output_dir, row, baseline_assets, improved_assets))
        output_paths.append(save_error_visualization(output_dir, row, baseline_assets, improved_assets))
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PPT-ready comparison visuals for baseline and improved runs.")
    parser.add_argument("--baseline-csv", required=True, help="Path to baseline batch_results CSV.")
    parser.add_argument("--improved-csv", required=True, help="Path to improved batch_results CSV.")
    parser.add_argument("--baseline-dataset-dir", required=True, help="Dataset directory used by baseline.")
    parser.add_argument("--improved-dataset-dir", required=True, help="Dataset directory used by improved run.")
    parser.add_argument("--baseline-detection-dir", required=True, help="Detection .mat directory for baseline.")
    parser.add_argument("--improved-detection-dir", required=True, help="Detection .mat directory for improved run.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated comparison figures.")
    parser.add_argument("--dataset-prefix", default="urban", help="Dataset prefix.")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample ids for case comparison figures. If omitted, export all samples.",
    )
    parser.add_argument(
        "--exclude-samples",
        nargs="*",
        default=["abu_urban_5"],
        help="Sample ids to remove from summary charts and case figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    baseline_rows = load_csv_rows(args.baseline_csv)
    improved_rows = load_csv_rows(args.improved_csv)
    merged_rows = merge_results(baseline_rows, improved_rows, exclude_samples=args.exclude_samples)
    merged_rows = sorted(merged_rows, key=lambda item: item["sample_id"])
    localization_rows = compute_localization_rows(
        merged_rows,
        args.baseline_dataset_dir,
        args.baseline_detection_dir,
        args.improved_dataset_dir,
        args.improved_detection_dir,
        args.dataset_prefix,
    )

    comparison_csv = save_comparison_csv(args.output_dir, merged_rows)
    manifest_csv = save_case_manifest(args.output_dir, merged_rows)
    localization_csv = save_localization_csv(args.output_dir, localization_rows)
    summary_plots = save_summary_plots(args.output_dir, merged_rows)
    localization_plots = save_localization_summary_plots(args.output_dir, localization_rows)
    case_plots = save_selected_case_visualizations(
        args.output_dir,
        merged_rows,
        args.baseline_dataset_dir,
        args.baseline_detection_dir,
        args.improved_dataset_dir,
        args.improved_detection_dir,
        args.dataset_prefix,
        args.samples,
    )

    print(f"Saved merged CSV: {comparison_csv}")
    print(f"Saved manifest CSV: {manifest_csv}")
    print(f"Saved localization CSV: {localization_csv}")
    for path in summary_plots + localization_plots + case_plots:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
