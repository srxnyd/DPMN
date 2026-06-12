from __future__ import print_function

import argparse
import csv
import glob
import os
import re
import time

import numpy as np
import scipy.io
import torch
import torch.optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from scipy import ndimage

from convert_abu_to_urban import convert_abu_directory
from model.AGM import AGM
from utils.inpainting_utils import *
from utils.traditonal import total_variation

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


ABLATION_MODES = {
    "baseline": {"adaptive_mask": False, "sam_loss": False},
    "mask_only": {"adaptive_mask": True, "sam_loss": False},
    "sam_only": {"adaptive_mask": False, "sam_loss": True},
    "mask_sam": {"adaptive_mask": True, "sam_loss": True},
    "prior_blindspot": {"adaptive_mask": True, "sam_loss": False, "prior_mode": "consensus", "blindspot": True},
    "sp_imp_dpmn": {
        "adaptive_mask": True,
        "sam_loss": True,
        "prior_mode": "sp_imp",
        "superpixel_perturb": True,
        "online_background_mining": True,
        "blindspot": False,
        "num_iter": 800,
        "sp_target_weight": 0.6,
        "residual_weight": 0.8,
        "contrast_weight": 0.15,
        "uncertainty_weight": 0.05,
        "adaptive_score_fusion": True,
    },
    "full_innovation": {
        "adaptive_mask": True,
        "sam_loss": True,
        "prior_mode": "consensus",
        "blindspot": True,
        "low_rank_sparse": True,
    },
}


def format_background_for_savemat(background_img):
    background_img = np.asarray(background_img)
    if background_img.ndim == 4:
        background_img = np.squeeze(background_img, axis=0)
    if background_img.ndim == 3:
        return np.transpose(background_img, (1, 2, 0))
    if background_img.ndim == 2:
        return background_img
    raise ValueError(f"Unexpected background image shape: {background_img.shape}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_score_components(output_dir, sample_id, score_maps, weights):
    ensure_dir(output_dir)
    mat_path = os.path.join(output_dir, f"urban_scores_{sample_id}.mat")
    save_data = {
        "residual_score": np.asarray(score_maps["residual_score"], dtype=np.float32),
        "contrast_score": np.asarray(score_maps["contrast_score"], dtype=np.float32),
        "uncertainty_score": np.asarray(score_maps["uncertainty_score"], dtype=np.float32),
        "fused_score": np.asarray(score_maps["fused_score"], dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
    }
    if "highfreq_score" in score_maps and score_maps["highfreq_score"] is not None:
        save_data["highfreq_score"] = np.asarray(score_maps["highfreq_score"], dtype=np.float32)
    if "highfreq_alpha" in score_maps and score_maps["highfreq_alpha"] is not None:
        save_data["highfreq_alpha"] = np.asarray(score_maps["highfreq_alpha"], dtype=np.float32)
    if "highfreq_alpha_map" in score_maps and score_maps["highfreq_alpha_map"] is not None:
        save_data["highfreq_alpha_map"] = np.asarray(score_maps["highfreq_alpha_map"], dtype=np.float32)
    scipy.io.savemat(mat_path, save_data)
    return mat_path


def save_mask_snapshots(output_dir, sample_id, snapshot_records):
    ensure_dir(output_dir)
    if not snapshot_records:
        return None

    snapshot_records = sorted(snapshot_records, key=lambda item: item[0])
    mat_path = os.path.join(output_dir, f"urban_mask_history_{sample_id}.mat")
    iterations = np.asarray([item[0] for item in snapshot_records], dtype=np.int32)
    mask_snapshots = np.stack([np.asarray(item[1], dtype=np.float32) for item in snapshot_records], axis=0)
    scipy.io.savemat(
        mat_path,
        {
            "iterations": iterations,
            "mask_snapshots": mask_snapshots,
        },
    )
    return mat_path


def save_training_visualizations(output_dir, sample_id, mode_name, history):
    ensure_dir(output_dir)

    csv_path = os.path.join(output_dir, f"{mode_name}_sample_{sample_id}_loss_history.csv")
    png_path = os.path.join(output_dir, f"{mode_name}_sample_{sample_id}_loss_curve.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration",
            "total_loss",
            "mse_loss",
            "tv_loss",
            "sam_loss",
            "sam_weight",
            "low_rank_loss",
            "sparse_loss",
            "blindspot_active",
        ])
        for idx, metrics in enumerate(history, start=1):
            writer.writerow([
                idx,
                metrics["total_loss"],
                metrics["mse_loss"],
                metrics["tv_loss"],
                metrics["sam_loss"],
                metrics.get("sam_weight", 0.0),
            ])

    if plt is not None:
        iterations = [idx for idx in range(1, len(history) + 1)]
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, [item["total_loss"] for item in history], label="total")
        plt.plot(iterations, [item["mse_loss"] for item in history], label="mse")
        plt.plot(iterations, [item["tv_loss"] for item in history], label="tv")
        if any(item["sam_loss"] != 0.0 for item in history):
            plt.plot(iterations, [item["sam_loss"] for item in history], label="sam")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curves - mode={mode_name}, sample={sample_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
    else:
        png_path = None

    return csv_path, png_path


def build_display_image_from_hsi(image_np, eps=1e-8):
    image_np = np.asarray(image_np, dtype=np.float32)
    if image_np.ndim != 3:
        raise ValueError(f"Expected hyperspectral cube with 3 dims, got shape {image_np.shape}")

    bands = image_np.shape[0]
    if bands >= 3:
        band_indices = np.linspace(0, bands - 1, 3).astype(int)
        display_img = np.transpose(image_np[band_indices], (1, 2, 0))
    else:
        single_band = image_np.mean(axis=0)
        display_img = np.stack([single_band, single_band, single_band], axis=-1)

    lower = np.percentile(display_img, 2, axis=(0, 1), keepdims=True)
    upper = np.percentile(display_img, 98, axis=(0, 1), keepdims=True)
    display_img = np.clip(display_img, lower, upper)
    return (display_img - lower) / (upper - lower + eps)


def dilate_binary_mask(mask_np, iterations=2):
    mask_np = np.asarray(mask_np, dtype=np.float32)
    dilated = ndimage.binary_dilation(mask_np > 0.0, iterations=iterations)
    return dilated.astype(np.float32)


def find_focus_bbox(mask_np, residual_np, padding=12):
    mask_binary = np.asarray(mask_np) > 0
    if mask_binary.any():
        ys, xs = np.where(mask_binary)
    else:
        peak_index = int(np.argmax(residual_np))
        height, width = residual_np.shape
        ys = np.array([peak_index // width])
        xs = np.array([peak_index % width])

    y0 = max(0, int(ys.min()) - padding)
    y1 = min(residual_np.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(residual_np.shape[1], int(xs.max()) + padding + 1)
    return y0, y1, x0, x1


def overlay_heatmap_on_image(image_rgb, heatmap, alpha=0.45, cmap="inferno"):
    if plt is None:
        return image_rgb
    color_map = plt.get_cmap(cmap)
    colored_heatmap = color_map(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * image_rgb + alpha * colored_heatmap, 0.0, 1.0)


def save_detection_visualization(output_dir, sample_id, mode_name, image_np, residual_np, label_np=None):
    ensure_dir(output_dir)
    compare_path = os.path.join(output_dir, f"{mode_name}_sample_{sample_id}_comparison.png")
    residual_path = os.path.join(output_dir, f"{mode_name}_sample_{sample_id}_residual.png")
    overlay_path = os.path.join(output_dir, f"{mode_name}_sample_{sample_id}_overlay.png")
    zoom_path = os.path.join(output_dir, f"{mode_name}_sample_{sample_id}_zoom.png")

    if plt is None:
        return compare_path, residual_path, overlay_path, zoom_path

    display_img = build_display_image_from_hsi(image_np)
    residual_img = normalize_numpy_minmax(residual_np)
    label_img = None if label_np is None else np.asarray(label_np, dtype=np.float32)
    label_display = None if label_img is None else dilate_binary_mask(label_img, iterations=2)

    if label_display is not None:
        figure, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(display_img, interpolation="nearest")
        axes[0].set_title("Original")
        axes[1].imshow(residual_img, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
        axes[1].set_title("Detection")
        axes[2].imshow(label_display, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
        axes[2].set_title("Ground Truth")
        target_axes = axes
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(display_img, interpolation="nearest")
        axes[0].set_title("Original")
        axes[1].imshow(residual_img, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
        axes[1].set_title("Detection")
        target_axes = axes

    for axis in np.atleast_1d(target_axes):
        axis.axis("off")

    figure.suptitle(f"Sample {sample_id} - {mode_name}")
    figure.tight_layout()
    figure.savefig(compare_path, dpi=350, bbox_inches="tight")
    plt.close(figure)

    residual_figure = plt.figure(figsize=(6, 6))
    plt.imshow(residual_img, cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(residual_path, dpi=350, bbox_inches="tight")
    plt.close(residual_figure)

    overlay_figure, overlay_axis = plt.subplots(1, 1, figsize=(6, 6))
    overlay_axis.imshow(overlay_heatmap_on_image(display_img, residual_img, cmap="inferno"), interpolation="nearest")
    if label_img is not None:
        overlay_axis.contour(label_display, levels=[0.5], colors="white", linewidths=1.2)
    overlay_axis.set_title("Detection Overlay")
    overlay_axis.axis("off")
    overlay_figure.suptitle(f"Sample {sample_id} - {mode_name}")
    overlay_figure.tight_layout()
    overlay_figure.savefig(overlay_path, dpi=350, bbox_inches="tight")
    plt.close(overlay_figure)

    y0, y1, x0, x1 = find_focus_bbox(label_img if label_img is not None else residual_img, residual_img)
    zoom_figure, zoom_axes = plt.subplots(1, 3 if label_img is not None else 2, figsize=(16, 5))
    zoom_axes = np.atleast_1d(zoom_axes)
    zoom_axes[0].imshow(display_img[y0:y1, x0:x1], interpolation="nearest")
    zoom_axes[0].set_title("Original Zoom")
    zoom_axes[1].imshow(residual_img[y0:y1, x0:x1], cmap="inferno", interpolation="nearest", vmin=0.0, vmax=1.0)
    zoom_axes[1].set_title("Detection Zoom")
    if label_img is not None:
        zoom_axes[2].imshow(label_display[y0:y1, x0:x1], cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
        zoom_axes[2].set_title("GT Zoom")
    for axis in zoom_axes:
        axis.axis("off")
    zoom_figure.suptitle(f"Sample {sample_id} - {mode_name} Zoom")
    zoom_figure.tight_layout()
    zoom_figure.savefig(zoom_path, dpi=350, bbox_inches="tight")
    plt.close(zoom_figure)
    return compare_path, residual_path, overlay_path, zoom_path


def min_max_normalize(tensor, eps=1e-8):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min + eps)


def normalize_numpy_minmax(array, eps=1e-8):
    array = np.asarray(array, dtype=np.float32)
    array_min = float(array.min())
    array_max = float(array.max())
    return (array - array_min) / (array_max - array_min + eps)


def normalize_endmember_columns(array, eps=1e-8):
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return array / norms


def make_grid_superpixel_labels(height, width, desired_segments):
    desired_segments = max(1, int(desired_segments))
    aspect = float(width) / float(max(height, 1))
    grid_rows = max(1, int(round(np.sqrt(desired_segments / max(aspect, 1e-6)))))
    grid_cols = max(1, int(np.ceil(float(desired_segments) / float(grid_rows))))
    y_bins = np.linspace(0, height, grid_rows + 1, dtype=np.int32)
    x_bins = np.linspace(0, width, grid_cols + 1, dtype=np.int32)
    labels = np.zeros((height, width), dtype=np.int32)
    label_id = 0
    for y_idx in range(grid_rows):
        for x_idx in range(grid_cols):
            labels[y_bins[y_idx]:y_bins[y_idx + 1], x_bins[x_idx]:x_bins[x_idx + 1]] = label_id
            label_id += 1
    return labels


def compute_superpixel_labels(image_np, desired_segments=256, compactness=0.08):
    _, height, width = image_np.shape
    try:
        from skimage.segmentation import slic

        display_img = build_display_image_from_hsi(image_np)
        labels = slic(
            display_img,
            n_segments=max(2, int(desired_segments)),
            compactness=float(compactness),
            sigma=1.0,
            start_label=0,
            channel_axis=-1,
        )
        return np.asarray(labels, dtype=np.int32)
    except Exception as exc:
        print(f"Superpixel SLIC unavailable ({exc}); using grid pooling fallback.")
        return make_grid_superpixel_labels(height, width, desired_segments)


def pool_hsi_by_superpixel(image_np, labels):
    image_np = np.asarray(image_np, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    bands, height, width = image_np.shape
    flat_labels = labels.reshape(-1)
    label_count = int(flat_labels.max()) + 1
    pooled = np.zeros((bands, height * width), dtype=np.float32)
    counts = np.bincount(flat_labels, minlength=label_count).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    for band_idx in range(bands):
        band_flat = image_np[band_idx].reshape(-1)
        sums = np.bincount(flat_labels, weights=band_flat, minlength=label_count).astype(np.float32)
        means = sums / counts
        pooled[band_idx] = means[flat_labels]
    return pooled.reshape(bands, height, width)


def smooth_mask_by_superpixel(mask_tensor, labels_tensor, eps=1e-8):
    if labels_tensor is None:
        return mask_tensor
    labels = labels_tensor.view(-1)
    mask_flat = mask_tensor.squeeze(0).squeeze(0).reshape(-1)
    label_count = int(labels.max().item()) + 1
    sums = torch.zeros(label_count, device=mask_tensor.device, dtype=mask_tensor.dtype)
    counts = torch.zeros(label_count, device=mask_tensor.device, dtype=mask_tensor.dtype)
    sums.scatter_add_(0, labels, mask_flat)
    counts.scatter_add_(0, labels, torch.ones_like(mask_flat))
    means = sums / counts.clamp_min(eps)
    return means[labels].view_as(mask_tensor).clamp(0.02, 1.0)


def print_sample_stats(sample_id, img_np, e_np, label_np, stage):
    print(
        f"sample {sample_id} [{stage}] image stats: "
        f"min={img_np.min():.6f}, max={img_np.max():.6f}, "
        f"mean={img_np.mean():.6f}, std={img_np.std():.6f}"
    )
    print(
        f"sample {sample_id} [{stage}] A stats: "
        f"min={e_np.min():.6f}, max={e_np.max():.6f}, "
        f"mean={e_np.mean():.6f}, std={e_np.std():.6f}"
    )
    print(
        f"sample {sample_id} [{stage}] mask stats: "
        f"sum={label_np.sum():.0f}, min={label_np.min():.6f}, max={label_np.max():.6f}"
    )


def compute_adaptive_mask(recon_img, target_img, warmup_progress, spatial_kernel=7, eps=1e-8):
    spectral_error = (recon_img - target_img).pow(2).sum(dim=1, keepdim=True)
    spatial_context = torch.nn.functional.avg_pool2d(
        spectral_error, kernel_size=spatial_kernel, stride=1, padding=spatial_kernel // 2
    )

    spectral_score = 1.0 - min_max_normalize(spectral_error, eps=eps)
    spatial_score = 1.0 - min_max_normalize(spatial_context, eps=eps)

    spectral_weight = 0.45 + 0.15 * warmup_progress
    spatial_weight = 1.0 - spectral_weight
    confidence_map = (
        spectral_weight * spectral_score
        + spatial_weight * spatial_score
    )

    flat_score = confidence_map.flatten()
    quantile = 0.50 + 0.25 * warmup_progress
    threshold = torch.quantile(flat_score, quantile)

    hard_mask = (confidence_map >= threshold).float()
    soft_mask = confidence_map.clamp(0.05, 1.0)
    refined_mask = (0.6 * soft_mask + 0.4 * hard_mask).clamp(0.05, 1.0)
    residual_map = min_max_normalize(spectral_error.squeeze(0).squeeze(0), eps=eps)
    return refined_mask, residual_map


def compute_online_background_mining_mask(
    recon_img,
    target_img,
    abundance_map,
    warmup_progress,
    spatial_kernel=7,
    superpixel_labels=None,
    eps=1e-8,
):
    residual_score = (recon_img - target_img).pow(2).sum(dim=1, keepdim=True)
    residual_score = min_max_normalize(residual_score, eps=eps)
    spatial_score = F.avg_pool2d(
        residual_score,
        kernel_size=spatial_kernel,
        stride=1,
        padding=spatial_kernel // 2,
    )
    spatial_score = min_max_normalize(spatial_score, eps=eps)
    contrast_score = compute_spectral_contrast_score(target_img, kernel_size=spatial_kernel, eps=eps)
    uncertainty_score = compute_abundance_uncertainty_score(abundance_map, eps=eps)

    anomaly_score = (
        0.45 * rank_normalize_score(residual_score, eps=eps)
        + 0.25 * rank_normalize_score(spatial_score, eps=eps)
        + 0.20 * rank_normalize_score(contrast_score, eps=eps)
        + 0.10 * rank_normalize_score(uncertainty_score, eps=eps)
    )
    anomaly_score = F.avg_pool2d(
        anomaly_score,
        kernel_size=spatial_kernel,
        stride=1,
        padding=spatial_kernel // 2,
    )
    anomaly_score = min_max_normalize(anomaly_score, eps=eps)

    background_confidence = (1.0 - anomaly_score).clamp(0.02, 1.0)
    background_confidence = smooth_mask_by_superpixel(background_confidence, superpixel_labels, eps=eps)
    quantile = 0.45 + 0.25 * warmup_progress
    threshold = torch.quantile(background_confidence.flatten(), quantile)
    hard_background = (background_confidence >= threshold).float()
    hard_weight = 0.35 + 0.35 * warmup_progress
    refined_mask = ((1.0 - hard_weight) * background_confidence + hard_weight * hard_background).clamp(0.02, 1.0)
    return refined_mask, anomaly_score.squeeze(0).squeeze(0)



def rank_normalize_score(score, eps=1e-8):
    flat = score.flatten()
    if flat.numel() <= 1:
        return torch.zeros_like(score)
    order = torch.argsort(flat)
    ranks = torch.empty_like(order, dtype=score.dtype)
    ranks[order] = torch.arange(flat.numel(), device=score.device, dtype=score.dtype)
    ranks = ranks / max(float(flat.numel() - 1), eps)
    return ranks.view_as(score)


def compute_rx_score(target_img, eps=1e-5):
    _, channels, height, width = target_img.shape
    pixels = target_img.squeeze(0).permute(1, 2, 0).reshape(-1, channels)
    pixels = pixels - pixels.mean(dim=0, keepdim=True)
    denom = max(pixels.shape[0] - 1, 1)
    cov = pixels.t().matmul(pixels) / float(denom)
    cov = cov + eps * torch.eye(channels, device=target_img.device, dtype=target_img.dtype)
    inv_cov = torch.linalg.pinv(cov)
    score = (pixels.matmul(inv_cov) * pixels).sum(dim=1)
    return min_max_normalize(score.view(1, 1, height, width), eps=eps)


def compute_consensus_anomaly_prior(
    recon_img,
    target_img,
    abundance_map,
    warmup_progress,
    spatial_kernel=7,
    eps=1e-8,
):
    residual_score = (recon_img - target_img).pow(2).sum(dim=1, keepdim=True)
    residual_score = min_max_normalize(residual_score, eps=eps)
    contrast_score = compute_spectral_contrast_score(target_img, kernel_size=spatial_kernel, eps=eps)
    uncertainty_score = compute_abundance_uncertainty_score(abundance_map, eps=eps)
    rx_score = compute_rx_score(target_img)

    residual_rank = rank_normalize_score(residual_score, eps=eps)
    contrast_rank = rank_normalize_score(contrast_score, eps=eps)
    uncertainty_rank = rank_normalize_score(uncertainty_score, eps=eps)
    rx_rank = rank_normalize_score(rx_score, eps=eps)

    residual_weight = 0.20 + 0.20 * warmup_progress
    contrast_weight = 0.30 - 0.05 * warmup_progress
    rx_weight = 0.30 - 0.10 * warmup_progress
    uncertainty_weight = 1.0 - residual_weight - contrast_weight - rx_weight
    anomaly_prior = (
        residual_weight * residual_rank
        + contrast_weight * contrast_rank
        + rx_weight * rx_rank
        + uncertainty_weight * uncertainty_rank
    )
    anomaly_prior = F.avg_pool2d(
        anomaly_prior,
        kernel_size=spatial_kernel,
        stride=1,
        padding=spatial_kernel // 2,
    )
    anomaly_prior = min_max_normalize(anomaly_prior, eps=eps)

    threshold = torch.quantile(anomaly_prior.flatten(), 0.78 - 0.10 * warmup_progress)
    hard_background = (anomaly_prior <= threshold).float()
    soft_background = (1.0 - anomaly_prior).clamp(0.05, 1.0)
    hard_weight = 0.25 + 0.25 * warmup_progress
    background_mask = ((1.0 - hard_weight) * soft_background + hard_weight * hard_background).clamp(0.05, 1.0)
    return background_mask, anomaly_prior.squeeze(0).squeeze(0)


def make_blindspot_weight(base_mask, blindspot_ratio, guard_window, eps=1e-8):
    if blindspot_ratio <= 0.0:
        return base_mask
    sampled = (torch.rand_like(base_mask) < blindspot_ratio).float()
    if guard_window > 1:
        if guard_window % 2 == 0:
            guard_window += 1
        sampled = F.max_pool2d(sampled, kernel_size=guard_window, stride=1, padding=guard_window // 2)
    sampled = sampled.clamp(0.0, 1.0)
    blindspot_weight = base_mask * sampled
    if blindspot_weight.sum() <= eps:
        return base_mask
    return blindspot_weight.clamp(0.0, 1.0)


def low_rank_background_loss(background_img, rank_fraction=0.15, eps=1e-8):
    _, channels, height, width = background_img.shape
    matrix = background_img.squeeze(0).reshape(channels, height * width)
    singular_values = torch.linalg.svdvals(matrix)
    keep = max(1, int(float(min(matrix.shape)) * rank_fraction))
    if singular_values.numel() <= keep:
        return torch.tensor(0.0, device=background_img.device, dtype=background_img.dtype)
    tail = singular_values[keep:]
    return tail.pow(2).mean() / (singular_values.pow(2).mean().detach() + eps)


def sparse_residual_loss(background_img, target_img, eps=1e-8):
    residual = (background_img - target_img).abs().sum(dim=1)
    l1 = residual.mean()
    l2 = residual.pow(2).mean().sqrt().detach()
    return l1 / (l2 + eps)

def spectral_angle_loss(recon_img, target_img, eps=1e-8):
    recon = recon_img.squeeze(0).permute(1, 2, 0).reshape(-1, recon_img.shape[1])
    target = target_img.squeeze(0).permute(1, 2, 0).reshape(-1, target_img.shape[1])

    dot_product = (recon * target).sum(dim=1)
    recon_norm = recon.norm(dim=1)
    target_norm = target.norm(dim=1)
    cosine = dot_product / (recon_norm * target_norm + eps)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    return torch.acos(cosine).mean()


def load_mat_file(file_path):
    try:
        import h5py

        with h5py.File(file_path, "r") as mat:
            return {key: np.array(mat[key]) for key in mat.keys()}
    except Exception:
        mat = scipy.io.loadmat(file_path)
        return {key: value for key, value in mat.items() if not key.startswith("__")}


def collect_dataset_files(dataset_dir, prefix="urban"):
    pattern = os.path.join(dataset_dir, f"{prefix}*.mat")
    dataset_files = []
    ignored_prefixes = (
        f"{prefix}_background",
        f"{prefix}_detection",
        "urban_background",
        "urban_detection",
    )
    for file_path in sorted(glob.glob(pattern)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if any(file_name.startswith(ignored_prefix) for ignored_prefix in ignored_prefixes):
            continue
        dataset_files.append(file_path)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found for pattern: {pattern}")
    return dataset_files


def extract_sample_id(file_path, prefix="urban"):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if file_name.startswith(prefix):
        suffix = file_name[len(prefix):].lstrip("_-")
        return suffix if suffix else file_name
    return file_name


def maybe_prepare_imported_dataset(args, dataset_dir):
    if not args.import_mat_dir:
        return []

    print(f"Importing external dataset files from {args.import_mat_dir}")
    results = convert_abu_directory(
        source_dir=args.import_mat_dir,
        dataset_dir=dataset_dir,
        file_pattern=args.import_pattern,
        output_prefix=args.dataset_prefix,
        target_clusters=args.import_target_clusters,
        background_clusters=args.import_background_clusters,
        normalize_data=not args.import_disable_normalize,
        overwrite=args.import_overwrite,
        random_state=args.import_random_state,
    )
    converted_count = sum(0 if item.get("skipped") else 1 for item in results)
    skipped_count = sum(1 if item.get("skipped") else 0 for item in results)
    print(f"Imported {converted_count} file(s), skipped {skipped_count} file(s).")
    return results


def save_run_summary(output_dir, results, mode_name):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, f"batch_results_{mode_name}.csv")
    txt_path = os.path.join(output_dir, f"batch_summary_{mode_name}.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "roc_auc", "stop_iteration", "elapsed_seconds", "loss_final"],
        )
        writer.writeheader()
        writer.writerows(results)

    mean_auc = float(np.mean([item["roc_auc"] for item in results]))
    mean_iter = float(np.mean([item["stop_iteration"] for item in results]))
    mean_time = float(np.mean([item["elapsed_seconds"] for item in results]))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"sample_count: {len(results)}\n")
        f.write(f"mean_auc: {mean_auc:.12f}\n")
        f.write(f"mean_stop_iteration: {mean_iter:.2f}\n")
        f.write(f"mean_elapsed_seconds: {mean_time:.4f}\n")
        f.write("\nper_sample_results:\n")
        for item in results:
            f.write(
                f"{item['sample_id']}: auc={item['roc_auc']:.12f}, "
                f"stop_iteration={item['stop_iteration']}, "
                f"elapsed_seconds={item['elapsed_seconds']:.4f}, "
                f"loss_final={item['loss_final']:.12f}\n"
            )

    return csv_path, txt_path, mean_auc


def resolve_group_count(num_channels, preferred=4):
    for group_count in range(min(preferred, num_channels), 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


class DepthwiseSeparableBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = resolve_group_count(channels)
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            torch.nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(groups, channels),
            torch.nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SpatialSpectralFusionBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = resolve_group_count(channels)
        hidden_channels = max(channels, 16)
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.spatial_branch = DepthwiseSeparableBlock(channels)
        self.spectral_branch = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(resolve_group_count(hidden_channels), hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(groups, channels),
            torch.nn.SiLU(),
        )
        self.fusion_gate = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            torch.nn.Sigmoid(),
        )
        self.out_proj = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(groups, channels),
        )

    def forward(self, x):
        spatial_feat = self.spatial_branch(x)
        spectral_feat = self.spectral_branch(x)
        fusion_weight = self.fusion_gate(torch.cat([spatial_feat, spectral_feat], dim=1))
        fused = fusion_weight * spatial_feat + (1.0 - fusion_weight) * spectral_feat
        return x + self.residual_scale * self.out_proj(fused)


class SpectralDifferenceEnhancer(torch.nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        groups = resolve_group_count(channels)
        self.kernel_size = kernel_size
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.delta_proj = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(groups, channels),
            torch.nn.SiLU(),
        )
        self.delta_gate = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, enhancement_prior=None):
        local_background = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        spectral_delta = x - local_background
        fusion_input = torch.cat([x, spectral_delta], dim=1)
        enhanced_delta = self.delta_proj(fusion_input)
        enhancement_gate = self.delta_gate(fusion_input)
        if enhancement_prior is None:
            prior_weight = spectral_delta.abs().mean(dim=1, keepdim=True)
            prior_weight = min_max_normalize(prior_weight).detach()
        else:
            prior_weight = enhancement_prior
            if prior_weight.shape[-2:] != x.shape[-2:]:
                prior_weight = F.interpolate(prior_weight, size=x.shape[-2:], mode="bilinear", align_corners=False)
            prior_weight = prior_weight.clamp(0.0, 1.0)
        return x + self.residual_scale * prior_weight * enhancement_gate * enhanced_delta


def weighted_mse_loss(prediction, target, weight, eps=1e-8):
    squared_error = (prediction - target).pow(2) * weight
    normalizer = (weight.sum() * prediction.shape[1]).clamp_min(eps)
    return squared_error.sum() / normalizer


def mask_guided_spectral_angle_loss(recon_img, target_img, mask, eps=1e-8):
    recon = recon_img.squeeze(0).permute(1, 2, 0).reshape(-1, recon_img.shape[1])
    target = target_img.squeeze(0).permute(1, 2, 0).reshape(-1, target_img.shape[1])
    weight = mask.squeeze(0).squeeze(0).reshape(-1)

    dot_product = (recon * target).sum(dim=1)
    recon_norm = recon.norm(dim=1)
    target_norm = target.norm(dim=1)
    cosine = dot_product / (recon_norm * target_norm + eps)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cosine)
    return (angle * weight).sum() / weight.sum().clamp_min(eps)


def compute_spectral_contrast_score(target_img, kernel_size=5, eps=1e-8):
    local_background = F.avg_pool2d(
        target_img,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    contrast = (target_img - local_background).pow(2).sum(dim=1, keepdim=True)
    return min_max_normalize(contrast, eps=eps)


def compute_abundance_uncertainty_score(abundance_map, eps=1e-8):
    safe_abundance = abundance_map.clamp_min(eps)
    entropy = -(safe_abundance * torch.log(safe_abundance)).sum(dim=1, keepdim=True)
    max_entropy = max(float(np.log(max(abundance_map.shape[1], 2))), eps)
    return min_max_normalize(entropy / max_entropy, eps=eps)


def compute_rank_correlation(score_a, score_b, eps=1e-8):
    rank_a = rank_normalize_score(score_a, eps=eps).flatten()
    rank_b = rank_normalize_score(score_b, eps=eps).flatten()
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    covariance = (rank_a * rank_b).mean()
    denominator = rank_a.std(unbiased=False) * rank_b.std(unbiased=False) + eps
    return covariance / denominator


def compute_top_overlap(score_a, score_b, top_fraction=0.05):
    flat_a = score_a.flatten()
    flat_b = score_b.flatten()
    top_k = max(1, int(float(flat_a.numel()) * float(top_fraction)))
    idx_a = torch.topk(flat_a, top_k, largest=True).indices
    idx_b = torch.topk(flat_b, top_k, largest=True).indices
    selected = torch.zeros(flat_a.numel(), device=flat_a.device, dtype=torch.bool)
    selected[idx_a] = True
    overlap = selected[idx_b].float().mean()
    return overlap


def compute_top_concentration(score, top_fraction=0.01, eps=1e-8):
    flat = score.flatten().clamp_min(0.0)
    top_k = max(1, int(float(flat.numel()) * float(top_fraction)))
    top_sum = torch.topk(flat, top_k, largest=True).values.sum()
    total_sum = flat.sum().clamp_min(eps)
    return top_sum / total_sum


def compute_spatial_entropy(score, eps=1e-8):
    flat = score.flatten().clamp_min(0.0)
    probability = flat / flat.sum().clamp_min(eps)
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum()
    max_entropy = max(float(np.log(max(int(flat.numel()), 2))), eps)
    return entropy / max_entropy


def compute_top_sharpness(score, top_fraction=0.01, eps=1e-8):
    flat = score.flatten().clamp_min(0.0)
    top_k = max(1, int(float(flat.numel()) * float(top_fraction)))
    top_mean = torch.topk(flat, top_k, largest=True).values.mean()
    return top_mean / flat.mean().clamp_min(eps)


def should_use_uncertainty_score(residual_score, contrast_score, uncertainty_score):
    residual_contrast_score = min_max_normalize(0.85 * residual_score + 0.15 * contrast_score)
    top_overlap = compute_top_overlap(residual_contrast_score, uncertainty_score, top_fraction=0.05)
    residual_corr = compute_rank_correlation(residual_score, uncertainty_score)
    contrast_corr = compute_rank_correlation(contrast_score, uncertainty_score)
    return bool((top_overlap > 0.20 and residual_corr > 0.15 and contrast_corr > 0.05).item())


def compute_stationary_haar_highfreq_score(residual_score, eps=1e-8):
    kernels = torch.tensor(
        [
            [[1.0, 1.0], [-1.0, -1.0]],
            [[1.0, -1.0], [1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ],
        device=residual_score.device,
        dtype=residual_score.dtype,
    ).view(3, 1, 2, 2) * 0.5
    padded = F.pad(residual_score, (0, 1, 0, 1), mode="replicate")
    detail = F.conv2d(padded, kernels)
    highfreq_score = torch.sqrt((detail * detail).sum(dim=1, keepdim=True).clamp_min(eps))
    return min_max_normalize(highfreq_score, eps=eps)


def compute_pywt_highfreq_score(residual_score, wavelet="haar", level=1, eps=1e-8):
    try:
        import pywt
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyWavelets is required for --highfreq-score-mode pywt. Install PyWavelets first.") from exc

    residual_np = residual_score.detach().cpu().squeeze().numpy()
    max_level = pywt.dwtn_max_level(residual_np.shape, wavelet)
    effective_level = max(1, min(int(level), int(max_level)))
    coeffs = pywt.wavedec2(residual_np, wavelet=wavelet, level=effective_level, mode="periodization")
    detail_coeffs = [np.zeros_like(coeffs[0])]
    for detail in coeffs[1:]:
        detail_coeffs.append(tuple(np.asarray(item) for item in detail))
    reconstructed = pywt.waverec2(detail_coeffs, wavelet=wavelet, mode="periodization")
    reconstructed = np.abs(reconstructed[: residual_np.shape[0], : residual_np.shape[1]])
    highfreq_score = torch.from_numpy(reconstructed).to(device=residual_score.device, dtype=residual_score.dtype)
    return min_max_normalize(highfreq_score.view_as(residual_score), eps=eps)


def compute_highfreq_score(residual_score, mode="none", wavelet="haar", level=1):
    if mode == "none":
        return None
    if mode == "stationary_haar":
        return compute_stationary_haar_highfreq_score(residual_score)
    if mode == "pywt":
        return compute_pywt_highfreq_score(residual_score, wavelet=wavelet, level=level)
    raise ValueError(f"Unknown high-frequency score mode: {mode}")


def compute_highfreq_adaptive_alpha(
    base_score,
    highfreq_score,
    low_alpha=0.1,
    high_alpha=0.4,
    top_overlap_threshold=0.20,
    rank_corr_threshold=0.15,
    min_top_concentration=0.08,
    max_entropy=0.85,
    min_peak_ratio=0.75,
    max_peak_ratio=3.0,
):
    top_overlap = compute_top_overlap(base_score, highfreq_score, top_fraction=0.05)
    rank_corr = compute_rank_correlation(base_score, highfreq_score)
    hf_top_concentration = compute_top_concentration(highfreq_score, top_fraction=0.01)
    hf_entropy = compute_spatial_entropy(highfreq_score)
    hf_sharpness = compute_top_sharpness(highfreq_score, top_fraction=0.01)
    base_sharpness = compute_top_sharpness(base_score, top_fraction=0.01)
    hf_to_base_peak_ratio = hf_sharpness / base_sharpness.clamp_min(1e-8)
    agreement_ok = top_overlap >= top_overlap_threshold and rank_corr >= rank_corr_threshold
    concentration_ok = hf_top_concentration >= min_top_concentration
    entropy_ok = hf_entropy <= max_entropy
    peak_ratio_ok = hf_to_base_peak_ratio >= min_peak_ratio and hf_to_base_peak_ratio <= max_peak_ratio
    use_high_alpha = bool((agreement_ok and concentration_ok and entropy_ok and peak_ratio_ok).item())
    alpha = high_alpha if use_high_alpha else low_alpha
    diagnostics = {
        "top_overlap": float(top_overlap.item()),
        "rank_corr": float(rank_corr.item()),
        "hf_top_concentration": float(hf_top_concentration.item()),
        "hf_entropy": float(hf_entropy.item()),
        "hf_to_base_peak_ratio": float(hf_to_base_peak_ratio.item()),
        "use_high_alpha": float(use_high_alpha),
    }
    return float(alpha), diagnostics


def compute_input_edge_score(target_img, eps=1e-8):
    intensity = target_img.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=target_img.device,
        dtype=target_img.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=target_img.device,
        dtype=target_img.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(intensity, sobel_x, padding=1)
    grad_y = F.conv2d(intensity, sobel_y, padding=1)
    edge_score = torch.sqrt((grad_x * grad_x + grad_y * grad_y).clamp_min(eps))
    return min_max_normalize(edge_score, eps=eps)


def compute_highfreq_soft_alpha_map(
    base_score,
    highfreq_score,
    low_alpha=0.05,
    high_alpha=0.4,
    top_quantile=0.85,
    gate_slope=16.0,
    diffuse_guard=0.5,
    edge_score=None,
    edge_guard_enabled=False,
    edge_guard_quantile=0.85,
    edge_guard_strength=0.75,
    **adaptive_kwargs,
):
    _, diagnostics = compute_highfreq_adaptive_alpha(
        base_score,
        highfreq_score,
        low_alpha=low_alpha,
        high_alpha=high_alpha,
        **adaptive_kwargs,
    )
    base_rank = rank_normalize_score(base_score)
    hf_rank = rank_normalize_score(highfreq_score)
    base_high = torch.sigmoid((base_rank - top_quantile) * gate_slope)
    hf_high = torch.sigmoid((hf_rank - top_quantile) * gate_slope)
    agreement_map = base_high * hf_high

    hf_without_base = hf_high * torch.sigmoid((top_quantile - base_rank) * gate_slope)
    guard_factor = 1.0 if diagnostics["use_high_alpha"] > 0.5 else float(diffuse_guard)
    alpha_map = low_alpha + (high_alpha - low_alpha) * agreement_map * guard_factor
    alpha_map = alpha_map * (1.0 - hf_without_base) + low_alpha * hf_without_base
    if edge_guard_enabled and edge_score is not None:
        edge_rank = rank_normalize_score(edge_score)
        edge_high = torch.sigmoid((edge_rank - edge_guard_quantile) * gate_slope)
        base_weak = torch.sigmoid((top_quantile - base_rank) * gate_slope)
        edge_guard = edge_high * base_weak
        alpha_map = alpha_map * (1.0 - float(edge_guard_strength) * edge_guard.clamp(0.0, 1.0))
        diagnostics["edge_guard_mean"] = float(edge_guard.mean().item())
        diagnostics["edge_guard_max"] = float(edge_guard.max().item())
    alpha_map = alpha_map.clamp(min=min(low_alpha, high_alpha), max=max(low_alpha, high_alpha))
    diagnostics["alpha_mean"] = float(alpha_map.mean().item())
    diagnostics["alpha_max"] = float(alpha_map.max().item())
    diagnostics["alpha_min"] = float(alpha_map.min().item())
    return alpha_map, diagnostics


def fuse_with_highfreq_score(
    base_score,
    highfreq_score,
    highfreq_weight,
    fusion_mode="fixed",
    adaptive_low_alpha=0.1,
    adaptive_high_alpha=0.4,
    adaptive_top_overlap_threshold=0.20,
    adaptive_rank_corr_threshold=0.15,
    adaptive_min_top_concentration=0.08,
    adaptive_max_entropy=0.85,
    adaptive_min_peak_ratio=0.75,
    adaptive_max_peak_ratio=3.0,
    soft_map_top_quantile=0.85,
    soft_map_gate_slope=16.0,
    soft_map_diffuse_guard=0.5,
    input_dynamic_range=None,
    diagnostic_dynamic_range_threshold=10.0,
    diagnostic_texture_entropy_threshold=0.80,
    diagnostic_texture_concentration_threshold=0.25,
    diagnostic_fixed_alpha=None,
    diagnostic_soft_low_alpha=None,
    diagnostic_soft_high_alpha=None,
    edge_score=None,
    edge_guard_enabled=False,
    edge_guard_quantile=0.85,
    edge_guard_strength=0.75,
):
    if highfreq_score is None:
        return base_score, None, None, {}
    if fusion_mode == "adaptive":
        alpha, diagnostics = compute_highfreq_adaptive_alpha(
            base_score,
            highfreq_score,
            low_alpha=adaptive_low_alpha,
            high_alpha=adaptive_high_alpha,
            top_overlap_threshold=adaptive_top_overlap_threshold,
            rank_corr_threshold=adaptive_rank_corr_threshold,
            min_top_concentration=adaptive_min_top_concentration,
            max_entropy=adaptive_max_entropy,
            min_peak_ratio=adaptive_min_peak_ratio,
            max_peak_ratio=adaptive_max_peak_ratio,
        )
    elif fusion_mode == "soft_map":
        alpha_map, diagnostics = compute_highfreq_soft_alpha_map(
            base_score,
            highfreq_score,
            low_alpha=adaptive_low_alpha,
            high_alpha=adaptive_high_alpha,
            top_quantile=soft_map_top_quantile,
            gate_slope=soft_map_gate_slope,
            diffuse_guard=soft_map_diffuse_guard,
            edge_score=edge_score,
            edge_guard_enabled=edge_guard_enabled,
            edge_guard_quantile=edge_guard_quantile,
            edge_guard_strength=edge_guard_strength,
            top_overlap_threshold=adaptive_top_overlap_threshold,
            rank_corr_threshold=adaptive_rank_corr_threshold,
            min_top_concentration=adaptive_min_top_concentration,
            max_entropy=adaptive_max_entropy,
            min_peak_ratio=adaptive_min_peak_ratio,
            max_peak_ratio=adaptive_max_peak_ratio,
        )
        fused_score = min_max_normalize((1.0 - alpha_map) * base_score + alpha_map * highfreq_score)
        alpha = float(alpha_map.mean().item())
        diagnostics["selected_fusion"] = "soft_map"
        return fused_score, alpha, alpha_map, diagnostics
    elif fusion_mode == "diagnostic":
        soft_low_alpha = adaptive_low_alpha if diagnostic_soft_low_alpha is None else float(diagnostic_soft_low_alpha)
        soft_high_alpha = adaptive_high_alpha if diagnostic_soft_high_alpha is None else float(diagnostic_soft_high_alpha)
        fixed_alpha = adaptive_high_alpha if diagnostic_fixed_alpha is None else float(diagnostic_fixed_alpha)
        _, diagnostics = compute_highfreq_adaptive_alpha(
            base_score,
            highfreq_score,
            low_alpha=soft_low_alpha,
            high_alpha=soft_high_alpha,
            top_overlap_threshold=adaptive_top_overlap_threshold,
            rank_corr_threshold=adaptive_rank_corr_threshold,
            min_top_concentration=adaptive_min_top_concentration,
            max_entropy=adaptive_max_entropy,
            min_peak_ratio=adaptive_min_peak_ratio,
            max_peak_ratio=adaptive_max_peak_ratio,
        )
        dynamic_range = 0.0 if input_dynamic_range is None else float(input_dynamic_range)
        concentrated_texture = (
            diagnostics["hf_entropy"] < float(diagnostic_texture_entropy_threshold)
            and diagnostics["hf_top_concentration"] > float(diagnostic_texture_concentration_threshold)
        )
        use_soft_map = (
            dynamic_range > float(diagnostic_dynamic_range_threshold)
            or diagnostics["hf_to_base_peak_ratio"] > float(adaptive_max_peak_ratio)
            or concentrated_texture
        )
        diagnostics["concentrated_texture_guard"] = float(concentrated_texture)
        diagnostics["input_dynamic_range"] = dynamic_range
        if use_soft_map:
            alpha_map, diagnostics = compute_highfreq_soft_alpha_map(
                base_score,
                highfreq_score,
                low_alpha=soft_low_alpha,
                high_alpha=soft_high_alpha,
                top_quantile=soft_map_top_quantile,
                gate_slope=soft_map_gate_slope,
                diffuse_guard=soft_map_diffuse_guard,
                edge_score=edge_score,
                edge_guard_enabled=edge_guard_enabled,
                edge_guard_quantile=edge_guard_quantile,
                edge_guard_strength=edge_guard_strength,
                top_overlap_threshold=adaptive_top_overlap_threshold,
                rank_corr_threshold=adaptive_rank_corr_threshold,
                min_top_concentration=adaptive_min_top_concentration,
                max_entropy=adaptive_max_entropy,
                min_peak_ratio=adaptive_min_peak_ratio,
                max_peak_ratio=adaptive_max_peak_ratio,
            )
            diagnostics["input_dynamic_range"] = dynamic_range
            diagnostics["selected_fusion"] = "soft_map"
            fused_score = min_max_normalize((1.0 - alpha_map) * base_score + alpha_map * highfreq_score)
            alpha = float(alpha_map.mean().item())
            return fused_score, alpha, alpha_map, diagnostics
        alpha = min(1.0, max(0.0, float(fixed_alpha)))
        diagnostics["selected_fusion"] = "fixed"
        return min_max_normalize((1.0 - alpha) * base_score + alpha * highfreq_score), alpha, None, diagnostics
    else:
        alpha = min(1.0, max(0.0, float(highfreq_weight)))
        diagnostics = {}
    if alpha <= 0.0:
        return base_score, alpha, None, diagnostics
    return min_max_normalize((1.0 - alpha) * base_score + alpha * highfreq_score), alpha, None, diagnostics


def fuse_detection_scores(residual_score, contrast_score, uncertainty_score, weights, adaptive=False):
    if adaptive and not should_use_uncertainty_score(residual_score, contrast_score, uncertainty_score):
        fused_score = 0.85 * residual_score + 0.15 * contrast_score
        return min_max_normalize(fused_score)

    residual_weight, contrast_weight, uncertainty_weight = weights
    fused_score = (
        residual_weight * residual_score
        + contrast_weight * contrast_score
        + uncertainty_weight * uncertainty_score
    )
    return min_max_normalize(fused_score)


def compute_mask_guided_joint_loss(
    out,
    out_h,
    target_img,
    mask,
    lam,
    sam_weight,
    use_sam_loss,
    loss_mask=None,
    low_rank_weight=0.0,
    sparse_weight=0.0,
    low_rank_fraction=0.15,
):
    b, channels, height, width = out.shape
    effective_mask = mask if loss_mask is None else loss_mask
    loss1 = weighted_mse_loss(out_h, target_img, effective_mask)
    loss2 = total_variation(out.view(b * channels, height, width))
    if use_sam_loss and sam_weight > 0.0:
        loss3 = mask_guided_spectral_angle_loss(out_h, target_img, mask)
    else:
        loss3 = torch.tensor(0.0, device=out_h.device)
    if low_rank_weight > 0.0:
        loss4 = low_rank_background_loss(out_h, rank_fraction=low_rank_fraction)
    else:
        loss4 = torch.tensor(0.0, device=out_h.device)
    if sparse_weight > 0.0:
        loss5 = sparse_residual_loss(out_h, target_img)
    else:
        loss5 = torch.tensor(0.0, device=out_h.device)
    total_loss = loss1 + lam * loss2 + sam_weight * loss3 + low_rank_weight * loss4 + sparse_weight * loss5
    metrics = {
        "total_loss": float(total_loss.item()),
        "mse_loss": float(loss1.item()),
        "tv_loss": float(loss2.item()),
        "sam_loss": float(loss3.item()),
        "low_rank_loss": float(loss4.item()),
        "sparse_loss": float(loss5.item()),
    }
    return total_loss, metrics


def compute_final_artifacts(
    net,
    net_input_saved,
    mask_var,
    img_var,
    spatial_kernel,
    weights,
    adaptive_score_fusion=False,
    highfreq_score_mode="none",
    highfreq_weight=0.0,
    highfreq_wavelet="haar",
    highfreq_level=1,
    highfreq_fusion_mode="fixed",
    highfreq_adaptive_low_alpha=0.1,
    highfreq_adaptive_high_alpha=0.4,
    highfreq_adaptive_top_overlap=0.20,
    highfreq_adaptive_rank_corr=0.15,
    highfreq_adaptive_min_top_concentration=0.08,
    highfreq_adaptive_max_entropy=0.85,
    highfreq_adaptive_min_peak_ratio=0.75,
    highfreq_adaptive_max_peak_ratio=3.0,
    highfreq_soft_map_top_quantile=0.85,
    highfreq_soft_map_gate_slope=16.0,
    highfreq_soft_map_diffuse_guard=0.5,
    highfreq_diagnostic_dynamic_range=10.0,
    highfreq_diagnostic_texture_entropy=0.80,
    highfreq_diagnostic_texture_concentration=0.25,
    highfreq_diagnostic_fixed_alpha=None,
    highfreq_diagnostic_soft_low_alpha=None,
    highfreq_diagnostic_soft_high_alpha=None,
    raw_input_dynamic_range=None,
    highfreq_edge_guard=False,
    highfreq_edge_guard_quantile=0.85,
    highfreq_edge_guard_strength=0.75,
):
    enhancement_prior = (1.0 - mask_var.detach()).clamp(0.0, 1.0)
    with torch.no_grad():
        out, out_h = net(net_input_saved, enhancement_prior=enhancement_prior)
        residual_score = (out_h - img_var).pow(2).sum(dim=1, keepdim=True)
        residual_score = min_max_normalize(residual_score)
        contrast_score = compute_spectral_contrast_score(img_var, kernel_size=spatial_kernel)
        uncertainty_score = compute_abundance_uncertainty_score(out)
        highfreq_score = compute_highfreq_score(
            residual_score,
            mode=highfreq_score_mode,
            wavelet=highfreq_wavelet,
            level=highfreq_level,
        )
        edge_score = compute_input_edge_score(img_var) if highfreq_edge_guard else None
        fused_score = fuse_detection_scores(
            residual_score,
            contrast_score,
            uncertainty_score,
            weights=weights,
            adaptive=adaptive_score_fusion,
        )
        fused_score, highfreq_alpha, highfreq_alpha_map, highfreq_diagnostics = fuse_with_highfreq_score(
            fused_score,
            highfreq_score,
            highfreq_weight,
            fusion_mode=highfreq_fusion_mode,
            adaptive_low_alpha=highfreq_adaptive_low_alpha,
            adaptive_high_alpha=highfreq_adaptive_high_alpha,
            adaptive_top_overlap_threshold=highfreq_adaptive_top_overlap,
            adaptive_rank_corr_threshold=highfreq_adaptive_rank_corr,
            adaptive_min_top_concentration=highfreq_adaptive_min_top_concentration,
            adaptive_max_entropy=highfreq_adaptive_max_entropy,
            adaptive_min_peak_ratio=highfreq_adaptive_min_peak_ratio,
            adaptive_max_peak_ratio=highfreq_adaptive_max_peak_ratio,
            soft_map_top_quantile=highfreq_soft_map_top_quantile,
            soft_map_gate_slope=highfreq_soft_map_gate_slope,
            soft_map_diffuse_guard=highfreq_soft_map_diffuse_guard,
            input_dynamic_range=(
                float((img_var.max() - img_var.min()).item())
                if raw_input_dynamic_range is None
                else float(raw_input_dynamic_range)
            ),
            diagnostic_dynamic_range_threshold=highfreq_diagnostic_dynamic_range,
            diagnostic_texture_entropy_threshold=highfreq_diagnostic_texture_entropy,
            diagnostic_texture_concentration_threshold=highfreq_diagnostic_texture_concentration,
            diagnostic_fixed_alpha=highfreq_diagnostic_fixed_alpha,
            diagnostic_soft_low_alpha=highfreq_diagnostic_soft_low_alpha,
            diagnostic_soft_high_alpha=highfreq_diagnostic_soft_high_alpha,
            edge_score=edge_score,
            edge_guard_enabled=highfreq_edge_guard,
            edge_guard_quantile=highfreq_edge_guard_quantile,
            edge_guard_strength=highfreq_edge_guard_strength,
        )

    highfreq_score_np = None if highfreq_score is None else highfreq_score.detach().cpu().squeeze().numpy()
    highfreq_alpha_np = None if highfreq_alpha is None else np.asarray([highfreq_alpha], dtype=np.float32)
    highfreq_alpha_map_np = None if highfreq_alpha_map is None else highfreq_alpha_map.detach().cpu().squeeze().numpy()
    return {
        "background_img": out_h.detach().cpu().squeeze(0).numpy(),
        "residual_score": residual_score.detach().cpu().squeeze().numpy(),
        "contrast_score": contrast_score.detach().cpu().squeeze().numpy(),
        "uncertainty_score": uncertainty_score.detach().cpu().squeeze().numpy(),
        "highfreq_score": highfreq_score_np,
        "highfreq_alpha": highfreq_alpha_np,
        "highfreq_alpha_map": highfreq_alpha_map_np,
        "highfreq_diagnostics": highfreq_diagnostics,
        "fused_score": fused_score.detach().cpu().squeeze().numpy(),
        "mask": mask_var.detach().cpu().squeeze().numpy(),
    }


class DPMN(torch.nn.Module):
    def __init__(self, input_depth, rmax, band, e_torch, pad):
        super(DPMN, self).__init__()
        self.input_depth = input_depth
        self.band = band
        self.conv1 = torch.nn.Sequential(
            AGM(
                input_depth,
                rmax,
                num_channels_down=[128],
                num_channels_up=[128],
                num_channels_skip=[4],
                filter_size_up=3,
                filter_size_down=3,
                filter_skip_size=1,
                upsample_mode="bilinear",
                need1x1_up=True,
                need_sigmoid=True,
                need_bias=True,
                pad=pad,
                act_fun="LeakyReLU",
            ).type(dtype)
        )
        self.dual_branch_fusion = SpatialSpectralFusionBlock(rmax)
        self.spectral_difference_enhancer = SpectralDifferenceEnhancer(rmax)
        self.abundance_norm = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(input_depth, band)
        self.fc1.weight.data = e_torch.clone()
        self.conv = torch.nn.Conv2d(in_channels=band, out_channels=band, kernel_size=3, stride=1, padding=1)

    def forward(self, x, enhancement_prior=None):
        x = self.conv1(x)
        x = self.dual_branch_fusion(x)
        x = self.spectral_difference_enhancer(x, enhancement_prior=enhancement_prior)
        x = self.abundance_norm(x)
        x_size = x.size()
        rmax = x_size[1]
        row = x_size[2]
        col = x_size[3]
        out = torch.transpose(x.view(self.input_depth, -1), 1, 0)
        out_h = self.fc1(out)
        out = torch.transpose(out, 1, 0)
        out_h = torch.transpose(out_h, 1, 0)
        out_h = out_h.view(1, self.band, row, col)
        out = out.view(1, rmax, row, col)
        out_h = self.conv(out_h)
        return out, out_h


def run_single_sample(
    file_path,
    sample_id,
    mode_config,
    mode_name,
    visualization_dir,
    artifact_root_dir,
    log_interval,
    normalize_inputs,
    num_iter_override=None,
):
    start_time = time.time()
    torch.cuda.empty_cache()

    residual_root_path = os.path.join(artifact_root_dir, "detection_maps", mode_name)
    background_root_path = os.path.join(artifact_root_dir, "background_maps", mode_name)
    score_root_path = os.path.join(artifact_root_dir, "score_components", mode_name)
    mask_history_root_path = os.path.join(artifact_root_dir, "mask_history", mode_name)
    ensure_dir(residual_root_path)
    ensure_dir(background_root_path)
    ensure_dir(score_root_path)
    ensure_dir(mask_history_root_path)

    pad = "reflection"
    opt_over = "net"
    method = "2D"
    lr = 0.001
    num_iter = int(num_iter_override) if num_iter_override is not None else int(mode_config.get("num_iter", 1500))
    param_noise = False
    reg_noise_std = 0
    lam = 0.001
    sam_weight = mode_config["sam_weight"]
    residual_weight = mode_config["residual_weight"]
    contrast_weight = mode_config["contrast_weight"]
    uncertainty_weight = mode_config["uncertainty_weight"]
    spatial_kernel = 7
    mask_update_interval = 50
    mask_momentum = 0.9
    warmup_iters = max(200, num_iter // 6)
    sam_start_iter = max(100, warmup_iters)
    sam_full_iter = max(sam_start_iter + 1, num_iter // 2)
    use_adaptive_mask = mode_config["adaptive_mask"]
    use_sam_loss = mode_config["sam_loss"]
    prior_mode = mode_config.get("prior_mode", "adaptive")
    use_blindspot = bool(mode_config.get("blindspot", False))
    blindspot_ratio = float(mode_config.get("blindspot_ratio", 0.25))
    guard_window = int(mode_config.get("guard_window", 5))
    low_rank_weight = float(mode_config.get("low_rank_weight", 0.0))
    sparse_weight = float(mode_config.get("sparse_weight", 0.0))
    low_rank_fraction = float(mode_config.get("low_rank_fraction", 0.15))
    use_superpixel_perturb = bool(mode_config.get("superpixel_perturb", False))
    use_online_background_mining = bool(mode_config.get("online_background_mining", False))
    superpixel_segments = int(mode_config.get("superpixel_segments", 256))
    superpixel_compactness = float(mode_config.get("superpixel_compactness", 0.08))
    sp_target_weight = float(mode_config.get("sp_target_weight", 1.0))
    adaptive_score_fusion = bool(mode_config.get("adaptive_score_fusion", False))
    highfreq_score_mode = mode_config.get("highfreq_score_mode", "none")
    highfreq_weight = float(mode_config.get("highfreq_weight", 0.0))
    highfreq_wavelet = mode_config.get("highfreq_wavelet", "haar")
    highfreq_level = int(mode_config.get("highfreq_level", 1))
    highfreq_fusion_mode = mode_config.get("highfreq_fusion_mode", "fixed")
    highfreq_adaptive_low_alpha = float(mode_config.get("highfreq_adaptive_low_alpha", 0.1))
    highfreq_adaptive_high_alpha = float(mode_config.get("highfreq_adaptive_high_alpha", 0.4))
    highfreq_adaptive_top_overlap = float(mode_config.get("highfreq_adaptive_top_overlap", 0.20))
    highfreq_adaptive_rank_corr = float(mode_config.get("highfreq_adaptive_rank_corr", 0.15))
    highfreq_adaptive_min_top_concentration = float(mode_config.get("highfreq_adaptive_min_top_concentration", 0.08))
    highfreq_adaptive_max_entropy = float(mode_config.get("highfreq_adaptive_max_entropy", 0.85))
    highfreq_adaptive_min_peak_ratio = float(mode_config.get("highfreq_adaptive_min_peak_ratio", 0.75))
    highfreq_adaptive_max_peak_ratio = float(mode_config.get("highfreq_adaptive_max_peak_ratio", 3.0))
    highfreq_soft_map_top_quantile = float(mode_config.get("highfreq_soft_map_top_quantile", 0.85))
    highfreq_soft_map_gate_slope = float(mode_config.get("highfreq_soft_map_gate_slope", 16.0))
    highfreq_soft_map_diffuse_guard = float(mode_config.get("highfreq_soft_map_diffuse_guard", 0.5))
    highfreq_diagnostic_dynamic_range = float(mode_config.get("highfreq_diagnostic_dynamic_range", 10.0))
    highfreq_diagnostic_texture_entropy = float(mode_config.get("highfreq_diagnostic_texture_entropy", 0.80))
    highfreq_diagnostic_texture_concentration = float(mode_config.get("highfreq_diagnostic_texture_concentration", 0.25))
    highfreq_diagnostic_fixed_alpha = mode_config.get("highfreq_diagnostic_fixed_alpha", None)
    highfreq_diagnostic_soft_low_alpha = mode_config.get("highfreq_diagnostic_soft_low_alpha", None)
    highfreq_diagnostic_soft_high_alpha = mode_config.get("highfreq_diagnostic_soft_high_alpha", None)
    highfreq_edge_guard = bool(mode_config.get("highfreq_edge_guard", False))
    highfreq_edge_guard_quantile = float(mode_config.get("highfreq_edge_guard_quantile", 0.85))
    highfreq_edge_guard_strength = float(mode_config.get("highfreq_edge_guard_strength", 0.75))
    metric_history = []

    mat = load_mat_file(file_path)
    img_h5 = mat["image"]
    e = mat["A"]
    label = mat["mask"]

    label_np = np.array(label).transpose(1, 0)
    e_np = np.array(e).transpose(1, 0)
    img_np = np.array(img_h5).transpose(0, 2, 1)
    raw_input_dynamic_range = float(np.max(img_np) - np.min(img_np))
    print_sample_stats(sample_id, img_np, e_np, label_np, stage="raw")

    if normalize_inputs:
        img_np = normalize_numpy_minmax(img_np)
        e_np = normalize_endmember_columns(e_np)
        print_sample_stats(sample_id, img_np, e_np, label_np, stage="normalized")

    e_torch = torch.from_numpy(e_np).type(dtype)
    sp_labels_np = None
    train_target_np = img_np
    if use_superpixel_perturb:
        sp_labels_np = compute_superpixel_labels(
            img_np,
            desired_segments=superpixel_segments,
            compactness=superpixel_compactness,
        )
        sp_pooled_np = pool_hsi_by_superpixel(img_np, sp_labels_np)
        train_target_np = sp_target_weight * sp_pooled_np + (1.0 - sp_target_weight) * img_np
        print(
            f"sample {sample_id} [sp_imp] superpixels={int(sp_labels_np.max()) + 1}, "
            f"target_delta_mean={np.abs(train_target_np - img_np).mean():.6f}"
        )

    img_var = torch.from_numpy(img_np).type(dtype)
    train_target_var = torch.from_numpy(train_target_np).type(dtype)

    img_size = img_var.size()
    e_size = e_torch.size()
    rmax = e_size[1]
    band = img_size[0]
    row = img_size[1]
    col = img_size[2]
    input_depth = rmax

    net = DPMN(input_depth=input_depth, rmax=rmax, band=band, e_torch=e_torch, pad=pad)
    torch.nn.init.normal_(net.conv.weight, mean=0, std=0.01)
    net_input = get_noise(input_depth, method, img_np.shape[1:], noise_type="u").type(dtype)
    net.cuda()

    img_var = img_var[None, :].cuda()
    train_target_var = train_target_var[None, :].cuda()
    e_torch = e_torch.cuda()
    sp_labels_var = None
    if sp_labels_np is not None:
        sp_labels_var = torch.from_numpy(sp_labels_np.astype(np.int64)).cuda()

    mask_var = torch.ones(1, 1, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()
    mask_snapshot_iters = {1, 200, 800, num_iter}
    mask_snapshot_records = []

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    lossiter = []
    loss_last = 0

    def get_sam_weight(iter_num):
        if not use_sam_loss:
            return 0.0
        if iter_num < sam_start_iter:
            return 0.0
        if iter_num >= sam_full_iter:
            return float(mode_config["sam_weight"])
        progress = float(iter_num - sam_start_iter) / float(sam_full_iter - sam_start_iter)
        return float(mode_config["sam_weight"]) * progress

    def closure1(iter_num, mask_varr, residual_varr):
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        local_net_input = net_input_saved
        if reg_noise_std > 0:
            local_net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        enhancement_prior = (1.0 - mask_varr.detach()).clamp(0.0, 1.0)
        out, out_h = net(local_net_input, enhancement_prior=enhancement_prior)

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if use_adaptive_mask and iter_num % mask_update_interval == 0 and iter_num != 0:
            warmup_progress = min(1.0, float(iter_num) / float(warmup_iters))
            if use_online_background_mining or prior_mode == "sp_imp":
                refined_mask, residual_img = compute_online_background_mining_mask(
                    recon_img=out_h.detach(),
                    target_img=img_var.detach(),
                    abundance_map=out.detach(),
                    warmup_progress=warmup_progress,
                    spatial_kernel=spatial_kernel,
                    superpixel_labels=sp_labels_var,
                )
            elif prior_mode == "consensus":
                refined_mask, residual_img = compute_consensus_anomaly_prior(
                    recon_img=out_h.detach(),
                    target_img=img_var.detach(),
                    abundance_map=out.detach(),
                    warmup_progress=warmup_progress,
                    spatial_kernel=spatial_kernel,
                )
            else:
                refined_mask, residual_img = compute_adaptive_mask(
                    recon_img=out_h.detach(),
                    target_img=img_var.detach(),
                    warmup_progress=warmup_progress,
                    spatial_kernel=spatial_kernel,
                )
            residual_var_clone = residual_img
            mask_var_clone = mask_momentum * mask_var_clone + (1.0 - mask_momentum) * refined_mask

        current_sam_weight = get_sam_weight(iter_num)
        loss_mask = mask_var_clone
        blindspot_active = 0.0
        if use_blindspot and iter_num >= warmup_iters:
            loss_mask = make_blindspot_weight(mask_var_clone, blindspot_ratio, guard_window)
            blindspot_active = 1.0
        total_loss, metrics = compute_mask_guided_joint_loss(
            out=out,
            out_h=out_h,
            target_img=train_target_var,
            mask=mask_var_clone,
            lam=lam,
            sam_weight=current_sam_weight,
            use_sam_loss=use_sam_loss,
            loss_mask=loss_mask,
            low_rank_weight=low_rank_weight,
            sparse_weight=sparse_weight,
            low_rank_fraction=low_rank_fraction,
        )
        residual_score = (out_h - img_var).pow(2).sum(dim=1, keepdim=True)
        residual_score = min_max_normalize(residual_score)
        contrast_score = compute_spectral_contrast_score(img_var, kernel_size=spatial_kernel)
        uncertainty_score = compute_abundance_uncertainty_score(out)
        fused_score = fuse_detection_scores(
            residual_score,
            contrast_score,
            uncertainty_score,
            weights=(residual_weight, contrast_weight, uncertainty_weight),
            adaptive=adaptive_score_fusion,
        )
        residual_var_clone = fused_score.detach().squeeze(0).squeeze(0)
        total_loss.requires_grad_(True)
        total_loss.backward()
        metrics["total_loss"] = float(total_loss.item())
        metrics["sam_weight"] = float(current_sam_weight)
        metrics["blindspot_active"] = float(blindspot_active)
        return mask_var_clone, residual_var_clone, total_loss, metrics

    p = get_params(opt_over, net, net_input)
    print(f"Starting optimization with ADAM for sample {sample_id} under mode {mode_name}")
    optimizer = torch.optim.Adam(p, lr=lr)

    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, total_loss, metrics = closure1(j, mask_var, residual_varr)
        optimizer.step()
        lossiter.append(total_loss.item())
        metric_history.append(metrics)
        if (j + 1) in mask_snapshot_iters:
            mask_snapshot_records.append(
                (
                    int(j + 1),
                    mask_var.detach().cpu().squeeze().numpy().copy(),
                )
            )

        if ((j + 1) % log_interval == 0) or j == 0 or j == num_iter - 1:
            print(
                f"mode: {mode_name}; sample {sample_id}; iteration: {j + 1}; "
                f"loss: {metrics['total_loss']:.6f}; mse: {metrics['mse_loss']:.6f}; "
                f"tv: {metrics['tv_loss']:.6f}; sam: {metrics['sam_loss']:.6f}; "
                f"sam_w: {metrics['sam_weight']:.6f}; "
                f"lr: {metrics.get('low_rank_loss', 0.0):.6f}; "
                f"sp: {metrics.get('sparse_loss', 0.0):.6f}; "
                f"bs: {metrics.get('blindspot_active', 0.0):.0f}"
            )

        loss_last = total_loss.item()

        if j == num_iter - 1:
            final_artifacts = compute_final_artifacts(
                net,
                net_input_saved,
                mask_var,
                img_var,
                spatial_kernel,
                weights=(residual_weight, contrast_weight, uncertainty_weight),
                adaptive_score_fusion=adaptive_score_fusion,
                highfreq_score_mode=highfreq_score_mode,
                highfreq_weight=highfreq_weight,
                highfreq_wavelet=highfreq_wavelet,
                highfreq_level=highfreq_level,
                highfreq_fusion_mode=highfreq_fusion_mode,
                highfreq_adaptive_low_alpha=highfreq_adaptive_low_alpha,
                highfreq_adaptive_high_alpha=highfreq_adaptive_high_alpha,
                highfreq_adaptive_top_overlap=highfreq_adaptive_top_overlap,
                highfreq_adaptive_rank_corr=highfreq_adaptive_rank_corr,
                highfreq_adaptive_min_top_concentration=highfreq_adaptive_min_top_concentration,
                highfreq_adaptive_max_entropy=highfreq_adaptive_max_entropy,
                highfreq_adaptive_min_peak_ratio=highfreq_adaptive_min_peak_ratio,
                highfreq_adaptive_max_peak_ratio=highfreq_adaptive_max_peak_ratio,
                highfreq_soft_map_top_quantile=highfreq_soft_map_top_quantile,
                highfreq_soft_map_gate_slope=highfreq_soft_map_gate_slope,
                highfreq_soft_map_diffuse_guard=highfreq_soft_map_diffuse_guard,
                highfreq_diagnostic_dynamic_range=highfreq_diagnostic_dynamic_range,
                highfreq_diagnostic_texture_entropy=highfreq_diagnostic_texture_entropy,
                highfreq_diagnostic_texture_concentration=highfreq_diagnostic_texture_concentration,
                highfreq_diagnostic_fixed_alpha=highfreq_diagnostic_fixed_alpha,
                highfreq_diagnostic_soft_low_alpha=highfreq_diagnostic_soft_low_alpha,
                highfreq_diagnostic_soft_high_alpha=highfreq_diagnostic_soft_high_alpha,
                raw_input_dynamic_range=raw_input_dynamic_range,
                highfreq_edge_guard=highfreq_edge_guard,
                highfreq_edge_guard_quantile=highfreq_edge_guard_quantile,
                highfreq_edge_guard_strength=highfreq_edge_guard_strength,
            )
            residual_np = final_artifacts["fused_score"]
            residual_path = os.path.join(residual_root_path, f"urban_detection_{sample_id}.mat")
            scipy.io.savemat(residual_path, {"detection": residual_np})

            background_path = os.path.join(background_root_path, f"urban_background_{sample_id}.mat")
            scipy.io.savemat(
                background_path,
                {"detection": format_background_for_savemat(final_artifacts["background_img"])},
            )
            score_mat_path = save_score_components(
                score_root_path,
                sample_id,
                {
                    "residual_score": final_artifacts["residual_score"],
                    "contrast_score": final_artifacts["contrast_score"],
                    "uncertainty_score": final_artifacts["uncertainty_score"],
                    "highfreq_score": final_artifacts["highfreq_score"],
                    "highfreq_alpha": final_artifacts["highfreq_alpha"],
                    "highfreq_alpha_map": final_artifacts["highfreq_alpha_map"],
                    "fused_score": final_artifacts["fused_score"],
                },
                weights=(residual_weight, contrast_weight, uncertainty_weight),
            )
            mask_history_path = save_mask_snapshots(mask_history_root_path, sample_id, mask_snapshot_records)

            history_csv_path, history_png_path = save_training_visualizations(
                visualization_dir,
                sample_id,
                mode_name,
                metric_history,
            )
            (
                detection_compare_path,
                detection_residual_path,
                detection_overlay_path,
                detection_zoom_path,
            ) = save_detection_visualization(
                visualization_dir,
                sample_id,
                mode_name,
                img_np,
                residual_np,
                label_np=label_np,
            )

            label_np_flat = label_np.reshape(-1)
            residual_np_flat = residual_np.reshape(-1)
            fpr, tpr, _ = roc_curve(label_np_flat, residual_np_flat)
            roc_auc = auc(fpr, tpr)
            elapsed_seconds = time.time() - start_time

            print(f"mode {mode_name} sample {sample_id} auc: {roc_auc}")
            print(f"mode {mode_name} sample {sample_id} elapsed_seconds: {elapsed_seconds}")
            print(f"Saved loss history CSV: {history_csv_path}")
            print(f"Saved loss curve PNG: {history_png_path}")
            print(f"Saved comparison PNG: {detection_compare_path}")
            print(f"Saved residual PNG: {detection_residual_path}")
            print(f"Saved overlay PNG: {detection_overlay_path}")
            print(f"Saved zoom PNG: {detection_zoom_path}")
            if final_artifacts.get("highfreq_alpha") is not None:
                diagnostics = final_artifacts.get("highfreq_diagnostics", {})
                print(
                    f"highfreq alpha: {float(final_artifacts['highfreq_alpha'][0]):.6f}; "
                    f"top_overlap: {diagnostics.get('top_overlap', 0.0):.6f}; "
                    f"rank_corr: {diagnostics.get('rank_corr', 0.0):.6f}; "
                    f"hf_top_conc: {diagnostics.get('hf_top_concentration', 0.0):.6f}; "
                    f"hf_entropy: {diagnostics.get('hf_entropy', 0.0):.6f}; "
                    f"hf_peak_ratio: {diagnostics.get('hf_to_base_peak_ratio', 0.0):.6f}; "
                    f"selected: {diagnostics.get('selected_fusion', highfreq_fusion_mode)}; "
                    f"texture_guard: {diagnostics.get('concentrated_texture_guard', 0.0):.0f}; "
                    f"dyn_range: {diagnostics.get('input_dynamic_range', 0.0):.6f}; "
                    f"edge_guard: {diagnostics.get('edge_guard_mean', 0.0):.6f}; "
                    f"alpha_min: {diagnostics.get('alpha_min', float(final_artifacts['highfreq_alpha'][0])):.6f}; "
                    f"alpha_max: {diagnostics.get('alpha_max', float(final_artifacts['highfreq_alpha'][0])):.6f}"
                )
            print(f"Saved score components MAT: {score_mat_path}")
            if mask_history_path is not None:
                print(f"Saved mask history MAT: {mask_history_path}")
            torch.cuda.empty_cache()
            return {
                "sample_id": str(sample_id),
                "roc_auc": float(roc_auc),
                "stop_iteration": int(j + 1),
                "elapsed_seconds": float(elapsed_seconds),
                "loss_final": float(lossiter[-1]),
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPMN ablation experiments.")
    parser.add_argument(
        "--ablation-mode",
        choices=sorted(ABLATION_MODES.keys()),
        default="mask_sam",
        help="Experiment mode, including baseline/mask_sam and the new prior_blindspot/full_innovation modes.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for experiment summaries.",
    )
    parser.add_argument(
        "--sam-weight",
        type=float,
        default=0.01,
        help="Weight of spectral angle loss when sam_loss is enabled.",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=None,
        help="Weight of reconstruction residual in the fused anomaly map.",
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=None,
        help="Weight of local spectral contrast in the fused anomaly map.",
    )
    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=None,
        help="Weight of abundance uncertainty in the fused anomaly map.",
    )
    parser.add_argument(
        "--highfreq-score-mode",
        choices=["none", "stationary_haar", "pywt"],
        default="none",
        help="Optional high-frequency residual score fused only into the final anomaly map.",
    )
    parser.add_argument(
        "--highfreq-weight",
        type=float,
        default=0.0,
        help="Alpha for final fusion: (1-alpha) * fused_score + alpha * highfreq_score.",
    )
    parser.add_argument(
        "--highfreq-fusion-mode",
        choices=["fixed", "adaptive", "soft_map", "diagnostic"],
        default="fixed",
        help="Use fixed alpha, sample-level adaptive alpha, or a pixel-level soft alpha map.",
    )
    parser.add_argument(
        "--highfreq-adaptive-low-alpha",
        type=float,
        default=0.1,
        help="Low alpha used by adaptive high-frequency fusion when agreement is weak.",
    )
    parser.add_argument(
        "--highfreq-adaptive-high-alpha",
        type=float,
        default=0.4,
        help="High alpha used by adaptive high-frequency fusion when agreement is strong.",
    )
    parser.add_argument(
        "--highfreq-adaptive-top-overlap",
        type=float,
        default=0.20,
        help="Top-5-percent overlap threshold for adaptive high-frequency fusion.",
    )
    parser.add_argument(
        "--highfreq-adaptive-rank-corr",
        type=float,
        default=0.15,
        help="Rank correlation threshold for adaptive high-frequency fusion.",
    )
    parser.add_argument(
        "--highfreq-adaptive-min-top-concentration",
        type=float,
        default=0.08,
        help="Minimum top-1-percent high-frequency score concentration for high adaptive alpha.",
    )
    parser.add_argument(
        "--highfreq-adaptive-max-entropy",
        type=float,
        default=0.85,
        help="Maximum normalized spatial entropy of high-frequency score for high adaptive alpha.",
    )
    parser.add_argument(
        "--highfreq-adaptive-min-peak-ratio",
        type=float,
        default=0.75,
        help="Minimum high-frequency/base top-1-percent sharpness ratio for high adaptive alpha.",
    )
    parser.add_argument(
        "--highfreq-adaptive-max-peak-ratio",
        type=float,
        default=3.0,
        help="Maximum high-frequency/base top-1-percent sharpness ratio for high adaptive alpha.",
    )
    parser.add_argument(
        "--highfreq-soft-map-top-quantile",
        type=float,
        default=0.85,
        help="Rank threshold used by soft-map fusion to identify jointly high base/high-frequency pixels.",
    )
    parser.add_argument(
        "--highfreq-soft-map-gate-slope",
        type=float,
        default=16.0,
        help="Sigmoid slope for soft-map high-response gates.",
    )
    parser.add_argument(
        "--highfreq-soft-map-diffuse-guard",
        type=float,
        default=0.5,
        help="Multiplier on soft-map high-alpha boost when sample-level diffuse-response guard fails.",
    )
    parser.add_argument(
        "--highfreq-diagnostic-dynamic-range",
        type=float,
        default=10.0,
        help="Input dynamic-range threshold that makes diagnostic high-frequency fusion choose conservative soft-map.",
    )
    parser.add_argument(
        "--highfreq-diagnostic-texture-entropy",
        type=float,
        default=0.80,
        help="Entropy threshold below which concentrated high-frequency texture response uses conservative soft-map.",
    )
    parser.add_argument(
        "--highfreq-diagnostic-texture-concentration",
        type=float,
        default=0.25,
        help="Top concentration threshold above which concentrated high-frequency texture response uses conservative soft-map.",
    )
    parser.add_argument(
        "--highfreq-diagnostic-fixed-alpha",
        type=float,
        default=None,
        help="Fixed alpha used by diagnostic fusion for non-protected samples. Defaults to highfreq adaptive high alpha.",
    )
    parser.add_argument(
        "--highfreq-diagnostic-soft-low-alpha",
        type=float,
        default=None,
        help="Low alpha used by diagnostic fusion's protected soft-map branch. Defaults to highfreq adaptive low alpha.",
    )
    parser.add_argument(
        "--highfreq-diagnostic-soft-high-alpha",
        type=float,
        default=None,
        help="High alpha used by diagnostic fusion's protected soft-map branch. Defaults to highfreq adaptive high alpha.",
    )
    parser.add_argument(
        "--highfreq-edge-guard",
        action="store_true",
        help="Suppress high-frequency alpha on strong input edges where base anomaly evidence is weak.",
    )
    parser.add_argument(
        "--highfreq-edge-guard-quantile",
        type=float,
        default=0.85,
        help="Rank threshold for input edge regions used by high-frequency edge guard.",
    )
    parser.add_argument(
        "--highfreq-edge-guard-strength",
        type=float,
        default=0.75,
        help="Strength of high-frequency alpha suppression on guarded edge regions.",
    )
    parser.add_argument(
        "--highfreq-wavelet",
        default="haar",
        help="PyWavelets wavelet name used when --highfreq-score-mode pywt is selected.",
    )
    parser.add_argument(
        "--highfreq-level",
        type=int,
        default=1,
        help="PyWavelets decomposition level used when --highfreq-score-mode pywt is selected.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print training metrics every N iterations instead of every iteration.",
    )
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Apply min-max normalization to image and row-wise norm normalization to A before training.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=["adaptive", "consensus", "sp_imp"],
        default=None,
        help="Mask prior strategy. Leave unset to use the selected ablation mode default.",
    )
    parser.add_argument(
        "--enable-blindspot",
        action="store_true",
        help="Enable blind-spot / guard-window self-supervised reconstruction loss.",
    )
    parser.add_argument(
        "--disable-blindspot",
        action="store_true",
        help="Disable blind-spot loss even if the selected ablation mode enables it.",
    )
    parser.add_argument(
        "--blindspot-ratio",
        type=float,
        default=0.25,
        help="Fraction of pixels sampled for blind-spot reconstruction after warmup.",
    )
    parser.add_argument(
        "--guard-window",
        type=int,
        default=5,
        help="Guard-window size expanded around sampled blind-spot pixels.",
    )
    parser.add_argument(
        "--low-rank-weight",
        type=float,
        default=0.0,
        help="Weight for the truncated low-rank background loss.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.0,
        help="Weight for sparse residual separation loss.",
    )
    parser.add_argument(
        "--low-rank-fraction",
        type=float,
        default=0.15,
        help="Fraction of singular values retained before penalizing the low-rank tail.",
    )
    parser.add_argument(
        "--superpixel-segments",
        type=int,
        default=256,
        help="Approximate number of superpixels for sp_imp_dpmn.",
    )
    parser.add_argument(
        "--superpixel-compactness",
        type=float,
        default=0.08,
        help="SLIC compactness used by sp_imp_dpmn when scikit-image is available.",
    )
    parser.add_argument(
        "--sp-target-weight",
        type=float,
        default=None,
        help="Blend weight for the superpixel pooled target in sp_imp_dpmn.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="Override per-sample optimization iterations; useful for smoke tests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of dataset files processed; useful for smoke tests.",
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated sample ids to run, for example: abu_beach_2,abu_urban_2.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(BASE_DIR, "dataset"),
        help="Directory containing trainable .mat files.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="urban",
        help="Prefix used when collecting trainable .mat files.",
    )
    parser.add_argument(
        "--import-mat-dir",
        default=None,
        help="Optional external directory containing ABU .mat files with data/map keys.",
    )
    parser.add_argument(
        "--import-pattern",
        default="abu-*.mat",
        help="Glob pattern used to locate source .mat files in --import-mat-dir.",
    )
    parser.add_argument(
        "--import-target-clusters",
        type=int,
        default=3,
        help="Fallback cluster budget used when importing files that do not contain raw A.",
    )
    parser.add_argument(
        "--import-background-clusters",
        type=int,
        default=5,
        help="Fallback cluster budget used when importing files that do not contain raw A.",
    )
    parser.add_argument(
        "--import-disable-normalize",
        action="store_true",
        help="Do not min-max normalize ABU cubes during import.",
    )
    parser.add_argument(
        "--import-overwrite",
        action="store_true",
        help="Overwrite already converted imported dataset files.",
    )
    parser.add_argument(
        "--import-random-state",
        type=int,
        default=0,
        help="Random seed for KMeans when importing ABU files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    batch_output_dir = args.results_dir or os.path.join(BASE_DIR, "results")
    visualization_dir = os.path.join(batch_output_dir, "training_curves")
    ensure_dir(dataset_dir)
    maybe_prepare_imported_dataset(args, dataset_dir)
    dataset_files = collect_dataset_files(dataset_dir, prefix=args.dataset_prefix)
    if args.sample_ids:
        requested_sample_ids = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        dataset_files = [
            file_path for file_path in dataset_files
            if extract_sample_id(file_path, prefix=args.dataset_prefix) in requested_sample_ids
        ]
        if len(dataset_files) != len(requested_sample_ids):
            found_sample_ids = {extract_sample_id(file_path, prefix=args.dataset_prefix) for file_path in dataset_files}
            missing_sample_ids = sorted(requested_sample_ids - found_sample_ids)
            raise ValueError(f"Requested sample id(s) not found: {missing_sample_ids}")
    if args.max_samples is not None:
        dataset_files = dataset_files[: max(0, int(args.max_samples))]
    mode_name = args.ablation_mode
    mode_config = dict(ABLATION_MODES[mode_name])
    mode_config["sam_weight"] = args.sam_weight
    if args.prior_mode is not None:
        mode_config["prior_mode"] = args.prior_mode
    else:
        mode_config.setdefault("prior_mode", "adaptive")
    if args.enable_blindspot:
        mode_config["blindspot"] = True
    if args.disable_blindspot:
        mode_config["blindspot"] = False
    mode_config["blindspot_ratio"] = args.blindspot_ratio
    mode_config["guard_window"] = args.guard_window
    default_low_rank_weight = 1e-4 if mode_config.get("low_rank_sparse", False) else 0.0
    default_sparse_weight = 1e-3 if mode_config.get("low_rank_sparse", False) else 0.0
    mode_config["low_rank_weight"] = args.low_rank_weight if args.low_rank_weight > 0.0 else default_low_rank_weight
    mode_config["sparse_weight"] = args.sparse_weight if args.sparse_weight > 0.0 else default_sparse_weight
    mode_config["low_rank_fraction"] = args.low_rank_fraction
    mode_config["superpixel_segments"] = args.superpixel_segments
    mode_config["superpixel_compactness"] = args.superpixel_compactness
    mode_config["sp_target_weight"] = min(
        1.0,
        max(0.0, args.sp_target_weight if args.sp_target_weight is not None else mode_config.get("sp_target_weight", 1.0)),
    )
    mode_config["num_iter"] = int(args.num_iter) if args.num_iter is not None else int(mode_config.get("num_iter", 1500))
    residual_weight = args.residual_weight if args.residual_weight is not None else mode_config.get("residual_weight", 0.6)
    contrast_weight = args.contrast_weight if args.contrast_weight is not None else mode_config.get("contrast_weight", 0.25)
    uncertainty_weight = args.uncertainty_weight if args.uncertainty_weight is not None else mode_config.get("uncertainty_weight", 0.15)
    weight_sum = residual_weight + contrast_weight + uncertainty_weight
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")
    mode_config["residual_weight"] = residual_weight / weight_sum
    mode_config["contrast_weight"] = contrast_weight / weight_sum
    mode_config["uncertainty_weight"] = uncertainty_weight / weight_sum
    mode_config["highfreq_score_mode"] = args.highfreq_score_mode
    mode_config["highfreq_weight"] = min(1.0, max(0.0, float(args.highfreq_weight)))
    mode_config["highfreq_fusion_mode"] = args.highfreq_fusion_mode
    mode_config["highfreq_adaptive_low_alpha"] = min(1.0, max(0.0, float(args.highfreq_adaptive_low_alpha)))
    mode_config["highfreq_adaptive_high_alpha"] = min(1.0, max(0.0, float(args.highfreq_adaptive_high_alpha)))
    mode_config["highfreq_adaptive_top_overlap"] = float(args.highfreq_adaptive_top_overlap)
    mode_config["highfreq_adaptive_rank_corr"] = float(args.highfreq_adaptive_rank_corr)
    mode_config["highfreq_adaptive_min_top_concentration"] = float(args.highfreq_adaptive_min_top_concentration)
    mode_config["highfreq_adaptive_max_entropy"] = float(args.highfreq_adaptive_max_entropy)
    mode_config["highfreq_adaptive_min_peak_ratio"] = float(args.highfreq_adaptive_min_peak_ratio)
    mode_config["highfreq_adaptive_max_peak_ratio"] = float(args.highfreq_adaptive_max_peak_ratio)
    mode_config["highfreq_soft_map_top_quantile"] = min(1.0, max(0.0, float(args.highfreq_soft_map_top_quantile)))
    mode_config["highfreq_soft_map_gate_slope"] = max(1e-6, float(args.highfreq_soft_map_gate_slope))
    mode_config["highfreq_soft_map_diffuse_guard"] = min(1.0, max(0.0, float(args.highfreq_soft_map_diffuse_guard)))
    mode_config["highfreq_diagnostic_dynamic_range"] = max(0.0, float(args.highfreq_diagnostic_dynamic_range))
    mode_config["highfreq_diagnostic_texture_entropy"] = float(args.highfreq_diagnostic_texture_entropy)
    mode_config["highfreq_diagnostic_texture_concentration"] = float(args.highfreq_diagnostic_texture_concentration)
    mode_config["highfreq_diagnostic_fixed_alpha"] = (
        None if args.highfreq_diagnostic_fixed_alpha is None else min(1.0, max(0.0, float(args.highfreq_diagnostic_fixed_alpha)))
    )
    mode_config["highfreq_diagnostic_soft_low_alpha"] = (
        None if args.highfreq_diagnostic_soft_low_alpha is None else min(1.0, max(0.0, float(args.highfreq_diagnostic_soft_low_alpha)))
    )
    mode_config["highfreq_diagnostic_soft_high_alpha"] = (
        None if args.highfreq_diagnostic_soft_high_alpha is None else min(1.0, max(0.0, float(args.highfreq_diagnostic_soft_high_alpha)))
    )
    mode_config["highfreq_edge_guard"] = bool(args.highfreq_edge_guard)
    mode_config["highfreq_edge_guard_quantile"] = min(1.0, max(0.0, float(args.highfreq_edge_guard_quantile)))
    mode_config["highfreq_edge_guard_strength"] = min(1.0, max(0.0, float(args.highfreq_edge_guard_strength)))
    mode_config["highfreq_wavelet"] = args.highfreq_wavelet
    mode_config["highfreq_level"] = max(1, int(args.highfreq_level))
    ensure_dir(batch_output_dir)
    ensure_dir(visualization_dir)

    print(f"Found {len(dataset_files)} dataset files.")
    print(f"Running ablation mode: {mode_name} with config: {mode_config}")
    results = []
    for file_path in dataset_files:
        sample_id = extract_sample_id(file_path, prefix=args.dataset_prefix)
        print(f"Running sample {sample_id}: {file_path}")
        results.append(
            run_single_sample(
                file_path,
                sample_id,
                mode_config,
                mode_name,
                visualization_dir,
                batch_output_dir,
                args.log_interval,
                args.normalize_inputs,
                num_iter_override=args.num_iter,
            )
        )

    csv_path, txt_path, mean_auc = save_run_summary(batch_output_dir, results, mode_name)
    print(f"Mode {mode_name} Mean AUC: {mean_auc}")
    print(f"Saved batch CSV: {csv_path}")
    print(f"Saved batch summary: {txt_path}")
    return results


if __name__ == "__main__":
    total_start = time.time()
    main()
    print(time.time() - total_start)
