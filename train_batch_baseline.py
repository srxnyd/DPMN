from __future__ import print_function

import argparse
import csv
import glob
import os
import time

import numpy as np
import scipy.io
import torch
import torch.optim
from sklearn.metrics import roc_curve, auc

from model.AGM import AGM
from utils.inpainting_utils import *
from utils.traditonal import total_variation

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def format_background_for_savemat(background_img):
    background_img = np.asarray(background_img)
    if background_img.ndim == 4:
        background_img = np.squeeze(background_img, axis=0)
    if background_img.ndim == 3:
        return np.transpose(background_img, (1, 2, 0))
    if background_img.ndim == 2:
        return background_img
    raise ValueError("Unexpected background image shape: {}".format(background_img.shape))


def load_mat_file(file_path):
    try:
        import h5py
        with h5py.File(file_path, "r") as mat:
            return {key: np.array(mat[key]) for key in mat.keys()}
    except Exception:
        mat = scipy.io.loadmat(file_path)
        return {key: value for key, value in mat.items() if not key.startswith("__")}


def collect_dataset_files(dataset_dir, prefix="urban"):
    pattern = os.path.join(dataset_dir, "{}*.mat".format(prefix))
    dataset_files = []
    ignored_prefixes = (
        "{}_background".format(prefix),
        "{}_detection".format(prefix),
        "urban_background",
        "urban_detection",
    )
    for file_path in sorted(glob.glob(pattern)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if any(file_name.startswith(x) for x in ignored_prefixes):
            continue
        dataset_files.append(file_path)
    if not dataset_files:
        raise FileNotFoundError("No dataset files found for pattern: {}".format(pattern))
    return dataset_files


def extract_sample_id(file_path, prefix="urban"):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if file_name.startswith(prefix):
        suffix = file_name[len(prefix):].lstrip("_-")
        return suffix if suffix else file_name
    return file_name


def save_run_summary(output_dir, results):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "batch_results_baseline.csv")
    txt_path = os.path.join(output_dir, "batch_summary_baseline.txt")

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
        f.write("sample_count: {}\n".format(len(results)))
        f.write("mean_auc: {:.12f}\n".format(mean_auc))
        f.write("mean_stop_iteration: {:.2f}\n".format(mean_iter))
        f.write("mean_elapsed_seconds: {:.4f}\n".format(mean_time))
        f.write("\nper_sample_results:\n")
        for item in results:
            f.write(
                "{}: auc={:.12f}, stop_iteration={}, elapsed_seconds={:.4f}, loss_final={:.12f}\n".format(
                    item["sample_id"],
                    item["roc_auc"],
                    item["stop_iteration"],
                    item["elapsed_seconds"],
                    item["loss_final"],
                )
            )

    return csv_path, txt_path, mean_auc


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
        self.fc1 = torch.nn.Linear(input_depth, band)
        self.fc1.weight.data = e_torch.clone()
        self.conv = torch.nn.Conv2d(in_channels=band, out_channels=band, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
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


def run_single_sample(file_path, sample_id, log_interval):
    start_time = time.time()
    torch.cuda.empty_cache()

    residual_root_path = "./dataset/urban_detection"
    background_root_path = "./dataset/urban_background"
    ensure_dir(residual_root_path)
    ensure_dir(background_root_path)

    thres = 0.000001
    pad = "reflection"
    opt_over = "net"
    method = "2D"
    lr = 0.001
    num_iter = 3000
    param_noise = False
    reg_noise_std = 0
    lam = 0.001

    mat = load_mat_file(file_path)
    img_h5 = mat["image"]
    e = mat["A"]
    label = mat["mask"]

    label_np = np.array(label).transpose(1, 0)
    e_np = np.array(e).transpose(1, 0)
    img_np = np.array(img_h5).transpose(0, 2, 1)

    e_torch = torch.from_numpy(e_np).type(dtype)
    img_var = torch.from_numpy(img_np).type(dtype)

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

    mse = torch.nn.MSELoss().type(dtype)
    img_var = img_var[None, :].cuda()
    e_torch = e_torch.cuda()

    mask_var = torch.ones(1, band, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    lossiter = []
    loss_last = 0
    end_iter = False

    def closure1(iter_num, mask_varr, residual_varr):
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        local_net_input = net_input_saved
        if reg_noise_std > 0:
            local_net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out, out_h = net(local_net_input)
        out_h_np = out_h.detach().cpu().squeeze(0).numpy()

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 50 == 0 and iter_num != 0:
            img_var_clone = img_var.detach().clone()
            net_output_clone = out_h.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :])
            residual_img = temp.sum(0)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            residual_img = r_max - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):
                mask_var_clone[0, i, :] = residual_img[:]

        b, channels, height, width = out.shape
        loss1 = mse(out_h * mask_var_clone, img_var * mask_var_clone)
        loss2 = total_variation(out.view(b * channels, height, width))
        total_loss = loss1 + lam * loss2
        total_loss.backward()

        metrics = {
            "total_loss": float(total_loss.item()),
            "mse_loss": float(loss1.item()),
            "tv_loss": float(loss2.item()),
        }
        return mask_var_clone, residual_var_clone, out_h_np, total_loss, metrics

    p = get_params(opt_over, net, net_input)
    print("Starting optimization with ADAM for sample {}".format(sample_id))
    optimizer = torch.optim.Adam(p, lr=lr)

    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, total_loss, metrics = closure1(j, mask_var, residual_varr)
        optimizer.step()
        lossiter.append(total_loss.item())

        if ((j + 1) % log_interval == 0) or j == 0 or j == num_iter - 1:
            print(
                "sample {}; iteration: {}; loss: {:.6f}; mse: {:.6f}; tv: {:.6f}".format(
                    sample_id,
                    j + 1,
                    metrics["total_loss"],
                    metrics["mse_loss"],
                    metrics["tv_loss"],
                )
            )

        if j >= 1:
            index = j - int(j / 50) * 50
            loss_np[0][index - 1] = abs(total_loss.item() - loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = total_loss.item()

        if j == num_iter - 1 or end_iter:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()
            residual_path = os.path.join(residual_root_path, "urban_detection_{}.mat".format(sample_id))
            scipy.io.savemat(residual_path, {"detection": residual_np})

            background_path = os.path.join(background_root_path, "urban_background_{}.mat".format(sample_id))
            scipy.io.savemat(background_path, {"detection": format_background_for_savemat(background_img)})

            label_np_flat = label_np.reshape(-1)
            residual_np_flat = residual_np.reshape(-1)
            fpr, tpr, _ = roc_curve(label_np_flat, residual_np_flat)
            roc_auc = auc(fpr, tpr)
            elapsed_seconds = time.time() - start_time

            print("sample {} auc: {}".format(sample_id, roc_auc))
            print("sample {} elapsed_seconds: {}".format(sample_id, elapsed_seconds))
            torch.cuda.empty_cache()
            return {
                "sample_id": str(sample_id),
                "roc_auc": float(roc_auc),
                "stop_iteration": int(j + 1),
                "elapsed_seconds": float(elapsed_seconds),
                "loss_final": float(lossiter[-1]),
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPMNold baseline in batch mode.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(BASE_DIR, "dataset"),
        help="Directory containing trainable .mat files.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="urban",
        help="Prefix used when collecting .mat files.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(BASE_DIR, "results_baseline"),
        help="Directory for batch summaries.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print metrics every N iterations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_files = collect_dataset_files(args.dataset_dir, prefix=args.dataset_prefix)
    ensure_dir(args.results_dir)

    print("Found {} dataset files.".format(len(dataset_files)))
    results = []
    for file_path in dataset_files:
        sample_id = extract_sample_id(file_path, prefix=args.dataset_prefix)
        print("Running sample {}: {}".format(sample_id, file_path))
        results.append(run_single_sample(file_path, sample_id, args.log_interval))

    csv_path, txt_path, mean_auc = save_run_summary(args.results_dir, results)
    print("Baseline Mean AUC: {}".format(mean_auc))
    print("Saved batch CSV: {}".format(csv_path))
    print("Saved batch summary: {}".format(txt_path))


if __name__ == "__main__":
    total_start = time.time()
    main()
    print(time.time() - total_start)
