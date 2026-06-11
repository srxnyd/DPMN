# DPMN Innovation Plan

## 1. Current Method Summary

The current project is a DIP-style unsupervised hyperspectral anomaly detection method. A random noise input is fed into an AGM encoder-decoder network to generate an abundance map. The abundance map is then projected by the endmember matrix `A` to reconstruct the hyperspectral background. The anomaly map is mainly derived from the reconstruction residual.

Existing implemented components:

- AGM/DIP reconstruction backbone in `train.py` and `model/AGM.py`.
- Mamba-enhanced feature modeling in `model/MambaHSI.py`, including vertical, horizontal, and diagonal scan branches.
- Adaptive training mask based on reconstruction error and spatial context.
- Mask-guided spectral angle loss.
- Final anomaly score fusion using reconstruction residual, local spectral contrast, and abundance uncertainty.

A key observation from existing ablation results is that the enhanced `mask_sam` setting is not always better than the baseline. This suggests the main weakness is not simply model capacity, but unstable masking and anomaly leakage during reconstruction.

## 2. Chosen Innovation Route

The selected first-stage route is:

1. Blind-Spot / Guard-Window self-supervised reconstruction.
2. Consensus anomaly-prior curriculum masking.

The selected second-stage route is:

3. Low-rank sparse DPMN regularization.

This route directly targets the main failure mode of reconstruction-based HAD: anomalies can be reconstructed together with the background, causing low residual scores and unstable detection.

## 3. Implemented Changes

All implemented code changes are in `train.py`.

### 3.1 New Experiment Modes

Added two new ablation modes:

- `prior_blindspot`: enables consensus prior masking and blind-spot reconstruction.
- `full_innovation`: enables consensus prior masking, blind-spot reconstruction, SAM loss, and low-rank sparse regularization.

The original modes are preserved:

- `baseline`
- `mask_only`
- `sam_only`
- `mask_sam`

This keeps old experiments reproducible.

### 3.2 Consensus Anomaly Prior

Implemented a consensus prior that fuses four unsupervised anomaly cues:

- reconstruction residual score,
- local spectral contrast score,
- abundance entropy / uncertainty score,
- RX-style Mahalanobis anomaly score.

Each cue is rank-normalized before fusion to reduce scale sensitivity. The prior is then smoothed and converted into a background training mask by a curriculum threshold. Early training relies more on RX and local contrast; later training increases the weight of reconstruction residual.

Main functions:

- `rank_normalize_score`
- `compute_rx_score`
- `compute_consensus_anomaly_prior`

### 3.3 Blind-Spot / Guard-Window Reconstruction

Implemented a blind-spot training weight. After warmup, a random subset of pixels is sampled and expanded by a guard window. The reconstruction loss is computed on this sampled region multiplied by the current background mask.

This is intended to prevent the network from simply copying anomalous pixels into the reconstructed background.

Main function:

- `make_blindspot_weight`

Important parameters:

- `--enable-blindspot`
- `--disable-blindspot`
- `--blindspot-ratio`
- `--guard-window`

### 3.4 Low-Rank Sparse DPMN Regularization

Implemented optional second-stage regularization:

- truncated low-rank background loss on the reconstructed background,
- sparse residual concentration loss between reconstruction and input.

These are disabled by default for old modes. In `full_innovation`, default weights are:

- `low_rank_weight = 1e-4`
- `sparse_weight = 1e-3`

Main functions:

- `low_rank_background_loss`
- `sparse_residual_loss`

Important parameters:

- `--low-rank-weight`
- `--sparse-weight`
- `--low-rank-fraction`

### 3.5 Extended Training Logs

The loss history CSV now records:

- `low_rank_loss`
- `sparse_loss`
- `blindspot_active`

Training console logs also print these values.

## 4. Recommended Experiments

### 4.1 First-Stage Main Experiment

Run the recommended first-stage method:

```bash
python train.py --ablation-mode prior_blindspot --results-dir results_prior_blindspot
```

Compare with:

```bash
python train.py --ablation-mode baseline --results-dir results_compare_baseline
python train.py --ablation-mode mask_sam --results-dir results_compare_mask_sam
```

Primary metrics:

- mean AUC,
- per-sample AUC,
- low-performing samples such as airport and difficult urban scenes,
- mask history stability.

### 4.2 Component Ablation

Consensus prior only:

```bash
python train.py --ablation-mode mask_only --prior-mode consensus --results-dir results_consensus_only
```

Blind-spot only on existing adaptive mask:

```bash
python train.py --ablation-mode mask_only --enable-blindspot --results-dir results_blindspot_only
```

Consensus prior plus blind-spot:

```bash
python train.py --ablation-mode prior_blindspot --results-dir results_prior_blindspot
```

### 4.3 Second-Stage Low-Rank Sparse Experiment

Run the full innovation route:

```bash
python train.py --ablation-mode full_innovation --results-dir results_full_innovation
```

Tune low-rank and sparse weights if needed:

```bash
python train.py --ablation-mode full_innovation \
  --low-rank-weight 0.00005 \
  --sparse-weight 0.0005 \
  --results-dir results_full_innovation_tuned
```

## 5. Expected Benefits

Expected gains from `prior_blindspot`:

- reduced anomaly leakage into reconstructed background,
- more stable training mask than residual-only adaptive masking,
- better performance on difficult samples where `mask_sam` previously failed.

Expected gains from `full_innovation`:

- stronger physical/model-driven explanation,
- better background compactness through low-rank regularization,
- anomaly residuals that are more spatially sparse and easier to separate.

## 6. Verification Already Done

The following checks were run successfully:

```bash
python -m py_compile train.py
python train.py --help
```

Full training was not run yet because the existing full 15-sample batch is time-consuming.

## 7. Next Practical Step

The next recommended step is to run:

```bash
python train.py --ablation-mode prior_blindspot --results-dir results_prior_blindspot
```

Then compare its `batch_summary_prior_blindspot.txt` against the existing `results_final_ablation_no_leak/batch_summary_baseline.txt` and `results_final_ablation_no_leak/batch_summary_mask_sam.txt`.
