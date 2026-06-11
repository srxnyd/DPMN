# 2025-Oriented Innovation Directions for DPMN

## 1. Background

The recent `prior_blindspot` experiment did not improve DPMN. The partial run was stopped after several samples because the early AUC results were already unstable:

```text
1:             0.915367
3:             0.356696
abu_airport_1: 0.689709
abu_airport_2: 0.765668
abu_airport_3: 0.206455
abu_airport_4: 0.977218
partial mean:  0.651852
```

This is far below previous strong results in the workspace, for example:

```text
resultold/batch_summary_baseline.txt      mean_auc: 0.949646
resultnew_all_v5/batch_summary_mask_sam.txt mean_auc: 0.940880
```

The conclusion is that the current blind-spot plus consensus-prior route is not reliable. The failure is not a small tuning issue. It suggests that DPMN needs a larger structural change that directly addresses anomaly leakage and identity mapping.

## 2. Core Diagnosis

DPMN is a reconstruction-based hyperspectral anomaly detection method. Its main risk is that the network can reconstruct both background and anomalies. When anomalies are reconstructed well, residual-based anomaly scores become weak.

The failed `prior_blindspot` run shows that simply adding random guard-window masking and heuristic consensus priors can destabilize specific samples, especially `sample 3` and `abu_airport_3`.

The next innovation should therefore focus on:

- preventing identity mapping,
- forcing the model to learn background structure instead of copying pixels,
- adding physically or statistically meaningful anomaly separation,
- keeping the current DPMN abundance/endmember structure where it is useful.

## 3. Recommended Large Innovation Directions

### 3.1 Anti-Identity / IMP-DPMN

Most recommended direction.

Relevant paper:

- *Overcoming the Identity Mapping Problem in Self-Supervised Hyperspectral Anomaly Detection*, 2025 arXiv.
- Link: <https://arxiv.org/abs/2504.04115>

The paper directly targets the identity mapping problem in self-supervised hyperspectral anomaly detection. Its useful ideas include:

- superpixel pooling and up-pooling perturbation,
- error-adaptive convolution,
- online background pixel mining.

This is highly compatible with DPMN because DPMN already reconstructs a background image through an abundance representation.

Proposed DPMN integration:

```text
Original HSI X
  -> superpixel pooling / region-level perturbation
  -> DPMN reconstructs region-level background manifold
  -> up-pooling to pixel space
  -> anomaly score from residual + abundance uncertainty
```

Expected benefit:

- prevents pixel-level copying,
- makes the network reconstruct background regions rather than individual anomalous pixels,
- directly addresses the failure mode observed in reconstruction-based HAD.

Possible method name:

```text
SP-IMP-DPMN
Anti-Identity DPMN
Superpixel Perturbation DPMN
```

Implementation sketch:

1. Add superpixel segmentation preprocessing for each HSI sample.
2. Pool pixels within each superpixel to create a region-level input.
3. Feed the perturbed/pooled representation to DPMN.
4. Up-pool reconstructed background to the original resolution.
5. Use residual, abundance entropy, and optional local contrast for scoring.
6. Add online background mining so high-confidence background pixels dominate training.

This should be the first direction to implement.

### 3.2 Diffusion Background Suppression + DPMN

Relevant paper:

- *Background Suppression Diffusion Model for Hyperspectral Anomaly Detection*.
- Code: <https://github.com/majitao-xd/BSDM-HAD>
- Paper: <https://arxiv.org/abs/2307.09861>

Although the arXiv version is older, the method is associated with a 2025 journal publication and represents a strong generative-prior direction.

Main idea:

```text
Use diffusion to model or suppress background distribution,
then improve anomaly contrast by reducing background response.
```

Lightweight DPMN integration:

```text
BSDM or diffusion branch -> background-suppressed prior S_diff
DPMN branch              -> reconstructed background B and residual R
Final score              -> R + abundance uncertainty + S_diff
```

Larger integration:

```text
DPMN produces background reconstruction B.
Diffusion denoiser acts as a plug-and-play background prior.
Every N iterations, B is corrected toward the learned background distribution.
```

Expected benefit:

- much stronger novelty,
- uses generative background modeling,
- can reduce anomaly leakage if diffusion is trained or guided toward background-only structure.

Risk:

- significantly larger engineering cost,
- likely needs careful training data preparation,
- may be expensive to run compared with the current DPMN.

This is a strong second-stage innovation after SP-IMP-DPMN is stable.

### 3.3 Frequency-Aware / Wavelet-DPMN

Relevant works:

- *Frequency Domain Mask Guided Diffusion Model for Hyperspectral Anomaly Detection*, IGARSS 2025.
- Reference page: <https://www.researchgate.net/publication/397979597_Frequency_Domain_Mask_Guided_Diffusion_Model_for_Hyperspectral_Anomaly_Detection>
- *Wave-MambaAD: Wavelet-driven State Space Model for Multi-class Unsupervised Anomaly Detection*, ICCV 2025.
- Link: <https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Wave-MambaAD_Wavelet-driven_State_Space_Model_for_Multi-class_Unsupervised_Anomaly_Detection_ICCV_2025_paper.html>

The useful idea is that background and anomaly signals behave differently in spatial-frequency or wavelet domains. Background tends to be smoother and more structured, while anomalies often appear as local high-frequency or irregular spectral-spatial deviations.

Proposed DPMN integration:

```text
X, B = DPMN reconstruction
R = X - B

score_spatial = residual norm
score_freq    = abnormal-frequency residual response
score_final   = weighted fusion of score_spatial and score_freq
```

Stronger model-level integration:

```text
Feature map
  -> wavelet decomposition: LL, LH, HL, HH
  -> LL branch uses Mamba/state-space modeling for global background
  -> high-frequency branches preserve local anomaly details
  -> inverse wavelet / fusion
  -> DPMN abundance reconstruction
```

This could replace or improve the current `model/MambaHSI.py` horizontal/vertical/diagonal scanning design.

Possible method name:

```text
Wave-DPMN
Frequency-Aware DPMN
Wavelet State-Space DPMN
```

Expected benefit:

- stronger technical novelty than another mask loss,
- naturally combines with the existing Mamba direction,
- may improve difficult local anomaly cases.

Risk:

- frequency scores can be dataset-sensitive,
- needs careful normalization to avoid amplifying noise.

### 3.4 Flow Density on DPMN Abundance

Related 2025 direction:

- IGARSS 2025 includes flow-based density estimation for hyperspectral anomaly detection.
- Proceedings TOC: <https://www.proceedings.com/content/083/083020webtoc.pdf>

DPMN already produces abundance maps. These abundance vectors are a compact physical representation of each pixel. Instead of relying only on reconstruction residuals, a density estimator can identify pixels whose abundance composition is unlikely under the learned background distribution.

Proposed DPMN integration:

```text
DPMN:
  X -> abundance Z -> reconstructed background B

Normalizing flow:
  estimate p(Z) for background abundance distribution

Anomaly score:
  residual_score + low_density_score
```

Expected benefit:

- catches anomalies that are reconstructed well but have unusual abundance combinations,
- uses the physical structure already present in DPMN,
- lighter than diffusion.

Risk:

- requires reliable background sample selection,
- flow may overfit on small images unless regularized.

This is a useful add-on after anti-identity reconstruction is fixed.

## 4. Priority Recommendation

Recommended order:

1. **SP-IMP-DPMN / Anti-Identity DPMN**
2. **Frequency-aware or Wavelet-DPMN**
3. **Flow density on abundance**
4. **Diffusion background suppression teacher**

The first priority should be SP-IMP-DPMN because it directly attacks the core failure mode observed in the current project: anomaly leakage through reconstruction.

## 5. Proposed Next Implementation Plan

### Stage 1: Build SP-IMP-DPMN

Core changes:

- add superpixel segmentation,
- pool HSI pixels by superpixel,
- train DPMN on perturbed region-level input,
- up-pool reconstruction to original resolution,
- add online background pixel mining,
- keep existing baseline modes unchanged.

Target command:

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn
```

Expected comparison targets:

```text
resultold/batch_summary_baseline.txt
resultnew_all_v5/batch_summary_mask_sam.txt
resultfinal_v3vis/batch_summary_mask_sam.txt
```

### Stage 2: Add Frequency/Wavelet Branch

Core changes:

- add wavelet decomposition module,
- apply state-space/Mamba modeling to low-frequency background,
- preserve high-frequency residual for anomaly scoring,
- add frequency residual score to final fusion.

Target command:

```bash
python train.py --ablation-mode wave_sp_imp_dpmn --results-dir results_wave_sp_imp_dpmn
```

### Stage 3: Add Abundance Flow Score

Core changes:

- collect background abundance vectors from mined background mask,
- train lightweight normalizing flow or Gaussian mixture baseline,
- add negative log-likelihood as an anomaly score.

Target command:

```bash
python train.py --ablation-mode sp_imp_flow --results-dir results_sp_imp_flow
```

## 6. What Not To Continue

The following route should not be prioritized:

```text
consensus prior + random blindspot + guard window
```

Reason:

- it already showed severe sample instability,
- it does not robustly solve identity mapping,
- it can damage anomaly ranking on hard samples,
- it is less novel than the 2025 anti-identity, diffusion, and frequency-domain directions.

## 7. Short Conclusion

The most valuable next move is a large structural change:

```text
DPMN should stop being a pixel-copying reconstruction model
and become a region/background-manifold reconstruction model.
```

The most practical and publishable route is:

```text
SP-IMP-DPMN
  + online background mining
  + optional wavelet/frequency anomaly branch
  + optional abundance density score
```

This direction is better aligned with recent 2025 work and with the actual failure observed in the current experiment.
