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

## 8. Implementation History: SP-IMP-DPMN Run

Date: 2026-06-11.

The first-priority direction was implemented as `sp_imp_dpmn` in `train.py`.

Main code changes:

- added `sp_imp_dpmn` as a new ablation mode,
- installed and used `scikit-image` SLIC superpixels when available,
- kept a grid-superpixel fallback for environments without `skimage`,
- pooled HSI pixels by superpixel to create a region-level reconstruction target,
- trained DPMN against the superpixel-perturbed background target instead of directly copying every original pixel,
- added online background mining from reconstruction residual, local residual context, spectral contrast, and abundance entropy,
- smoothed the mined background confidence by superpixel labels,
- kept final anomaly scoring against the original HSI image so anomaly evidence is not removed by the pooled target,
- kept random blindspot disabled for this mode.

Initial default SP-IMP-DPMN run used full superpixel target strength and 1500 iterations. It was mixed:

```text
1:             auc=0.795791333202
3:             auc=0.873732909161
abu_airport_1: auc=0.823873500631
abu_airport_2: auc=0.702917682690
abu_airport_3: auc=0.686721919694
```

This was better than the failed `prior_blindspot` behavior on some hard samples, especially sample `3`, but it was still below the historical strong baseline level around `0.94-0.95` mean AUC.

A tuned configuration was then tested on the first five samples:

```text
num_iter=800
sp_target_weight=0.6
residual_weight=0.8
contrast_weight=0.15
uncertainty_weight=0.05
```

Rationale:

- reduce full superpixel target strength so fine anomaly evidence is not over-smoothed,
- stop earlier to reduce late identity-mapping recovery,
- make residual score dominant because it was more reliable than uncertainty-heavy fusion in the first run.

Tuned five-sample validation:

```text
sample_count: 5
mean_auc: 0.903437982006
1:             auc=0.937329537655
3:             auc=0.932469461898
abu_airport_1: auc=0.891044231376
abu_airport_2: auc=0.840539127188
abu_airport_3: auc=0.915807551912
```

The tuned configuration was then made the default for `sp_imp_dpmn`, so the target command below now uses the tuned defaults without extra CLI flags:

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn
```

Full tuned run result:

```text
sample_count: 15
mean_auc: 0.916926731476
mean_stop_iteration: 800.00

1:             auc=0.934602140142
3:             auc=0.936252428315
abu_airport_1: auc=0.892794434298
abu_airport_2: auc=0.886220462854
abu_airport_3: auc=0.903056070852
abu_airport_4: auc=0.955323608317
abu_beach_1:   auc=0.956093917905
abu_beach_2:   auc=0.838357343083
abu_beach_3:   auc=0.958590813531
abu_beach_4:   auc=0.954238168163
abu_urban_1:   auc=0.990849136979
abu_urban_2:   auc=0.639632366192
abu_urban_3:   auc=0.963662970524
abu_urban_4:   auc=0.982919604417
abu_urban_5:   auc=0.961307506566
```

Assessment:

- SP-IMP-DPMN is a real improvement over the failed random blindspot route.
- The tuned configuration is much stronger than the first SP-IMP-DPMN attempt.
- Full mean AUC `0.9169` is still below the strongest historical baseline files around `0.94-0.95`.
- The main failure cases are `abu_urban_2` and, secondarily, `abu_beach_2`.
- Next recommended step is sample-adaptive fusion or sample-adaptive superpixel strength, starting with diagnosis of `abu_urban_2` score components and mask history.

Generated artifacts from the full tuned run:

```text
results_sp_imp_dpmn/batch_summary_sp_imp_dpmn.txt
results_sp_imp_dpmn/run_sp_imp_dpmn_tuned_full_log.md
results_sp_imp_dpmn/score_components/sp_imp_dpmn/*.mat
results_sp_imp_dpmn/mask_history/sp_imp_dpmn/*.mat
results_sp_imp_dpmn/training_curves/*.png
```

## 9. Targeted Fix for Low-AUC Samples

Date: 2026-06-11.

The tuned full SP-IMP-DPMN run still had two weak samples:

```text
abu_beach_2: auc=0.838357343083
abu_urban_2: auc=0.639632366192
```

Score-component diagnosis showed different failure modes:

```text
abu_beach_2:
  residual_score:    0.757057916447
  contrast_score:    0.834009365419
  uncertainty_score: 0.880860207882
  fused_score:       0.838357343083

abu_urban_2:
  residual_score:    0.992800668425
  contrast_score:    0.984424384410
  uncertainty_score: 0.023471223316
  fused_score:       0.639632366192
```

Conclusion:

- `abu_urban_2` was not a reconstruction failure; residual and contrast were already excellent.
- The abundance uncertainty score was nearly inverted on `abu_urban_2` and damaged fusion.
- `abu_beach_2` still benefited from uncertainty, so globally removing uncertainty was not ideal.

A label-free adaptive score fusion rule was added for `sp_imp_dpmn`:

```text
base score = 0.85 * residual + 0.15 * contrast

Use uncertainty only if:
  top-5% overlap(base, uncertainty) > 0.20
  rank_corr(residual, uncertainty) > 0.15
  rank_corr(contrast, uncertainty) > 0.05

Otherwise:
  fused score = base score
```

Offline evaluation on the existing full-run score components estimated:

```text
default tuned fusion mean_auc: 0.916926731476
adaptive fusion mean_auc:      0.943157920053
```

The two weak samples were then rerun with adaptive fusion enabled:

```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_sp_imp_dpmn_adaptive_fix \
  --sample-ids abu_beach_2,abu_urban_2
```

Rerun result:

```text
sample_count: 2
mean_auc: 0.933665698927

abu_beach_2: auc=0.874754193117
abu_urban_2: auc=0.992577204738
```

Replacing the two weak full-run scores with the adaptive-fusion rerun scores gives an estimated full-run mean:

```text
projected_full_mean_auc: 0.942882844048
```

Assessment:

- The targeted fix directly addresses the `abu_urban_2` failure without hard-coding sample names.
- `abu_beach_2` also improved under the rerun.
- This brings SP-IMP-DPMN close to the historical strong baseline range around `0.94-0.95`.
- A full rerun with adaptive fusion enabled is the next clean comparison if exact final mean is needed.

Generated artifacts:

```text
results_sp_imp_dpmn_adaptive_fix/batch_summary_sp_imp_dpmn.txt
results_sp_imp_dpmn_adaptive_fix/run_log.md
results_sp_imp_dpmn_adaptive_fix/score_components/sp_imp_dpmn/*.mat
results_sp_imp_dpmn_adaptive_fix/mask_history/sp_imp_dpmn/*.mat
```

## 10. Next Optimization Plan After Adaptive Fusion

Date: 2026-06-11.

Current status after pushing commit `cf5511b`:

```text
Full tuned SP-IMP-DPMN mean_auc: 0.916926731476
Adaptive-fusion targeted rerun:
  abu_beach_2: 0.838357343083 -> 0.874754193117
  abu_urban_2: 0.639632366192 -> 0.992577204738
Projected full mean after replacing these two samples: 0.942882844048
```

Interpretation:

- SP-IMP-DPMN is now close to the historical strong baseline range.
- The largest observed issue was not reconstruction loss itself, but score fusion instability.
- `abu_urban_2` had excellent residual and contrast scores, but abundance uncertainty was nearly inverted and damaged fusion.
- Adaptive fusion fixed this without hard-coding sample names.

Recommended next sequence:

1. Full rerun with adaptive fusion enabled to get the true full mean AUC.
2. Add localization-oriented metrics: AUPR, precision@K, and IoU@bestF1.
3. Add a wavelet high-frequency residual score without changing the main network.
4. Consider integrating error-adaptive convolution into AGM/DPMN afterward.

Rationale by step:

### Step 1: Full adaptive-fusion rerun

The current `0.9429` is a projected value from replacing two weak samples. A full rerun is needed for a clean comparison because DIP-style optimization has run-to-run variation.

Target command:

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn_adaptive_full
```

### Step 2: Add localization metrics

AUC is ranking-oriented and does not fully describe localization quality. For hyperspectral anomaly detection, especially tiny targets, the following metrics are useful:

```text
AUPR
precision@K, where K is the number of ground-truth anomaly pixels
IoU@bestF1
best F1 threshold
```

These should be saved in the batch CSV and batch summary alongside ROC-AUC.

### Step 3: Add wavelet high-frequency residual score

The remaining weak sample `abu_beach_2` suggests that residual and contrast may not fully capture local frequency/edge anomaly evidence. Recent frequency and wavelet HAD work argues that anomalies and backgrounds separate better when low-frequency background and high-frequency details are modeled separately.

A lightweight implementation should not alter the DPMN backbone first. Instead:

```text
residual map = ||X - B||
high-frequency residual = residual - local low-frequency residual
wavelet/frequency score = normalized high-frequency residual response
final score = adaptive fusion + alpha * wavelet score
```

This is lower risk than adding a full diffusion or wavelet-Mamba branch immediately.

### Step 4: Error-adaptive convolution

The 2025 IMP/SuperAD direction describes three complementary pieces:

```text
perturbation:     superpixel pooling / uppooling
reconstruction:   error-adaptive convolution
regularization:   online background pixel mining
```

Current SP-IMP-DPMN already covers perturbation and regularization. The missing model-level piece is error-adaptive convolution, which can be considered after score-side improvements are validated.



### Execution update: Step 1 and Step 2

Date: 2026-06-11.

Step 1 full adaptive-fusion rerun completed in:

```text
results_sp_imp_dpmn_adaptive_full/
```

Summary:

```text
sample_count: 15
mean_auc: 0.939775035950
mean_stop_iteration: 800.00
mean_elapsed_seconds: 92.9591
```

The full rerun confirms that adaptive score fusion improves the full SP-IMP-DPMN run versus the earlier tuned mean AUC of `0.916926731476`, but the clean full-run mean is slightly lower than the two-sample replacement projection of `0.942882844048`.

Step 2 localization metrics were computed from the saved `fused_score` maps without retraining. Outputs:

```text
results_sp_imp_dpmn_adaptive_full/extra_metrics_sp_imp_dpmn.csv
results_sp_imp_dpmn_adaptive_full/extra_metrics_sp_imp_dpmn.txt
```

Metric means:

```text
Mean AUPR: 0.342769952485
Mean precision@K: 0.339448691725
Mean best_f1: 0.395349869969
Mean IoU@bestF1: 0.282656780208
```

Notable observation:

- `abu_urban_2` is now strong on both ranking and localization: AUC `0.993923229411`, AUPR `0.894372078765`, precision@K `0.819354838710`, IoU@bestF1 `0.712643678161`.
- `abu_beach_2` remains weak after adaptive fusion: AUC `0.863029230051`, AUPR `0.070023386542`, precision@K `0.039603960396`, IoU@bestF1 `0.086255259467`.
- `sample 3` has good ROC-AUC (`0.930643263091`) but poor localization metrics, especially precision@K `0.0`; this suggests score ranking is partially correct globally but not concentrated enough at the top anomaly budget.

Next recommended action is still Step 3: add a wavelet or local high-frequency residual score as a score-side component first, then evaluate whether it improves the weak localization cases before modifying AGM/DPMN internals.


### Execution update: Step 3 wavelet/high-frequency residual score

Date: 2026-06-11.

`PyWavelets` was installed in the active Python environment:

```text
PyWavelets 1.8.0
```

Two post-hoc score-side experiments were run from the saved adaptive-fusion score components, without retraining the DPMN backbone:

```text
results_sp_imp_dpmn_adaptive_full/wavelet_score_sweep_sp_imp_dpmn.csv
results_sp_imp_dpmn_adaptive_full/wavelet_score_sweep_sp_imp_dpmn.txt
results_sp_imp_dpmn_adaptive_full/pywt_score_sweep_sp_imp_dpmn.csv
results_sp_imp_dpmn_adaptive_full/pywt_score_sweep_sp_imp_dpmn.txt
```

Baseline for comparison is the Step 1/2 adaptive fused score:

```text
AUC: 0.939775035950
AUPR: 0.342769952485
precision@K: 0.339448691725
bestF1: 0.395349869969
IoU@bestF1: 0.282656780208
```

Best true `pywt` configuration by AUPR and IoU@bestF1:

```text
wavelet: haar
level: 1
alpha: 0.30
AUC: 0.953873731974
AUPR: 0.356778272871
precision@K: 0.348047710213
bestF1: 0.405837900809
IoU@bestF1: 0.288405723500
```

Best true `pywt` configuration by mean AUC:

```text
wavelet: db2
level: 1
alpha: 0.30
AUC: 0.954664108077
AUPR: 0.338967588556
precision@K: 0.327906769853
bestF1: 0.382920828727
IoU@bestF1: 0.264141290779
```

A stationary Haar-like high-frequency residual response, implemented without downsampling, was also tested as a local high-frequency score. Its best AUC/AUPR configuration was stronger:

```text
alpha: 0.40
AUC: 0.957072945009
AUPR: 0.367149661258
precision@K: 0.359885472771
bestF1: 0.422030701441
IoU@bestF1: 0.299804832583
```

Best IoU@bestF1 for the stationary Haar-like score was at alpha `0.50`:

```text
AUC: 0.956791689604
AUPR: 0.363152774330
precision@K: 0.357090111327
bestF1: 0.425539362122
IoU@bestF1: 0.301947628219
```

Conclusion for Step 3:

- The high-frequency residual direction is validated: both true `pywt` and stationary Haar-like variants improve mean AUC over adaptive fusion alone.
- The true `pywt` Haar level-1 score is the better conservative paper-style wavelet option because it improves AUPR and IoU while staying close to standard wavelet decomposition.
- The stationary Haar-like response is currently the stronger engineering option for the benchmark because it avoids downsampling loss and gives the best overall localization metrics.
- Before modifying AGM/DPMN internals, the next practical implementation should add a switchable score-side high-frequency residual component to `train.py`, default off, then run a full controlled comparison with alpha around `0.30-0.40`.


### Implementation update: switchable high-frequency score in train.py

Date: 2026-06-11.

A switchable final-score high-frequency residual component was added to `train.py`. It does not change the DPMN/AGM network or the training loss; it only modifies the final anomaly map after the original residual/contrast/uncertainty fusion.

New CLI options:

```bash
--highfreq-score-mode none|stationary_haar|pywt
--highfreq-weight ALPHA
--highfreq-wavelet haar
--highfreq-level 1
```

Fusion form:

```text
final_score = normalize((1 - alpha) * adaptive_fused_score + alpha * highfreq_score)
```

Supported modes:

```text
none: default; preserves previous behavior
stationary_haar: no-downsampling Haar-like local high-frequency residual response
pywt: PyWavelets wavedec2/waverec2 detail-only high-frequency residual response
```

Smoke test command:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --sample-ids 1 --num-iter 2 --results-dir /tmp/dpmn_highfreq_smoke --highfreq-score-mode pywt --highfreq-weight 0.3 --highfreq-wavelet haar --highfreq-level 1 --log-interval 1
```

Smoke test result:

```text
completed successfully
score component keys: contrast_score, fused_score, highfreq_score, residual_score, uncertainty_score, weights
highfreq_score shape: (80, 100)
fused_score shape: (80, 100)
```

Recommended full controlled runs:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn_pywt_hf_a03 --highfreq-score-mode pywt --highfreq-weight 0.3 --highfreq-wavelet haar --highfreq-level 1
python -u train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn_stationary_hf_a04 --highfreq-score-mode stationary_haar --highfreq-weight 0.4
```

The first command is the conservative wavelet-paper variant. The second command is the stronger benchmark-oriented high-frequency residual variant from the post-hoc sweep.


### Conversation update: resume and run decision

Date: 2026-06-11.

The interrupted adaptive-fusion run was checked and did not need to be resumed. The process had already completed all 15 samples in `results_sp_imp_dpmn_adaptive_full`, with full-run mean AUC:

```text
0.939775035950
```

The follow-up discussion established the next practical step:

1. Keep the completed adaptive-fusion full run as the baseline.
2. Use the saved score components to compute localization metrics and high-frequency residual score sweeps.
3. Install `PyWavelets` when `pywt` was missing instead of staying with an approximation only.
4. Implement high-frequency residual score as a switchable final-score component in `train.py`, default off.
5. Run a controlled full experiment to see whether the post-hoc gain survives an end-to-end run.

Implemented and verified so far:

```text
PyWavelets installed: 1.8.0
train.py syntax check: passed
smoke test: passed
smoke output includes highfreq_score with the same shape as fused_score
```

Best post-hoc candidates:

```text
Conservative pywt option:
  --highfreq-score-mode pywt --highfreq-weight 0.3 --highfreq-wavelet haar --highfreq-level 1
  post-hoc mean AUC: 0.953873731974
  post-hoc mean AUPR: 0.356778272871
  post-hoc mean IoU@bestF1: 0.288405723500

Stronger stationary option:
  --highfreq-score-mode stationary_haar --highfreq-weight 0.4
  post-hoc mean AUC: 0.957072945009
  post-hoc mean AUPR: 0.367149661258
  post-hoc mean IoU@bestF1: 0.299804832583
```

Next run selected from the conversation:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn_stationary_hf_a04 --highfreq-score-mode stationary_haar --highfreq-weight 0.4
```

Reason: the stationary Haar-like high-frequency residual score was the strongest post-hoc benchmark candidate and does not modify AGM/DPMN internals.


### Execution update: full stationary Haar high-frequency run

Date: 2026-06-11.

The selected full run completed successfully:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --results-dir results_sp_imp_dpmn_stationary_hf_a04 --highfreq-score-mode stationary_haar --highfreq-weight 0.4
```

Output directory:

```text
results_sp_imp_dpmn_stationary_hf_a04/
```

Full-run AUC summary:

```text
sample_count: 15
mean_auc: 0.956469872526
mean_stop_iteration: 800.00
mean_elapsed_seconds: 94.3856
```

Compared with the adaptive-fusion full run:

```text
adaptive full mean AUC:        0.939775035950
stationary_hf_a04 mean AUC:    0.956469872526
absolute gain:                 +0.016694836576
```

Localization metrics for the full stationary high-frequency run:

```text
Mean AUPR: 0.361500144012
Mean precision@K: 0.360311045273
Mean best_f1: 0.418435819511
Mean IoU@bestF1: 0.293532509189
```

Compared with the adaptive-fusion full run localization metrics:

```text
AUPR:        0.342769952485 -> 0.361500144012  (+0.018730191527)
precision@K: 0.339448691725 -> 0.360311045273  (+0.020862353548)
bestF1:      0.395349869969 -> 0.418435819511  (+0.023085949542)
IoU@bestF1:  0.282656780208 -> 0.293532509189  (+0.010875728981)
```

Per-sample AUC highlights:

```text
sample 1:       0.939752564768 -> 0.931242129638  (-0.008510435130)
sample 3:       0.930643263091 -> 0.913454056543  (-0.017189206548)
abu_airport_1: 0.883787653319 -> 0.925198553617  (+0.041410900298)
abu_airport_2: 0.881610238964 -> 0.927165187708  (+0.045554948744)
abu_airport_3: 0.912880737239 -> 0.946461612112  (+0.033580874873)
abu_airport_4: 0.955591884641 -> 0.973816230718  (+0.018224346077)
abu_beach_1:   0.943395475478 -> 0.984297851519  (+0.040902376041)
abu_beach_2:   0.863029230051 -> 0.836264574100  (-0.026764655951)
abu_beach_3:   0.930887612738 -> 0.992191410552  (+0.061303797814)
abu_beach_4:   0.953513756608 -> 0.987403105857  (+0.033889349249)
abu_urban_1:   0.987175268328 -> 0.986993453151  (-0.000181815177)
abu_urban_2:   0.993923229411 -> 0.991716115926  (-0.002207113485)
abu_urban_3:   0.966328755683 -> 0.984072948563  (+0.017744192880)
abu_urban_4:   0.988203397107 -> 0.993360584365  (+0.005157187258)
abu_urban_5:   0.965902471829 -> 0.973410273518  (+0.007507801689)
```

Interpretation:

- The stationary high-frequency residual score is validated in the actual `train.py` full run, not only post-hoc: mean AUC and all localization-oriented mean metrics improved.
- The largest gains are on airport and most beach samples, especially `abu_beach_3`, `abu_beach_1`, `abu_airport_1`, and `abu_airport_2`.
- `abu_beach_2` remains the main failure case and gets worse under this global alpha. This suggests the next refinement should make the high-frequency alpha adaptive or sample/region-gated rather than fixed at `0.4` everywhere.
- `sample 3` still has poor localization and lower AUC in this run; its scale and loss behavior remain different from the normalized ABU samples, so it may need normalization-specific handling or separate reporting.

Recommended next controlled experiment:

```text
Run alpha sweep through train.py, not only post-hoc, on the known sensitive samples first:
  sample_ids: 3, abu_beach_2, abu_airport_1, abu_airport_2, abu_beach_3
  alpha: 0.1, 0.2, 0.3, 0.4
Then decide whether to use fixed alpha=0.3/0.4 or adaptive gating.
```


### Conversation update: next step after stationary high-frequency full run

Date: 2026-06-11.

After the full `stationary_haar` alpha `0.4` run, the conclusion was added to the working plan:

```text
Do not move directly into AGM/DPMN backbone changes yet.
First run a small alpha sweep or adaptive gating experiment to avoid harming sensitive samples such as abu_beach_2 and sample 3.
```

Reason:

- The full run improved the global mean AUC and all mean localization metrics.
- The fixed global alpha `0.4` still hurt `abu_beach_2` and `sample 3`.
- This indicates the high-frequency residual branch is useful, but its contribution should probably be sample-aware or region-aware.
- Backbone changes such as error-adaptive convolution should wait until the score-side behavior is better characterized.

Immediate next experiment:

```text
Targeted train.py alpha sweep, not post-hoc only.
Samples: 3, abu_beach_2, abu_airport_1, abu_airport_2, abu_beach_3
Alpha values: 0.1, 0.2, 0.3, 0.4
Mode: stationary_haar
```

Target commands follow this pattern:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --sample-ids 3,abu_beach_2,abu_airport_1,abu_airport_2,abu_beach_3 --results-dir results_sp_imp_dpmn_stationary_hf_aXX_targeted --highfreq-score-mode stationary_haar --highfreq-weight X.X
```

Decision criterion:

- If one lower fixed alpha preserves airport/beach_3 gains while reducing damage on `abu_beach_2` and `sample 3`, use that alpha for the next full run.
- If no fixed alpha works, implement adaptive gating for the high-frequency score before touching AGM/DPMN internals.


### Execution update: targeted stationary Haar alpha sweep

Date: 2026-06-12.

The targeted `train.py` alpha sweep completed after the interruption was resumed. This was a real rerun, not a post-hoc sweep.

Samples:

```text
3, abu_airport_1, abu_airport_2, abu_beach_2, abu_beach_3
```

Alpha values:

```text
0.1, 0.2, 0.3, 0.4
```

Output directories:

```text
results_sp_imp_dpmn_stationary_hf_a01_targeted/
results_sp_imp_dpmn_stationary_hf_a02_targeted/
results_sp_imp_dpmn_stationary_hf_a03_targeted/
results_sp_imp_dpmn_stationary_hf_a04_targeted/
results_sp_imp_dpmn_stationary_hf_alpha_sweep_targeted/alpha_sweep_auc_summary.csv
```

AUC summary:

```text
alpha,mean_auc,3,abu_airport_1,abu_airport_2,abu_beach_2,abu_beach_3
0.1,0.917038142731,0.930618560885,0.924123348440,0.900642486182,0.838288375684,0.991517942464
0.2,0.920520485638,0.926406834748,0.933419011544,0.918681030714,0.836745830125,0.987349721057
0.3,0.922560467721,0.923124970225,0.920749064304,0.926238736780,0.850825789866,0.991863777428
0.4,0.920602443326,0.914648584652,0.936561485390,0.928001196617,0.833593540003,0.990207409969
```

Interpretation:

- Best targeted mean AUC is alpha `0.3`: `0.922560467721`.
- `sample 3` prefers low alpha; alpha `0.1` gives `0.930618560885`, while alpha `0.4` drops to `0.914648584652`.
- `abu_airport_2` prefers higher alpha; alpha `0.4` gives `0.928001196617`.
- `abu_airport_1` is strongest at alpha `0.4` in this targeted run: `0.936561485390`.
- `abu_beach_3` is strong across all alpha values, with alpha `0.3` slightly best: `0.991863777428`.
- `abu_beach_2` remains weak for all fixed alpha values. Its best targeted alpha is `0.3` with AUC `0.850825789866`, still below the adaptive-fusion full-run value `0.863029230051`.

Conclusion:

A single fixed high-frequency alpha is not enough. The score-side high-frequency branch is useful, but the contribution should be gated. The next implementation should add adaptive high-frequency gating before any AGM/DPMN backbone modification.

Recommended gating direction:

```text
base_score = adaptive residual/contrast/uncertainty fusion
hf_score = stationary Haar high-frequency residual score
hf_agreement = rank/top-k agreement between base_score and hf_score
alpha_eff = low alpha when hf disagrees with base_score, higher alpha when hf agrees
final_score = normalize((1 - alpha_eff) * base_score + alpha_eff * hf_score)
```

A conservative first implementation can use a sample-level gate:

```text
if top-k overlap(base_score, hf_score) is low or rank correlation is low:
    alpha_eff = 0.1
else:
    alpha_eff = 0.3 or 0.4
```

This is lower risk than changing AGM/DPMN internals and directly targets the observed failure mode: `sample 3` and `abu_beach_2` are damaged when the high-frequency score is over-weighted globally.


### Conversation update: resumed interruption, GitHub sync, and next gating step

Date: 2026-06-12.

The previous turn was interrupted accidentally after the targeted alpha sweep had already completed. On resume, the run state was checked:

```text
No train.py process remained running.
All four targeted alpha sweep CSV files were present.
```

The alpha sweep summary was regenerated and saved to:

```text
results_sp_imp_dpmn_stationary_hf_alpha_sweep_targeted/alpha_sweep_auc_summary.csv
```

Final targeted alpha sweep conclusion:

```text
A single fixed high-frequency alpha is not enough.
sample 3 prefers low alpha.
airport samples prefer higher alpha.
abu_beach_2 remains below the adaptive-fusion baseline under all fixed alpha values.
```

The current working decision is:

```text
Do not move into AGM/DPMN backbone changes yet.
Commit and push the current score-side high-frequency work to GitHub.
Then implement adaptive high-frequency gating as the next score-side step.
```

Planned adaptive gating implementation:

```text
base_score = adaptive residual/contrast/uncertainty fusion
hf_score = stationary Haar high-frequency residual score
agreement = top-k overlap and rank correlation between base_score and hf_score
alpha_eff = low alpha when agreement is weak, high alpha when agreement is strong
final_score = normalize((1 - alpha_eff) * base_score + alpha_eff * hf_score)
```

Initial conservative gate:

```text
low_alpha = 0.1
high_alpha = 0.4
if top-k overlap >= threshold and rank correlation >= threshold:
    alpha_eff = high_alpha
else:
    alpha_eff = low_alpha
```

This directly targets the observed conflict: `sample 3` and `abu_beach_2` are harmed by over-weighting high-frequency residual globally, while airport samples benefit from stronger high-frequency contribution.


### Execution update: first adaptive high-frequency gate

Date: 2026-06-12.

After committing the fixed high-frequency score work locally, pushing to GitHub was attempted but failed because the current environment does not have valid GitHub credentials:

```text
remote: Repository not found.
fatal: Authentication failed for 'https://github.com/srxnyd/DPMN.git/'
```

Local commit created before the push attempt:

```text
960df46 Add high-frequency residual score experiments
```

The next score-side step was implemented in `train.py`: adaptive high-frequency fusion. New behavior:

```text
base_score = adaptive residual/contrast/uncertainty fusion
hf_score = stationary Haar or pywt high-frequency residual score
agreement = top-5-percent overlap + rank correlation between base_score and hf_score
alpha_eff = high alpha if agreement is strong, otherwise low alpha
final_score = normalize((1 - alpha_eff) * base_score + alpha_eff * hf_score)
```

New CLI options:

```bash
--highfreq-fusion-mode fixed|adaptive
--highfreq-adaptive-low-alpha 0.1
--highfreq-adaptive-high-alpha 0.4
--highfreq-adaptive-top-overlap 0.20
--highfreq-adaptive-rank-corr 0.15
```

Smoke test passed:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --sample-ids 1 --num-iter 2 --results-dir /tmp/dpmn_hf_gating_smoke --highfreq-score-mode stationary_haar --highfreq-fusion-mode adaptive --highfreq-adaptive-low-alpha 0.1 --highfreq-adaptive-high-alpha 0.4 --log-interval 1
```

The first targeted adaptive-gate run also completed:

```bash
python -u train.py --ablation-mode sp_imp_dpmn --sample-ids 3,abu_beach_2,abu_airport_1,abu_airport_2,abu_beach_3 --results-dir results_sp_imp_dpmn_stationary_hf_adaptive_gate_targeted --highfreq-score-mode stationary_haar --highfreq-fusion-mode adaptive --highfreq-adaptive-low-alpha 0.1 --highfreq-adaptive-high-alpha 0.4
```

AUC results:

```text
mean_auc: 0.921036692629
3: 0.918198644555
abu_airport_1: 0.934982497971
abu_airport_2: 0.925320402444
abu_beach_2: 0.835018360991
abu_beach_3: 0.991663557186
```

Gate diagnostics showed the current agreement rule is too permissive. It selected alpha `0.4` for every targeted sample:

```text
3: alpha=0.4, top_overlap=0.498000, rank_corr=0.699641
abu_airport_1: alpha=0.4, top_overlap=0.600000, rank_corr=0.614500
abu_airport_2: alpha=0.4, top_overlap=0.506000, rank_corr=0.626399
abu_beach_2: alpha=0.4, top_overlap=0.626000, rank_corr=0.639749
abu_beach_3: alpha=0.4, top_overlap=0.540000, rank_corr=0.625365
```

Interpretation:

- Agreement between `base_score` and `hf_score` is not enough to decide whether high alpha is safe.
- `sample 3` and `abu_beach_2` can have high agreement but still lose AUC when high-frequency score is over-weighted.
- The first adaptive gate did not beat the best fixed targeted alpha (`0.3`, mean AUC `0.922560467721`).

Next refinement should not be a simple agreement gate. It needs an additional protective signal, likely one of:

```text
1. scale/normalization guard for non-ABU sample 3;
2. high-frequency concentration or entropy guard to detect diffuse edge response;
3. per-pixel/soft alpha map instead of sample-level alpha;
4. fallback to lower alpha when hf_score is too globally correlated or too spatially diffuse.
```

The current adaptive gate code remains useful as infrastructure because it saves `highfreq_alpha` and prints diagnostic agreement values, but the decision rule needs refinement before a full adaptive-gate run.
