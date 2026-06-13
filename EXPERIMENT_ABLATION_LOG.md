# Experiment And Ablation Log

## 2026-06-13 09:13:20 UTC - Goal Run Initialized

No new experiments have been run yet.

Initial plan:
- First confirm existing project state and prior results.
- Then diagnose weak samples, especially `abu_beach_2`.
- Then run targeted experiments only, using new result directories.
- Full 15-sample experiments are allowed only after targeted results are stable.

## 2026-06-13 09:13:20 UTC - Existing Result Baseline Registry

No new experiments were run in Stage 1/2. Existing summaries were read and registered as comparison anchors:

| Anchor | Scope | mean_auc | Key per-sample notes |
| --- | --- | ---: | --- |
| Historical baseline: `resultold/batch_summary_baseline.txt` | full 15 | 0.949646376529 | `abu_beach_2=0.907725662340`, `abu_urban_2=0.987075803994` |
| Historical mask_sam: `resultnew_all_v5/batch_summary_mask_sam.txt` | full 15 | 0.940879525839 | `abu_beach_2=0.908014668583`, `abu_urban_2=0.999344025951` |
| Historical mask_sam v3vis: `resultfinal_v3vis/batch_summary_mask_sam.txt` | full 15 | 0.940115817311 | `abu_beach_2=0.806812463243`, `abu_urban_2=0.977124133751` |
| Initial SP-IMP: `results_sp_imp_dpmn/batch_summary_sp_imp_dpmn.txt` | full 15 | 0.916926731476 | `abu_beach_2=0.838357343083`, `abu_urban_2=0.639632366192` |
| Adaptive SP-IMP: `results_sp_imp_dpmn_adaptive_full/batch_summary_sp_imp_dpmn.txt` | full 15 | 0.939775035950 | `abu_beach_2=0.863029230051`, `abu_urban_2=0.993923229411` |
| HF fixed alpha 0.4: `results_sp_imp_dpmn_stationary_hf_a04/batch_summary_sp_imp_dpmn.txt` | full 15 | 0.956469872526 | `abu_beach_2=0.836264574100`, `abu_urban_2=0.991716115926` |
| HF diagnostic v2: `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full/batch_summary_sp_imp_dpmn.txt` | full 15 | 0.957551813119 | `abu_beach_2=0.853102977169`, `abu_urban_2=0.994495977981` |
| Current best md-only: `results_sp_imp_dpmn_md_only_full15/batch_summary_sp_imp_dpmn.txt` | full 15 | 0.958119941153 | `abu_beach_2=0.866870436278`, `abu_urban_2=0.994053637838` |

Current comparison anchors for future experiments:
- Best full mean: `0.958119941153` from `results_sp_imp_dpmn_md_only_full15`.
- Best full `abu_beach_2` among listed anchors: historical `mask_sam` v5 at `0.908014668583`, but its overall mean is lower.
- Current main-line `abu_beach_2`: `0.866870436278`.
- Current main-line `abu_urban_2`: `0.994053637838`.

## 2026-06-13 09:25:00 UTC - Planned Targeted Experiment: `goal_hf_md_edgeguard_targeted`

Experiment name: `goal_hf_md_edgeguard_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_hf_md_edgeguard_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --highfreq-score-mode stationary_haar \
  --highfreq-fusion-mode diagnostic \
  --highfreq-edge-guard
```

Scope: targeted, 7 samples. Priority samples are `abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3`; `abu_urban_2` is added as a stability sentinel.

Expected validation question:
- Does conservative HF-MD fusion with edge guard improve `abu_beach_2` or at least avoid worsening it while keeping `abu_urban_2`, sample `1`, sample `3`, and airport samples stable?

Result directory: `results_goal_hf_md_edgeguard_targeted`

Pre-run directory check: `OK`, directory did not exist.

Pre-run git status:
```text
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

## 2026-06-13 09:35:00 UTC - Result: `goal_hf_md_edgeguard_targeted`

Experiment name: `goal_hf_md_edgeguard_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_hf_md_edgeguard_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --highfreq-score-mode stationary_haar \
  --highfreq-fusion-mode diagnostic \
  --highfreq-edge-guard
```

Scope: targeted, 7 samples.

Result directory: `results_goal_hf_md_edgeguard_targeted`

Summary:
- mean_auc: `0.929970013404`
- `1`: `0.936028503393`
- `3`: `0.928531224471`
- `abu_airport_1`: `0.938358191288`
- `abu_airport_2`: `0.931509883109`
- `abu_airport_3`: `0.947550116690`
- `abu_beach_2`: `0.833684486024`
- `abu_urban_2`: `0.994127688855`

Comparison to current best md-only full15 per-sample anchors:
- `1`: improved from `0.932095560370` to `0.936028503393`.
- `3`: nearly unchanged from `0.928377717904` to `0.928531224471`.
- `abu_airport_1`: improved from `0.930877553436` to `0.938358191288`.
- `abu_airport_2`: improved from `0.924751081536` to `0.931509883109`.
- `abu_airport_3`: nearly unchanged from `0.947419663695` to `0.947550116690`.
- `abu_beach_2`: worsened from `0.866870436278` to `0.833684486024`.
- `abu_urban_2`: stable from `0.994053637838` to `0.994127688855`.

Post-hoc component AUC for `abu_beach_2` in this run:
- residual_score: `0.732042708251`
- contrast_score: `0.834009365419`
- uncertainty_score: `0.921784401343`
- highfreq_score: `0.827824530769`
- fused_score: `0.833684486024`
- highfreq alpha: `0.1078998`, alpha_map min/mean/max: `0.1/0.10789983/0.21041712`

Conclusion: locally helpful for some airport samples, stable for `abu_urban_2`, but harmful to the main weak sample `abu_beach_2`. This candidate is not eligible for full 15-sample evaluation. The next direction should not add high-frequency response for `abu_beach_2`; instead it should exploit reliability-gated uncertainty because uncertainty is strong on `abu_beach_2` but must remain suppressed on samples where it is unreliable.

## 2026-06-13 09:44:00 UTC - Planned Targeted Experiment: `goal_uncertainty_boost_md_targeted`

Experiment name: `goal_uncertainty_boost_md_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_uncertainty_boost_md_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --adaptive-uncertainty-boost \
  --uncertainty-boost-weight 0.25
```

Scope: targeted, 7 samples. Priority samples are `abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3`; `abu_urban_2` is included as an uncertainty-failure stability sentinel.

Expected validation question:
- Does label-free reliability-gated uncertainty boosting improve `abu_beach_2` without destabilizing `abu_urban_2`, sample `1`, sample `3`, and airport samples?

Result directory: `results_goal_uncertainty_boost_md_targeted`

Pre-run directory check: `OK`, directory did not exist.

Pre-run git status:
```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

## 2026-06-13 09:55:00 UTC - Result: `goal_uncertainty_boost_md_targeted`

Experiment name: `goal_uncertainty_boost_md_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_uncertainty_boost_md_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --adaptive-uncertainty-boost \
  --uncertainty-boost-weight 0.25
```

Scope: targeted, 7 samples.

Result directory: `results_goal_uncertainty_boost_md_targeted`

Summary:
- mean_auc: `0.920354420898`
- `1`: `0.940588091359`
- `3`: `0.931756626808`
- `abu_airport_1`: `0.900157405528`
- `abu_airport_2`: `0.857564257314`
- `abu_airport_3`: `0.909722338579`
- `abu_beach_2`: `0.908870571687`
- `abu_urban_2`: `0.993821655007`

Comparison to current best md-only full15 per-sample anchors:
- `abu_beach_2` improved strongly from `0.866870436278` to `0.908870571687`.
- `1` improved from `0.932095560370` to `0.940588091359`.
- `3` improved from `0.928377717904` to `0.931756626808`.
- `abu_urban_2` remained stable, changing from `0.994053637838` to `0.993821655007`.
- Airport samples degraded badly: `abu_airport_1` from `0.930877553436` to `0.900157405528`, `abu_airport_2` from `0.924751081536` to `0.857564257314`, `abu_airport_3` from `0.947419663695` to `0.909722338579`.

Diagnostics:
- `abu_beach_2` final weights were `[0.6315789, 0.11842106, 0.25]`, gate stats `[top_overlap=0.2280, residual_corr=0.4787, contrast_corr=0.1658, used=1, boosted=1]`.
- Airport and `abu_urban_2` final weights were `[0.85, 0.15, 0.0]`, so final uncertainty boost did not directly cause their final-score degradation.

Conclusion: locally effective for `abu_beach_2`, stable for `abu_urban_2`, but harmful to airport samples. The likely cause is that the enabled boost also affected training-time multidirectional suppression through the intermediate fused score. This candidate is not eligible for full 15-sample evaluation in its current form. Next fix: make `--adaptive-uncertainty-boost` affect final scoring only by default, with a separate explicit `--training-uncertainty-boost` flag for training-time use.

## 2026-06-13 10:02:00 UTC - Planned Targeted Experiment: `goal_uncertainty_final_only_md_targeted`

Experiment name: `goal_uncertainty_final_only_md_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_uncertainty_final_only_md_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --adaptive-uncertainty-boost \
  --uncertainty-boost-weight 0.25
```

Scope: targeted, 7 samples. This uses the revised final-only boost; it does not pass `--training-uncertainty-boost`.

Expected validation question:
- Does final-score-only reliability-gated uncertainty boosting retain the `abu_beach_2` improvement while avoiding the airport degradation caused by training-time boost coupling?

Result directory: `results_goal_uncertainty_final_only_md_targeted`

Pre-run directory check: `OK`, directory did not exist.

Pre-run git status:
```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

## 2026-06-13 10:15:00 UTC - Result: `goal_uncertainty_final_only_md_targeted`

Experiment name: `goal_uncertainty_final_only_md_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_uncertainty_final_only_md_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --adaptive-uncertainty-boost \
  --uncertainty-boost-weight 0.25
```

Scope: targeted, 7 samples. This used final-score-only boost; `--training-uncertainty-boost` was not enabled.

Result directory: `results_goal_uncertainty_final_only_md_targeted`

Summary:
- mean_auc: `0.923862041645`
- `1`: `0.937413090314`
- `3`: `0.932448288578`
- `abu_airport_1`: `0.917806694174`
- `abu_airport_2`: `0.877633109199`
- `abu_airport_3`: `0.910878463288`
- `abu_beach_2`: `0.891653479494`
- `abu_urban_2`: `0.999201166467`

Comparison to current best md-only full15 per-sample anchors:
- `abu_beach_2` improved from `0.866870436278` to `0.891653479494`.
- `1` improved from `0.932095560370` to `0.937413090314`.
- `3` improved from `0.928377717904` to `0.932448288578`.
- `abu_urban_2` improved from `0.994053637838` to `0.999201166467`.
- Airport samples degraded: `abu_airport_1` from `0.930877553436` to `0.917806694174`, `abu_airport_2` from `0.924751081536` to `0.877633109199`, `abu_airport_3` from `0.947419663695` to `0.910878463288`.

Diagnostics:
- Final uncertainty boost triggered only for `abu_beach_2`: final weights `[0.6315789, 0.11842106, 0.25]`, gate stats `[top_overlap=0.2280, residual_corr=0.4169, contrast_corr=0.1879, used=1, boosted=1]`.
- For `1`, `3`, airport samples, and `abu_urban_2`, final weights were `[0.85, 0.15, 0.0]`, so airport degradation in this rerun is not caused by final uncertainty boost. It appears to be run-to-run training variability or the current code path producing a different training trajectory, not a direct final-fusion effect.

Conclusion: final-only uncertainty boost is locally effective for `abu_beach_2` and stable for `abu_urban_2`, but the targeted rerun is not stable enough to justify a full 15-sample run. Next action is offline evaluation on existing `md_only_full15` score components to isolate the final fusion rule from training randomness.

## 2026-06-13 10:25:00 UTC - Offline Evaluation: Existing `md_only_full15` Final Uncertainty Boost

Scope: offline post-hoc evaluation on full 15 stored score components from `results_sp_imp_dpmn_md_only_full15`. This is not a new training run and is not a final reported method result.

Rule evaluated:
- Use the current `--adaptive-uncertainty-boost` final-fusion rule with uncertainty target weight `0.25` on existing score maps.

Summary:
- Existing stored fused mean AUC: `0.958119941153`.
- Offline boosted mean AUC: `0.945336408093`.
- Triggered samples: `abu_beach_2`, `abu_beach_3`, `abu_beach_4`.

Key per-sample deltas:
- `abu_beach_2`: `0.866870436278 -> 0.898356201205`, delta `+0.031485764927`.
- `abu_beach_3`: `0.988942382075 -> 0.950258921177`, delta `-0.038683460898`.
- `abu_beach_4`: `0.984730322229 -> 0.975027796425`, delta `-0.009702525803`.

Conclusion: current boost gate is locally useful but too broad. It cannot be the final method. Next targeted step should use stricter label-free reliability thresholds and same-seed targeted comparisons.

## 2026-06-13 10:35:00 UTC - Planned Targeted Experiment: `goal_md_seed0_targeted`

Experiment name: `goal_md_seed0_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_md_seed0_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --random-seed 0
```

Scope: targeted, 7 samples.

Expected validation question:
- Establish a same-seed targeted baseline for SP-IMP-DPMN + multidirectional suppression before testing stricter uncertainty boost.

Result directory: `results_goal_md_seed0_targeted`

Pre-run directory check: `OK_MD`, directory did not exist.

## 2026-06-13 10:50:00 UTC - Result: `goal_md_seed0_targeted`

Experiment name: `goal_md_seed0_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_md_seed0_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --random-seed 0
```

Scope: targeted, 7 samples.

Result directory: `results_goal_md_seed0_targeted`

Summary:
- mean_auc: `0.912948844147`
- `1`: `0.939740628674`
- `3`: `0.933341096884`
- `abu_airport_1`: `0.901527834145`
- `abu_airport_2`: `0.856031381061`
- `abu_airport_3`: `0.894680150799`
- `abu_beach_2`: `0.872845842453`
- `abu_urban_2`: `0.992474975016`

Conclusion: seed0 baseline itself is much weaker on airport samples than the existing unseeded current-best full run. Use this only as a same-seed comparison anchor for strict uncertainty boost, not as a final method result.

## 2026-06-13 10:50:00 UTC - Planned Targeted Experiment: `goal_strict_uncertainty_seed0_targeted`

Experiment name: `goal_strict_uncertainty_seed0_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_strict_uncertainty_seed0_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --adaptive-uncertainty-boost \
  --uncertainty-boost-weight 0.25 \
  --uncertainty-boost-min-residual-corr 0.35 \
  --uncertainty-boost-min-contrast-corr 0.10 \
  --random-seed 0
```

Scope: targeted, 7 samples, same seed as `goal_md_seed0_targeted`.

Expected validation question:
- Does stricter final-only uncertainty boost improve `abu_beach_2` relative to the same-seed md baseline without harming airport samples or `abu_urban_2`?

Result directory: `results_goal_strict_uncertainty_seed0_targeted`

## 2026-06-13 11:20:00 UTC - Result: `goal_strict_uncertainty_seed0_targeted`

Experiment name: `goal_strict_uncertainty_seed0_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_strict_uncertainty_seed0_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --adaptive-uncertainty-boost \
  --uncertainty-boost-weight 0.25 \
  --uncertainty-boost-min-residual-corr 0.35 \
  --uncertainty-boost-min-contrast-corr 0.10 \
  --random-seed 0
```

Scope: targeted, 7 samples, same seed as `results_goal_md_seed0_targeted`.

Result directory: `results_goal_strict_uncertainty_seed0_targeted`

Summary:
- mean_auc: `0.914178548004`
- `1`: `0.937938278457`
- `3`: `0.932820586113`
- `abu_airport_1`: `0.891242925911`
- `abu_airport_2`: `0.840052131707`
- `abu_airport_3`: `0.901318293340`
- `abu_beach_2`: `0.903818772875`
- `abu_urban_2`: `0.992058847622`

Same-seed comparison to `results_goal_md_seed0_targeted`:
- mean: `0.912948844147 -> 0.914178548004`, delta `+0.001229703857`
- `1`: delta `-0.001802350217`
- `3`: delta `-0.000520510771`
- `abu_airport_1`: delta `-0.010284908234`
- `abu_airport_2`: delta `-0.015979249354`
- `abu_airport_3`: delta `+0.006638142541`
- `abu_beach_2`: delta `+0.030972930422`
- `abu_urban_2`: delta `-0.000416127394`

Fusion diagnostics:
- Boost triggered only for `abu_beach_2`.
- `abu_beach_2` final weights: `[0.6315789, 0.1184211, 0.25]`.
- `abu_beach_2` gate stats: top overlap `0.2140`, residual corr `0.4520`, contrast corr `0.2014`, used `1`, boosted `1`.
- All other targeted samples used weights `[0.85, 0.15, 0.0]` or no boost.

Conclusion: strict final-only uncertainty boost is locally effective for `abu_beach_2` and preserves `abu_urban_2`, but targeted stability is still insufficient because airport_1 and airport_2 degrade relative to the same-seed md-only targeted anchor. Do not run full 15-sample evaluation for this candidate. Keep the result as evidence that label-free uncertainty can rescue `abu_beach_2`, but the final method needs a less invasive or more stable integration.

## 2026-06-13 11:25:00 UTC - Offline Diagnostic: Strict Gate on Existing `md_only_full15` Components

Scope: post-hoc diagnostic only, using stored score components from `results_sp_imp_dpmn_md_only_full15` and ground-truth masks only for evaluation.

Procedure:
- Reused `train.py::fuse_detection_scores` with strict boost thresholds.
- Selected mask orientation by reproducing the stored `batch_summary_sp_imp_dpmn.txt` AUC for the old `fused_score`.

Result:
- Boost triggered only for `abu_beach_2`.
- Existing stored fused mean: `0.958119941153`.
- Recomputed strict-gate mean: `0.945003251739`.
- `abu_beach_2`: `0.866870436278 -> 0.898356201205`, delta `+0.031485764927`.
- `abu_urban_2`: `0.994053637838 -> 0.993106046954`, delta `-0.000947590885`.

Important limitation:
- Non-triggered samples changed substantially in the offline recomputation because stored `fused_score` appears to include later method-specific processing that is not exactly reproduced by recomputing only from `residual_score`, `contrast_score`, and `uncertainty_score`.
- Therefore this offline diagnostic is useful for confirming that the strict gate targets `abu_beach_2`, but it is not a valid replacement for a real full 15-sample training/evaluation run.

Decision:
- Do not promote strict uncertainty boost to the final method yet.
- Do not run full 15 samples yet because targeted stability is not sufficient.
- Next conservative direction should avoid changing training dynamics and avoid recomputing the entire final score for non-boosted samples where possible.

## 2026-06-13 11:50:00 UTC - System Ablation Plan Added Before Continuation

This section is added because ablation must be planned immediately before any further method exploration or full 15-sample run.

Current requirement:
- Do not run a full 15-sample experiment until targeted ablation is stable.
- Use the same dataset, same evaluation flow, and as much as possible the same seed/training length for each ablation group.
- Ground truth/AUC must only be used after each experiment for evaluation and logging, never for training, score fusion rule selection, or sample-adaptive decisions.
- Every experiment must use a new `results-dir`; do not overwrite existing results.

Targeted ablation sample set:
```text
abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2
```

`abu_urban_2` is included as a stability sentinel because earlier uncertainty-related routes could harm or depend on it.

Full 15-sample ablation:
- Only allowed after targeted results show no major degradation on `abu_beach_2`, `1`, `3`, airport samples, and `abu_urban_2`.
- Full 15-sample results are the only results that may be treated as final paper evidence.
- Targeted mean is diagnostic only and must not be reported as final method mean.

Planned ablation groups:

| Method | Purpose | Expected implementation path | Superpixel perturbation | Online background mining | Adaptive score fusion | Multidirectional suppression | HF diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | Original DPMN/DIP reconstruction residual baseline | `--ablation-mode baseline` | no | no | no | no | no |
| `mask_sam` | Historical strong improved baseline | `--ablation-mode mask_sam` | no | adaptive mask only | no/main historical fusion | no | no |
| `sp_only` | Isolate superpixel pooled target / anti-identity perturbation | likely new CLI/mode or `sp_imp_dpmn` with OBM/fusion disabled | yes | no | no | no | no |
| `sp_obm` | Add online background mining over `sp_only` | likely new CLI/mode or config flag | yes | yes | no | no | no |
| `sp_obm_adaptive_fusion` | Add adaptive score fusion / uncertainty reliability gate | existing `sp_imp_dpmn` default if no extra MD/HF flags | yes | yes | yes | no | no |
| `sp_obm_md` | Current strongest main line | `--ablation-mode sp_imp_dpmn --multidir-suppression` | yes | yes | yes | yes | no |
| `sp_obm_hf_diag` | Independent HF diagnostic contribution without MD | `--ablation-mode sp_imp_dpmn --highfreq-score-mode stationary_haar --highfreq-fusion-mode diagnostic` | yes | yes | yes | no | yes |
| `sp_obm_md_hf_diag` | Test conservative MD + HF complementarity | `--ablation-mode sp_imp_dpmn --multidir-suppression --highfreq-score-mode stationary_haar --highfreq-fusion-mode diagnostic` plus any existing edge guard only if planned | yes | yes | yes | yes | yes |
| `prior_blindspot` optional failure control | Demonstrate unreliable route as failure/diagnostic control | `--ablation-mode prior_blindspot` | no | prior route | varies | no | no |

Initial targeted commands to prepare, with unique result dirs:

```bash
python train.py --ablation-mode baseline \
  --results-dir results_goal_ablate_targeted_baseline \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --random-seed 0
```

```bash
python train.py --ablation-mode mask_sam \
  --results-dir results_goal_ablate_targeted_mask_sam \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_ablate_targeted_sp_obm_adaptive_fusion \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_ablate_targeted_sp_obm_md \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_ablate_targeted_sp_obm_hf_diag \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --highfreq-score-mode stationary_haar \
  --highfreq-fusion-mode diagnostic \
  --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_ablate_targeted_sp_obm_md_hf_diag \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --highfreq-score-mode stationary_haar \
  --highfreq-fusion-mode diagnostic \
  --random-seed 0
```

Need code support check before running `sp_only` and `sp_obm`:
- Inspect `train.py` config construction and `ABLATION_CONFIGS`/mode handling.
- If no clean CLI exists to disable online background mining or adaptive fusion inside `sp_imp_dpmn`, add minimal default-off ablation modes or flags.
- Preserve old default behavior for `baseline`, `mask_sam`, and `sp_imp_dpmn`.
- After modifying `train.py`, run `python -m py_compile train.py` and record the result.

Ablation table fields to produce after experiments:
- method name
- superpixel perturbation enabled
- online background mining enabled
- adaptive score fusion enabled
- multidirectional suppression enabled
- highfreq diagnostic enabled
- targeted mean_auc and, if later eligible, full 15-sample mean_auc
- `abu_beach_2` AUC
- `abu_urban_2` AUC
- comparison to `results_sp_imp_dpmn_md_only_full15`
- comparison to `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full`
- comparison to `resultold/batch_summary_baseline.txt`
- conclusion: effective, locally effective, ineffective, or harms stability

Immediate next action:
- Check whether `sp_only` and `sp_obm` are already expressible through existing CLI/config.
- If not, add the smallest explicit ablation mode or CLI flags needed for systematic ablation.

## 2026-06-13 12:05:00 UTC - Planned Targeted System Ablation Batch

Scope: targeted, 7 samples.

Samples:
```text
abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2
```

Fairness controls:
- `--num-iter 800` for all methods.
- `--random-seed 0` for all methods.
- Same sample list and same evaluation flow.
- AUC/ground truth used only after each run for evaluation.

Pre-run directory check:
```text
OK results_goal_ablate_targeted_baseline
OK results_goal_ablate_targeted_mask_sam
OK results_goal_ablate_targeted_sp_only
OK results_goal_ablate_targeted_sp_obm
OK results_goal_ablate_targeted_sp_obm_adaptive_fusion
OK results_goal_ablate_targeted_sp_obm_md
OK results_goal_ablate_targeted_sp_obm_hf_diag
OK results_goal_ablate_targeted_sp_obm_md_hf_diag
```

Commands to run sequentially:

```bash
python train.py --ablation-mode baseline --results-dir results_goal_ablate_targeted_baseline --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode mask_sam --results-dir results_goal_ablate_targeted_mask_sam --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode sp_only --results-dir results_goal_ablate_targeted_sp_only --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode sp_obm --results-dir results_goal_ablate_targeted_sp_obm --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_goal_ablate_targeted_sp_obm_adaptive_fusion --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_goal_ablate_targeted_sp_obm_md --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --multidir-suppression --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_goal_ablate_targeted_sp_obm_hf_diag --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --highfreq-score-mode stationary_haar --highfreq-fusion-mode diagnostic --num-iter 800 --random-seed 0
```

```bash
python train.py --ablation-mode sp_imp_dpmn --results-dir results_goal_ablate_targeted_sp_obm_md_hf_diag --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 --multidir-suppression --highfreq-score-mode stationary_haar --highfreq-fusion-mode diagnostic --num-iter 800 --random-seed 0
```

## 2026-06-13 15:40:00 UTC - Result: Targeted System Ablation Batch

Scope: targeted, 7 samples.

Samples:
```text
abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2
```

Fairness controls used:
- `--num-iter 800`
- `--random-seed 0`
- Same sample IDs and evaluation flow for all groups.
- AUC used only after each run for evaluation.

Result directories:
- `results_goal_ablate_targeted_baseline`
- `results_goal_ablate_targeted_mask_sam`
- `results_goal_ablate_targeted_sp_only`
- `results_goal_ablate_targeted_sp_obm`
- `results_goal_ablate_targeted_sp_obm_adaptive_fusion`
- `results_goal_ablate_targeted_sp_obm_md`
- `results_goal_ablate_targeted_sp_obm_hf_diag`
- `results_goal_ablate_targeted_sp_obm_md_hf_diag`

Targeted ablation table:

| method | SP | OBM | adaptive fusion | MD | HF diag | targeted mean | beach2 | urban2 | sample1 | sample3 | airport1 | airport2 | airport3 | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | 0 | 0 | 0 | 0 | 0.840707286026 | 0.911976883543 | 0.793256770262 | 0.985079882310 | 0.905355261835 | 0.881879622114 | 0.878195473029 | 0.529207109090 | reference baseline only under 800-iter seed0 targeted setting |
| mask_sam | 0 | 0 | 0 | 0 | 0 | 0.817316705151 | 0.902063514680 | 0.809591572601 | 0.969103420288 | 0.876644240592 | 0.689658865440 | 0.851875686287 | 0.622279636168 | reference baseline only under 800-iter seed0 targeted setting |
| sp_only | 1 | 0 | 0 | 0 | 0 | 0.890674632997 | 0.871375043199 | 0.868473598847 | 0.937037103349 | 0.931499018087 | 0.889217932675 | 0.895766733802 | 0.841353001017 | superpixel target gives clear stability gain over baseline/mask_sam |
| sp_obm | 1 | 1 | 0 | 0 | 0 | 0.886174087730 | 0.869797129744 | 0.876380019332 | 0.939901765945 | 0.932531217413 | 0.856748286436 | 0.833745540223 | 0.894114655018 | OBM alone is mixed and slightly below sp_only |
| sp_obm_adaptive_fusion | 1 | 1 | 1 | 0 | 0 | 0.902144736661 | 0.769446280207 | 0.993834761382 | 0.941620563503 | 0.930551512040 | 0.894622142181 | 0.873838022984 | 0.911099874334 | adaptive fusion boosts urban2 but beach2 fails |
| sp_obm_md | 1 | 1 | 1 | 1 | 0 | 0.912452982151 | 0.852661383713 | 0.992309179377 | 0.936607403959 | 0.932192444301 | 0.895638866342 | 0.882307106308 | 0.895454491054 | best non-HF main-line targeted mean; stable urban2 but beach2 weak |
| sp_obm_hf_diag | 1 | 1 | 1 | 0 | 1 | 0.927790965142 | 0.849183708940 | 0.993467127574 | 0.931671829027 | 0.929392272797 | 0.932996257215 | 0.915008852882 | 0.942816707558 | best targeted mean overall; improves airport/urban2 but beach2 still weak |
| sp_obm_md_hf_diag | 1 | 1 | 1 | 1 | 1 | 0.927348716405 | 0.831024820180 | 0.993512344567 | 0.933366754397 | 0.928337135709 | 0.935994994589 | 0.930630972217 | 0.938573993178 | near HF-only mean; improves airport but hurts beach2 more than MD-only |

Comparison notes:
- Under the controlled 800-iter seed0 targeted setting, `sp_only` gives a clear improvement over `baseline` and `mask_sam` in targeted mean and especially airport_3, supporting the anti-identity superpixel target contribution.
- `sp_obm` alone is mixed: it improves airport_3 and urban2 over `sp_only`, but hurts airport_1, airport_2, and beach2; it should be framed as useful only when paired with later fusion/suppression rather than a standalone guaranteed gain.
- `sp_obm_adaptive_fusion` dramatically stabilizes `abu_urban_2` but collapses `abu_beach_2`, confirming the earlier diagnosis that beach2 needs special care in final fusion.
- `sp_obm_md` is the strongest non-HF main-line targeted group and preserves `abu_urban_2`, but `abu_beach_2=0.852661383713` is still weak.
- `sp_obm_hf_diag` has the best targeted mean and the best airport scores, but `abu_beach_2=0.849183708940` remains weak.
- `sp_obm_md_hf_diag` is not complementary for beach2: airport scores improve, but `abu_beach_2` drops to `0.831024820180`, below MD-only and HF-only.

Decision:
- Do not run full 15-sample ablation yet.
- The targeted ablation supports a paper narrative that superpixel anti-identity target is the key stabilizing base, adaptive fusion fixes uncertainty failure on `abu_urban_2`, and MD/HF improve overall/airport stability.
- However, the full method still needs a conservative, label-free beach2-safe final score mechanism before full15, because all current main-line targeted variants leave `abu_beach_2` below the historical baseline/mask_sam beach2 level.
- Next method step remains: implement a default-off conservative final-score uncertainty overlay that leaves the original final fused score unchanged unless strict label-free uncertainty agreement gate fires.

Full-run eligibility:
- Not eligible yet.
- Reason: targeted mean improved for HF variants, but `abu_beach_2` remains weak and MD+HF hurts it further. Running full15 now would likely confirm the same failure rather than advance the method.

## 2026-06-13 16:15:00 UTC - Planned Targeted Experiment: `goal_overlay_md_targeted`

Experiment name: `goal_overlay_md_targeted`

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_overlay_md_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --conservative-uncertainty-overlay \
  --uncertainty-overlay-weight 0.15 \
  --num-iter 800 \
  --random-seed 0
```

Scope: targeted, 7 samples.

Expected validation question:
- Does a conservative final-score-only uncertainty overlay improve `abu_beach_2` relative to the same-seed `sp_obm_md` targeted anchor, without harming `abu_urban_2`, sample `1`, sample `3`, or airport samples?

Comparison anchor:
- `results_goal_ablate_targeted_sp_obm_md`, mean `0.912452982151`, `abu_beach_2=0.852661383713`, `abu_urban_2=0.992309179377`.

Result directory: `results_goal_overlay_md_targeted`

Pre-run directory check: `OK`, directory did not exist.

## 2026-06-13 16:25:00 UTC - Fix Overlay Signature After Failed Targeted Run

Failed experiment:
- `results_goal_overlay_md_targeted`

Command attempted:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_overlay_md_targeted \
  --sample-ids abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2 \
  --multidir-suppression \
  --conservative-uncertainty-overlay \
  --uncertainty-overlay-weight 0.15 \
  --num-iter 800 \
  --random-seed 0
```

Outcome:
- Failed at the end of sample `1` before any valid `batch_summary*.txt` was produced.
- Error: `TypeError: compute_final_artifacts() got an unexpected keyword argument 'conservative_uncertainty_overlay'`.

Cause:
- The call site passed the new overlay parameters, but `compute_final_artifacts` signature had not been updated.

Code fix:
- Added `conservative_uncertainty_overlay=False` and `uncertainty_overlay_weight=0.10` to `compute_final_artifacts` signature.

Safety decision:
- Do not reuse or overwrite `results_goal_overlay_md_targeted`.
- Re-run with a new directory: `results_goal_overlay_md_targeted_v2`.

## 2026-06-13 15:47:17 UTC - Experiment Plan Change: No More New Modules

User instruction:
- Stop trying new innovation modules.
- Complete ablations for existing methods.
- Run/verify full 15-sample evaluation with the best existing method.

Experiment decision:
- Cancel the planned rerun `results_goal_overlay_md_targeted_v2`.
- Do not evaluate `--conservative-uncertainty-overlay` in the current run.
- Continue with existing method ablations only.

Candidate full ablation groups:
- `baseline`
- `mask_sam`
- `sp_only`
- `sp_obm`
- `sp_obm_adaptive_fusion`
- `sp_obm_md`
- `sp_obm_hf_diag`
- `sp_obm_md_hf_diag`

Next step:
- Audit existing full summaries and run only missing full 15-sample ablations in fresh result directories.

## 2026-06-13 15:47:17 UTC - Planned Full Ablation Runs for Existing Methods

### Experiment: `goal_full_sp_only`

Command:
```bash
python train.py --ablation-mode sp_only \
  --results-dir results_goal_full_sp_only \
  --num-iter 800 \
  --random-seed 0
```

Scope: full 15 samples.

Expected validation question:
- Does superpixel pooled target alone provide a stable full-set gain relative to historical `baseline` and `mask_sam`?

Result directory: `results_goal_full_sp_only`

Pre-run directory check: OK, directory did not exist.

## 2026-06-13 16:05:00 UTC - Result: `goal_full_sp_only`

Result directory: `results_goal_full_sp_only`

Summary file: `results_goal_full_sp_only/batch_summary_sp_only.txt`

Scope: full 15 samples.

Command:
```bash
python train.py --ablation-mode sp_only \
  --results-dir results_goal_full_sp_only \
  --num-iter 800 \
  --random-seed 0
```

Result:
- mean_auc: `0.922499903785`
- sample_count: `15`

Per-sample AUC:
```text
1: 0.943190159884
3: 0.932326541991
abu_airport_1: 0.860413566468
abu_airport_2: 0.870824448565
abu_airport_3: 0.854729220274
abu_airport_4: 0.957987927565
abu_beach_1: 0.961258513037
abu_beach_2: 0.848679463782
abu_beach_3: 0.954176867281
abu_beach_4: 0.964006251573
abu_urban_1: 0.984146017121
abu_urban_2: 0.872942217271
abu_urban_3: 0.959435216974
abu_urban_4: 0.925482687935
abu_urban_5: 0.947899457059
```

Comparison:
- vs `results_sp_imp_dpmn_md_only_full15`: `-0.035620037368` mean_auc.
- vs `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full`: `-0.035051909334` mean_auc.
- vs historical baseline `resultold`: `-0.027146472744` mean_auc.
- `abu_beach_2`: `0.848679463782`, below MD full `0.866870436278`, below HF diagnostic full `0.853102977169`, and below historical baseline `0.907725662340`.
- `abu_urban_2`: `0.872942217271`, far below MD full `0.994053637838`.

Conclusion:
- `sp_only` is not competitive as a final method on the full set.
- It remains useful as an ablation control showing that superpixel pooled target alone is insufficient; adaptive fusion and multidirectional suppression are necessary for difficult urban/airport stability.

## 2026-06-13 16:25:00 UTC - Result: `goal_full_sp_obm`

Result directory: `results_goal_full_sp_obm`

Summary file: `results_goal_full_sp_obm/batch_summary_sp_obm.txt`

Scope: full 15 samples.

Command:
```bash
python train.py --ablation-mode sp_obm \
  --results-dir results_goal_full_sp_obm \
  --num-iter 800 \
  --random-seed 0
```

Result:
- mean_auc: `0.921017903552`
- sample_count: `15`

Per-sample AUC:
```text
1: 0.943249840355
3: 0.931497253644
abu_airport_1: 0.860357903815
abu_airport_2: 0.857274379052
abu_airport_3: 0.828363951888
abu_airport_4: 0.956716968478
abu_beach_1: 0.954316978782
abu_beach_2: 0.846024345239
abu_beach_3: 0.944957635217
abu_beach_4: 0.966380748091
abu_urban_1: 0.977503001453
abu_urban_2: 0.872058192303
abu_urban_3: 0.962926448300
abu_urban_4: 0.954783342202
abu_urban_5: 0.958857564461
```

Comparison:
- vs `sp_only` full: `-0.001482000233` mean_auc.
- vs `results_sp_imp_dpmn_md_only_full15`: `-0.037102037601` mean_auc.
- vs `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full`: `-0.036533909567` mean_auc.
- vs historical baseline `resultold`: `-0.028628472977` mean_auc.
- `abu_beach_2`: `0.846024345239`, below `sp_only` `0.848679463782` and MD full `0.866870436278`.
- `abu_urban_2`: `0.872058192303`, effectively unchanged from `sp_only` and far below MD full `0.994053637838`.

Conclusion:
- `sp_obm` alone is not a stable full-set improvement over `sp_only`.
- Online background mining should not be claimed as an independently sufficient module; its role is only meaningful when later adaptive fusion and multidirectional suppression are added.

## 2026-06-13 16:50:00 UTC - Result: `goal_full_sp_obm_md_hf_diag`

Result directory: `results_goal_full_sp_obm_md_hf_diag`

Summary file: `results_goal_full_sp_obm_md_hf_diag/batch_summary_sp_imp_dpmn.txt`

Scope: full 15 samples.

Command:
```bash
python train.py --ablation-mode sp_imp_dpmn \
  --results-dir results_goal_full_sp_obm_md_hf_diag \
  --multidir-suppression \
  --highfreq-score-mode stationary_haar \
  --highfreq-fusion-mode diagnostic \
  --num-iter 800 \
  --random-seed 0
```

Result:
- mean_auc: `0.955963412793`
- sample_count: `15`

Per-sample AUC:
```text
1: 0.932256697641
3: 0.925364048762
abu_airport_1: 0.935238264340
abu_airport_2: 0.922283637763
abu_airport_3: 0.939487164143
abu_airport_4: 0.972323943662
abu_beach_1: 0.981462708861
abu_beach_2: 0.820634742592
abu_beach_3: 0.990871777137
abu_beach_4: 0.985975916758
abu_urban_1: 0.990587683750
abu_urban_2: 0.997206376251
abu_urban_3: 0.979837462497
abu_urban_4: 0.993996257014
abu_urban_5: 0.971924510718
```

Comparison:
- vs `results_sp_imp_dpmn_md_only_full15`: `-0.002156528360` mean_auc.
- vs `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full`: `-0.001588400326` mean_auc.
- vs historical baseline `resultold`: `+0.006317036264` mean_auc.
- `abu_beach_2`: `0.820634742592`, much worse than MD full `0.866870436278`, HF diagnostic full `0.853102977169`, and historical baseline `0.907725662340`.
- `abu_urban_2`: `0.997206376251`, higher than MD full `0.994053637838` and HF diagnostic full `0.994495977981`.

Conclusion:
- MD + high-frequency diagnostic fusion is locally useful on several samples (`abu_airport_1`, `abu_airport_4`, `abu_beach_1`, `abu_beach_3`, `abu_beach_4`, `abu_urban_2`), but it badly damages `abu_beach_2`.
- Because the project explicitly prioritizes improving or at least not further hurting `abu_beach_2`, this combination should not be the final mainline.
- The final recommended existing method remains `sp_obm_md`: SP-IMP-DPMN with adaptive score fusion and multidirectional suppression, no high-frequency diagnostic branch.

## 2026-06-13 16:55:00 UTC - Full 15-Sample Ablation Table and Analysis

Full ablation table:

| method | SP | OBM | adaptive fusion | MD | HF diag | full mean_auc | abu_beach_2 | abu_urban_2 | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | 0 | 0 | 0 | 0 | 0.949646376529 | 0.907725662340 | 0.987075803994 | strong historical reconstruction baseline; best beach2 among listed methods |
| mask_sam | 0 | 0 | 0 | 0 | 0 | 0.940879525839 | 0.908014668583 | 0.999344025951 | strong historical variant but lower mean |
| sp_only | 1 | 0 | 0 | 0 | 0 | 0.922499903785 | 0.848679463782 | 0.872942217271 | superpixel target alone is insufficient |
| sp_obm | 1 | 1 | 0 | 0 | 0 | 0.921017903552 | 0.846024345239 | 0.872058192303 | OBM alone is not stable |
| sp_obm_adaptive_fusion | 1 | 1 | 1 | 0 | 0 | 0.939775035950 | 0.863029230051 | 0.993923229411 | adaptive fusion fixes urban2 but remains below baseline mean |
| sp_obm_md | 1 | 1 | 1 | 1 | 0 | 0.958119941153 | 0.866870436278 | 0.994053637838 | best full mean and final recommended existing method |
| sp_obm_hf_diag | 1 | 1 | 1 | 0 | 1 | 0.957551813119 | 0.853102977169 | 0.994495977981 | near-best mean but weaker beach2 than MD-only |
| sp_obm_md_hf_diag | 1 | 1 | 1 | 1 | 1 | 0.955963412793 | 0.820634742592 | 0.997206376251 | locally useful but hurts beach2 too much |

Analysis:
- The best full 15-sample mean remains `sp_obm_md` at `0.958119941153`.
- `sp_only` and `sp_obm` are below historical baseline, so the paper should not claim that superpixel pooling or OBM alone is sufficient.
- `sp_obm_adaptive_fusion` explains a key mechanism: it raises `abu_urban_2` from about `0.872` to `0.993923`, showing that reliability-aware fusion is necessary for uncertainty-heavy urban scenes.
- `sp_obm_md` adds the strongest overall stability, improving mean AUC over historical baseline by `+0.008473564624` while keeping `abu_urban_2` stable.
- High-frequency diagnostic fusion is competitive in mean (`0.957551813119`) but weaker than MD-only on `abu_beach_2`.
- Combining MD and HF diagnostic is not complementary for the current final goal: it improves some samples but drops `abu_beach_2` to `0.820634742592`.

Final recommendation:
- Use `SP-IMP-DPMN + adaptive score fusion + multidirectional suppression` as the final existing-method mainline.
- Treat high-frequency diagnostic fusion as an auxiliary/diagnostic ablation, not the final method.
- Treat `abu_beach_2` as the principal remaining failure case: the current best mean method still underperforms historical baseline on this sample.
