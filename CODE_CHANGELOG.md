# Code Changelog

## 2026-06-13 09:13:20 UTC - Goal Run Initialized

- Modified files: none yet.
- Training code changed: no.
- New CLI parameters or ablation modes: none yet.
- Effect on old `baseline`, `mask_sam`, `sp_imp_dpmn` behavior: none.
- Risk: none.

## 2026-06-13 09:42:00 UTC - Reliability-Gated Uncertainty Boost

- Modified file: `train.py`.
- Modified modules/functions:
  - `save_score_components`
  - `compute_uncertainty_reliability_stats` / `should_use_uncertainty_score`
  - `fuse_detection_scores`
  - `compute_final_artifacts`
  - `run_single_sample`
  - `parse_args`
  - `main`
- Purpose: add a default-off label-free uncertainty boost for samples where abundance uncertainty agrees with residual/contrast evidence. This targets `abu_beach_2`, where uncertainty is informative but the previous fixed weight `0.05` underuses it, while preserving uncertainty shutdown for unreliable cases like historical `abu_urban_2`.
- New CLI parameters:
  - `--adaptive-uncertainty-boost`
  - `--uncertainty-boost-weight` with default `0.25`
- Diagnostics added to score component MAT files when available:
  - `final_fusion_weights`
  - `uncertainty_gate_stats` = top overlap, residual correlation, contrast correlation, used flag, boosted flag
- Effect on old defaults:
  - Existing `baseline`, `mask_sam`, and `sp_imp_dpmn` default behavior is intended to remain unchanged because the boost is disabled unless `--adaptive-uncertainty-boost` is passed.
  - Existing adaptive uncertainty gate logic is preserved for default calls.
- Verification:
  - `python -m py_compile train.py` completed successfully.
  - `python train.py --help | grep -E 'adaptive-uncertainty|uncertainty-boost'` confirmed the new CLI flags.
- Risk:
  - When enabled, boosted uncertainty can affect both final scoring and the intermediate fused score used by multidirectional suppression during training. This is intentional for the candidate experiment but must be evaluated with targeted tests before any full run.

## 2026-06-13 10:00:00 UTC - Decouple Final-Score and Training-Time Uncertainty Boost

- Modified file: `train.py`.
- Modified modules/functions:
  - `run_single_sample`
  - `parse_args`
  - `main`
- Purpose: separate final-score reliability-gated uncertainty boost from the training-time fused score used by multidirectional suppression.
- New CLI parameter:
  - `--training-uncertainty-boost`
- Behavior after this change:
  - `--adaptive-uncertainty-boost` applies to final artifact/score fusion only.
  - Training-time use of boosted uncertainty requires explicit `--training-uncertainty-boost`.
- Effect on old defaults:
  - Existing `baseline`, `mask_sam`, and default `sp_imp_dpmn` behavior remains unchanged.
  - The newly added `--adaptive-uncertainty-boost` is now safer because it no longer changes training dynamics unless paired with `--training-uncertainty-boost`.
- Verification:
  - `python -m py_compile train.py` completed successfully.
  - `python train.py --help | grep -E 'adaptive-uncertainty|uncertainty-boost|training-uncertainty'` confirmed all relevant flags.
- Risk:
  - Final-only boosting may improve `abu_beach_2` less than training-coupled boosting, but should avoid the airport degradation seen in `results_goal_uncertainty_boost_md_targeted`.

## 2026-06-13 10:35:00 UTC - Stricter Boost Gate and Optional Random Seed

- Modified file: `train.py`.
- Modified modules/functions:
  - imports
  - `fuse_detection_scores`
  - `compute_final_artifacts`
  - `run_single_sample`
  - `parse_args`
  - `main`
- Purpose:
  - Make uncertainty boosting stricter than the base adaptive uncertainty-use gate.
  - Add optional reproducibility control for targeted comparisons.
- New CLI parameters:
  - `--uncertainty-boost-min-top-overlap` default `0.20`
  - `--uncertainty-boost-min-residual-corr` default `0.35`
  - `--uncertainty-boost-min-contrast-corr` default `0.10`
  - `--random-seed` default `None`
- Behavior:
  - `--adaptive-uncertainty-boost` now boosts uncertainty only if the base uncertainty gate passes and the stricter boost thresholds pass.
  - `--random-seed` affects Python, NumPy, Torch, and CUDA RNGs only when explicitly provided.
- Effect on old defaults:
  - Existing default modes remain unchanged because all new behavior is behind explicit CLI flags or an unset seed.
- Verification:
  - `python -m py_compile train.py` passed.
  - `python train.py --help | grep -E 'random-seed|uncertainty-boost-min|training-uncertainty|adaptive-uncertainty'` showed the new flags.
- Risk:
  - Seeded comparisons improve local reproducibility but do not retroactively make old unseeded runs directly comparable.

## 2026-06-13 11:55:00 UTC - Pre-Code-Edit Status For Ablation Modes

Command:
```bash
git status --short
```

Output:
```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Reason for edit:
- Existing `train.py` supports `sp_imp_dpmn`, but does not expose clean `sp_only` or `sp_obm` ablation groups.
- Add minimal explicit ablation modes so the planned ablation table can isolate superpixel perturbation and online background mining without changing old modes.

Planned behavior:
- `sp_only`: superpixel pooled target / anti-identity target only; no online background mining; no adaptive score fusion.
- `sp_obm`: `sp_only` plus online background mining; no adaptive score fusion.
- Existing `baseline`, `mask_sam`, and `sp_imp_dpmn` defaults must remain unchanged.

## 2026-06-13 12:00:00 UTC - Add Explicit `sp_only` and `sp_obm` Ablation Modes

Modified file: `train.py`.

Modified module:
- `ABLATION_MODES`.

Purpose:
- Add explicit ablation modes required by the system ablation plan.
- `sp_only` isolates superpixel perturbation / pooled target / anti-identity target behavior without online background mining and without adaptive score fusion.
- `sp_obm` adds online background mining on top of `sp_only`, but keeps adaptive score fusion disabled.

New ablation modes:
- `sp_only`
- `sp_obm`

Effect on old defaults:
- Existing `baseline`, `mask_sam`, and `sp_imp_dpmn` configurations were not changed.
- Existing `sp_imp_dpmn` remains the same main-line mode unless the user explicitly chooses a new mode.

Verification:
```bash
python -m py_compile train.py
python train.py --help | grep -n "sp_only\|sp_obm\|sp_imp_dpmn"
```

Results:
- `python -m py_compile train.py` exited successfully.
- Help output includes `sp_only` and `sp_obm` in `--ablation-mode` choices.

Risk:
- `sp_only` and `sp_obm` still inherit the `prior_mode="sp_imp"` training path. This is intentional because the ablation is designed to isolate superpixel target and online background mining, not to create a separate architecture.

## 2026-06-13 16:00:00 UTC - Pre-Code-Edit Status For Conservative Uncertainty Overlay

Command:
```bash
git status --short
```

Output:
```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Planned edit:
- Add a default-off conservative final-score uncertainty overlay.
- The overlay must run only after the normal final score has been computed.
- If the strict label-free uncertainty gate does not pass, final `fused_score` must remain unchanged.
- If the gate passes, mix a small amount of normalized `uncertainty_score` into the already computed final score.
- Add diagnostics to score component MAT files.

Reason:
- Targeted ablation shows MD/HF improve airport and urban2 but do not solve `abu_beach_2`.
- Earlier uncertainty boost improves `abu_beach_2` but recomputes final fusion and can destabilize other samples.
- Overlay should be less invasive because non-triggered samples keep their final score unchanged.

## 2026-06-13 16:10:00 UTC - Conservative Final-Score Uncertainty Overlay

Modified file: `train.py`.

Modified functions/modules:
- `save_score_components`
- `apply_conservative_uncertainty_overlay` (new helper)
- `compute_final_artifacts`
- `run_single_sample`
- `parse_args`
- `main`

Purpose:
- Add a default-off, label-free, conservative final-score uncertainty overlay.
- The overlay runs after normal final score computation, including high-frequency and region-prior fusion if enabled.
- If strict uncertainty agreement gates do not pass, the final fused score remains unchanged.
- If gates pass, only a small normalized `uncertainty_score` overlay is mixed into the already computed final score.

New CLI parameters:
- `--conservative-uncertainty-overlay`
- `--uncertainty-overlay-weight`, default `0.10`

Gate rule:
- Reuses existing label-free statistics: top overlap, residual/uncertainty rank correlation, contrast/uncertainty rank correlation.
- Reuses strict thresholds controlled by:
  - `--uncertainty-boost-min-top-overlap`
  - `--uncertainty-boost-min-residual-corr`
  - `--uncertainty-boost-min-contrast-corr`

Diagnostics added to score component MAT files:
- `uncertainty_overlay_stats` = top overlap, residual corr, contrast corr, enabled flag, applied flag, overlay weight.
- `pre_overlay_fused_score` = final score before overlay.

Effect on old defaults:
- Existing `baseline`, `mask_sam`, `sp_imp_dpmn`, `sp_only`, and `sp_obm` behavior remains unchanged unless `--conservative-uncertainty-overlay` is explicitly passed.
- `--adaptive-uncertainty-boost` behavior is not changed.
- Training-time dynamics are unchanged because overlay is applied only inside `compute_final_artifacts` after training.

Verification:
```bash
python -m py_compile train.py
python train.py --help | grep -n "conservative-uncertainty\|uncertainty-overlay"
```

Results:
- `python -m py_compile train.py` exited successfully.
- Help output includes `--conservative-uncertainty-overlay` and `--uncertainty-overlay-weight`.

Risk:
- Overlay may still hurt samples where uncertainty is correlated but not semantically aligned with anomalies.
- Because it is default-off and diagnostic-heavy, it should be evaluated only via targeted tests before any full15 run.

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

## 2026-06-13 16:55:00 UTC - No Additional Code Changes During Full Ablation Analysis

Modified files:
- None after the earlier `train.py` changes.

Purpose:
- Record that the subsequent full 15-sample ablation runs and result analysis did not introduce additional code changes.

Experiments completed after this point:
- `results_goal_full_sp_only`
- `results_goal_full_sp_obm`
- `results_goal_full_sp_obm_md_hf_diag`

Verification:
```bash
python -m py_compile train.py
```

Result:
- Compilation succeeded.

Effect on old defaults:
- No additional effect.
- The previously added overlay parameters remain default-off and were not used in the completed full ablations.

Risk:
- None from this analysis step.
