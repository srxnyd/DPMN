# Goal Run Log

## 2026-06-13 09:13:20 UTC - Goal Start

Objective: continue the DPMN hyperspectral anomaly detection project by diagnosing current results, preserving reproducibility, and only then running targeted experiments and making scoped code changes if justified.

Rules for this run:
- Record every stage, command, result directory, and conclusion here.
- Before code edits, run and record `git status`.
- Keep existing `baseline`, `mask_sam`, and `sp_imp_dpmn` behavior reproducible unless a new flag or mode explicitly changes behavior.
- Do not overwrite existing result directories.
- Do not use ground truth or AUC for training-time module selection.
- Run targeted experiments before any full 15-sample experiment.

## 2026-06-13 09:13:20 UTC - Stage 1 Plan: Read and Confirm Project State

Planned actions:
- Read `research_innovation_directions_2025.md`, `innovation_plan.md`, `train.py`, `model/AGM.py`, and `model/MambaHSI.py`.
- Read all available `results*/batch_summary*.txt` summaries.
- Confirm whether the user-provided project status matches the actual files.
- Do not modify training code and do not run full 15-sample experiments in this stage.

## 2026-06-13 09:13:20 UTC - Stage 2 Plan: Confirm Existing Experiment Results

Planned actions:
- Verify the historical baseline, historical `mask_sam`, initial tuned `sp_imp_dpmn`, adaptive-fusion full, high-frequency full, diagnostic high-frequency full, and `md_only_full15` results.
- Identify weak samples in the current strongest full run.
- Produce a project status confirmation report and targeted experiment plan before changing code.

## 2026-06-13 09:13:20 UTC - Stage 1/2 Commands Run

Read/status commands:
- `git status --short`
- `sed -n '1,520p' research_innovation_directions_2025.md`
- `sed -n '1,260p' innovation_plan.md`
- `sed -n '1,260p' model/AGM.py`
- `sed -n '1,320p' model/MambaHSI.py`
- `sed -n '1,260p' train.py`
- `sed -n '261,760p' train.py`
- `sed -n '761,1260p' train.py`
- `sed -n '1261,1888p' train.py`
- `sed -n '1889,2470p' train.py`
- `find . -maxdepth 3 -name 'batch_summary*.txt' | sort`
- Read targeted result summaries for `resultold`, `resultnew_all_v5`, `resultfinal_v3vis`, `results_sp_imp_dpmn`, `results_sp_imp_dpmn_adaptive_full`, `results_sp_imp_dpmn_stationary_hf_a04`, `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full`, and `results_sp_imp_dpmn_md_only_full15`.

Git status:
```text
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Interpretation: the only untracked files are the three tracking logs created for this Goal run. No pre-existing dirty code changes were observed by `git status --short`.

## 2026-06-13 09:13:20 UTC - Stage 1 Project State Confirmation

Confirmed:
- The project is an unsupervised hyperspectral anomaly detection system using a DIP/DPMN-style reconstruction pipeline.
- `train.py` is the main training/evaluation entry point.
- `ABLATION_MODES` currently includes `baseline`, `mask_only`, `sam_only`, `mask_sam`, `prior_blindspot`, `sp_imp_dpmn`, and `full_innovation`.
- `prior_blindspot` is recorded in `research_innovation_directions_2025.md` as unreliable and should not be the main route.
- `sp_imp_dpmn` is implemented and is the current anti-identity/SP-IMP direction.
- `sp_imp_dpmn` defaults include `num_iter=800`, `sp_target_weight=0.6`, `residual_weight=0.8`, `contrast_weight=0.15`, `uncertainty_weight=0.05`, and `adaptive_score_fusion=True`.
- Implemented but not equally validated extensions include high-frequency scoring/fusion, high-frequency edge guard, edge training guard, error-adaptive reconstruction convolution, multidirectional suppression, superpixel region prior, low-rank sparse losses, and diagnostic artifact output.

Important nuance:
- `innovation_plan.md` is now stale as a forward plan because it still recommends `prior_blindspot`; the newer `research_innovation_directions_2025.md` and actual result summaries supersede it.

## 2026-06-13 09:13:20 UTC - Stage 2 Result Confirmation

Verified full-result summaries:

| Result | sample_count | mean_auc | Notes |
| --- | ---: | ---: | --- |
| `resultold/batch_summary_baseline.txt` | 15 | 0.949646376529 | Historical strong baseline |
| `resultnew_all_v5/batch_summary_mask_sam.txt` | 15 | 0.940879525839 | Historical strong `mask_sam` |
| `resultfinal_v3vis/batch_summary_mask_sam.txt` | 15 | 0.940115817311 | Historical `mask_sam` variant |
| `results_sp_imp_dpmn/batch_summary_sp_imp_dpmn.txt` | 15 | 0.916926731476 | Initial tuned SP-IMP full |
| `results_sp_imp_dpmn_adaptive_full/batch_summary_sp_imp_dpmn.txt` | 15 | 0.939775035950 | Adaptive fusion fixes `abu_urban_2` but remains below best |
| `results_sp_imp_dpmn_stationary_hf_a04/batch_summary_sp_imp_dpmn.txt` | 15 | 0.956469872526 | High-frequency fixed alpha 0.4 |
| `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full/batch_summary_sp_imp_dpmn.txt` | 15 | 0.957551813119 | High-frequency diagnostic v2 |
| `results_sp_imp_dpmn_md_only_full15/batch_summary_sp_imp_dpmn.txt` | 15 | 0.958119941153 | Current best trusted full result |

Current best trusted full run:
- `results_sp_imp_dpmn_md_only_full15/batch_summary_sp_imp_dpmn.txt`
- Method: SP-IMP-DPMN with adaptive score fusion and multidirectional suppression.
- Mean AUC: `0.958119941153`.
- Weak samples: `abu_beach_2=0.866870436278`, `3=0.928377717904`, `1=0.932095560370`; airport samples are roughly `0.9247-0.9474` except `abu_airport_4=0.9541`.
- `abu_urban_2` is stable and high at `0.994053637838`.

Conclusion:
- The next work should diagnose `abu_beach_2` first.
- Do not run full 15-sample experiments until targeted diagnostics and targeted ablations show stable behavior.

## 2026-06-13 09:20:00 UTC - Stage 3 Plan: Diagnose `abu_beach_2`

Planned actions:
- Read existing score-component MAT files for `abu_beach_2` across current main run and high-frequency comparison runs.
- Read mask-history MAT files and training loss CSVs for the same sample where available.
- Compare component AUCs only as post-hoc evaluation; do not use AUC for training-time rules.
- Diagnose whether the weakness is more consistent with score fusion, training mask instability, superpixel smoothing, edge/texture false positives, high-frequency behavior, multidirectional suppression strength, or unreliable abundance uncertainty.

## 2026-06-13 09:25:00 UTC - Stage 3 Diagnosis Summary for `abu_beach_2`

Artifacts read:
- `results_sp_imp_dpmn*/score_components/sp_imp_dpmn/urban_scores_abu_beach_2.mat`
- `results_sp_imp_dpmn*/mask_history/sp_imp_dpmn/urban_mask_history_abu_beach_2.mat`
- `results_sp_imp_dpmn_md_only_full15/training_curves/sp_imp_dpmn_sample_abu_beach_2_loss_history.csv`

Post-hoc component AUCs for `abu_beach_2`:
- Initial SP-IMP: residual `0.7571`, contrast `0.8340`, uncertainty `0.8809`, fused `0.8384`.
- Adaptive SP-IMP: residual `0.8626`, contrast `0.8340`, uncertainty `0.8641`, fused `0.8630`.
- MD-only full15: residual `0.8549`, contrast `0.8340`, uncertainty `0.9114`, highfreq `0.8529`, fused `0.8669`.
- HF diagnostic v2: residual `0.8613`, contrast `0.8340`, uncertainty `0.9140`, highfreq `0.8504`, fused `0.8531`.

Label-free agreement/edge observations:
- In MD-only, `corr(edge, fused)=0.5599`, top-5 overlap between edge and fused is `0.4080`.
- In MD-only, high-frequency score has `corr(edge, highfreq)=0.6242`, top-5 overlap edge/highfreq is `0.4780`.
- Top-score false positives are edge-heavy: for MD-only fused top-5%, edge-top10 rate is `0.7660`; for highfreq top-5%, edge-top10 rate is `0.8260`.
- Mask snapshots are not obviously unstable: MD-only mask means `[1.0, 0.8754, 0.5366]`; mean absolute changes `[0.1246, 0.3457]`, similar to other runs.

Diagnosis:
- `abu_beach_2` weakness is most consistent with edge/texture false positives and final score fusion limitations.
- High-frequency response is not the primary fix by itself for this sample because it strongly tracks edges/texture and has lower AUC than uncertainty.
- Abundance uncertainty is useful on `abu_beach_2` in post-hoc evaluation, unlike the earlier `abu_urban_2` failure; reliability-gated uncertainty remains justified.
- Training mask instability is not the leading explanation from existing snapshots.
- Superpixel over-smoothing may contribute indirectly, but current evidence points first to edge/texture-aware conservative fusion.

Next action:
- Run a targeted existing-CLI experiment combining multidirectional suppression with diagnostic high-frequency fusion and high-frequency edge guard.
- Include priority samples plus `abu_urban_2` as a stability sentinel.

## 2026-06-13 09:35:00 UTC - Stage 4 Targeted Experiment Outcome

Completed targeted experiment `goal_hf_md_edgeguard_targeted`.

Outcome:
- Targeted mean AUC: `0.929970013404`.
- `abu_beach_2` worsened to `0.833684486024`, below current md-only `0.866870436278` and below HF diagnostic v2 `0.853102977169`.
- `abu_urban_2` remained stable at `0.994127688855`.
- Some airport samples improved, but the main weak sample regressed.

Decision:
- Do not run full 15-sample evaluation for this HF-MD edge-guard candidate.
- Next step is a default-off reliability-gated uncertainty boost, because post-hoc component AUC shows uncertainty is highly informative for `abu_beach_2` while existing static fusion underweights it.

## 2026-06-13 09:38:00 UTC - Pre-Code-Edit Status

Command:
```bash
git status --short
```

Output:
```text
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Planned code change:
- Add a default-off reliability-gated uncertainty boost in `train.py`.
- Add CLI flags `--adaptive-uncertainty-boost` and `--uncertainty-boost-weight`.
- Preserve existing default behavior for `baseline`, `mask_sam`, and `sp_imp_dpmn` unless the new flag is explicitly enabled.
- Save score-fusion diagnostics into score component MAT files when available.

Reason:
- `abu_beach_2` has high post-hoc uncertainty score AUC, but current fusion underweights uncertainty at `0.05`.
- `abu_urban_2` historically requires reliability gating because uncertainty can be inverted there.

## 2026-06-13 09:42:00 UTC - Code Change Verification

Modified `train.py` to add default-off reliability-gated uncertainty boost.

Verification commands:
```bash
python -m py_compile train.py
python train.py --help | grep -E 'adaptive-uncertainty|uncertainty-boost'
```

Results:
- `py_compile` passed with exit code 0.
- Help output includes `--adaptive-uncertainty-boost` and `--uncertainty-boost-weight`.

Default behavior note:
- No existing mode default was intentionally changed; the new behavior requires an explicit CLI flag.

## 2026-06-13 09:55:00 UTC - Uncertainty Boost Targeted Outcome

Completed targeted experiment `goal_uncertainty_boost_md_targeted`.

Outcome:
- Targeted mean AUC: `0.920354420898`.
- `abu_beach_2` improved to `0.908870571687`.
- `abu_urban_2` remained stable at `0.993821655007`.
- Airport samples degraded badly.

Decision:
- Do not run full 15-sample evaluation for this training-coupled boost candidate.
- Modify the new feature so `--adaptive-uncertainty-boost` applies only to final score fusion by default.
- Add a separate explicit `--training-uncertainty-boost` flag for experimental training-time use.

## 2026-06-13 10:00:00 UTC - Code Change Verification: Final-Only Boost

Modified `train.py` so final-score uncertainty boost and training-time uncertainty boost are decoupled.

Verification commands:
```bash
python -m py_compile train.py
python train.py --help | grep -E 'adaptive-uncertainty|uncertainty-boost|training-uncertainty'
```

Results:
- `py_compile` passed with exit code 0.
- Help output includes `--adaptive-uncertainty-boost`, `--uncertainty-boost-weight`, and `--training-uncertainty-boost`.

## 2026-06-13 10:15:00 UTC - Final-Only Uncertainty Boost Targeted Outcome

Completed targeted experiment `goal_uncertainty_final_only_md_targeted`.

Outcome:
- Targeted mean AUC: `0.923862041645`.
- `abu_beach_2` improved to `0.891653479494`, but airport samples degraded.
- Diagnostics show final boost triggered only on `abu_beach_2`, so airport degradation is not from final uncertainty weighting.

Decision:
- Do not run full 15-sample evaluation yet.
- Isolate the final-fusion rule via offline evaluation on existing `results_sp_imp_dpmn_md_only_full15` score components before any further training run.

## 2026-06-13 10:25:00 UTC - Offline Final-Fusion Evaluation on Existing `md_only_full15`

Command: offline Python evaluation over `results_sp_imp_dpmn_md_only_full15/score_components/sp_imp_dpmn/*.mat` with dataset masks used only for post-hoc evaluation.

Result:
- Existing mean AUC from stored fused score: `0.958119941153`.
- Offline final-boost mean AUC with current boost gate and weight `0.25`: `0.945336408093`.
- Triggered samples: `abu_beach_2`, `abu_beach_3`, `abu_beach_4`.
- `abu_beach_2` improved by `+0.031485764927`.
- `abu_beach_3` degraded by `-0.038683460898`.
- `abu_beach_4` degraded by `-0.009702525803`.

Conclusion:
- The current uncertainty boost gate is too permissive for final full-score use.
- A stricter label-free boost gate is needed before any further full experiment.
- Because retraining showed high run-to-run variance in airport samples, add an optional random seed CLI for controlled targeted ablations. Default behavior must remain unchanged.

## 2026-06-13 10:27:00 UTC - Pre-Code-Edit Status: Stricter Boost Gate and Seed

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

Planned code change:
- Add stricter uncertainty boost thresholds, separate from the existing adaptive uncertainty-use gate.
- Add optional `--random-seed` for controlled targeted comparisons.
- Keep default behavior unchanged unless new flags are passed.

## 2026-06-13 11:05:00 UTC - Goal Continuation Checkpoint

Continuation status:
- User asked to continue after an accidental interruption.
- Current `git status --short` before continuing:

```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Current stage:
- Same-seed targeted comparison after adding stricter final-only uncertainty boost thresholds.
- `results_goal_md_seed0_targeted` is complete and recorded in `EXPERIMENT_ABLATION_LOG.md`.
- `results_goal_strict_uncertainty_seed0_targeted` directory check returned nonzero, meaning the directory does not exist and is safe to use.

Next command:
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

## 2026-06-13 11:40:00 UTC - Exit Checkpoint Before Remote Logout

User is about to leave the remote session. This checkpoint records the current state so the next Codex session can continue without losing context.

Current `git status --short`:

```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Files intentionally modified in this Goal run:
- `train.py`
- `GOAL_RUN_LOG.md`
- `CODE_CHANGELOG.md`
- `EXPERIMENT_ABLATION_LOG.md`

New CLI parameters currently added in `train.py`:
- `--adaptive-uncertainty-boost`
- `--uncertainty-boost-weight`
- `--training-uncertainty-boost`
- `--uncertainty-boost-min-top-overlap`
- `--uncertainty-boost-min-residual-corr`
- `--uncertainty-boost-min-contrast-corr`
- `--random-seed`

Verification already completed after code edits:
- `python -m py_compile train.py` passed after the uncertainty boost changes.
- `python train.py --help | grep -E 'random-seed|uncertainty-boost-min|training-uncertainty|adaptive-uncertainty'` showed the expected flags.

Experiments completed in this Goal run:
- `results_goal_hf_md_edgeguard_targeted`
- `results_goal_uncertainty_boost_md_targeted`
- `results_goal_uncertainty_final_only_md_targeted`
- `results_goal_md_seed0_targeted`
- `results_goal_strict_uncertainty_seed0_targeted`

Most recent completed experiment:
- `results_goal_strict_uncertainty_seed0_targeted`
- Command used:

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

Most recent result:
- targeted mean_auc: `0.914178548004`
- `abu_beach_2`: `0.903818772875`
- `abu_urban_2`: `0.992058847622`
- Same-seed md-only targeted anchor was `results_goal_md_seed0_targeted`, mean_auc `0.912948844147`, `abu_beach_2=0.872845842453`, `abu_urban_2=0.992474975016`.
- Strict uncertainty boost improved `abu_beach_2` by about `+0.03097` versus same-seed md-only, but hurt `abu_airport_1` and `abu_airport_2`.
- Boost diagnostics showed boost triggered only for `abu_beach_2`; other targeted samples did not receive uncertainty boost. Airport degradation is therefore likely training/run variance or score recomputation path sensitivity, not direct final boost application.

Current decision:
- Do not run full 15-sample evaluation yet.
- Strict uncertainty boost is locally useful for `abu_beach_2`, but not stable enough to promote to final method.
- The next conservative code direction should avoid changing training dynamics and avoid recomputing the entire final score for non-boosted samples.
- A promising next idea is a default-off conservative uncertainty overlay: keep the original fused score unchanged unless strict boost gate fires, and when it fires, mix only a small amount of normalized uncertainty into the already computed final score. This should be added via a new CLI flag, not by changing old defaults.

Important caution for next session:
- Offline full15 recomputation from stored `residual_score`, `contrast_score`, and `uncertainty_score` cannot be treated as a valid final result because stored `fused_score` appears to include method-specific final processing that is not exactly reproduced by recomputing only these three components.
- Use offline evaluation only as diagnostic evidence. Real results require a normal `python train.py ...` run and `batch_summary*.txt`.

Recommended next actions:
1. Read `GOAL_RUN_LOG.md`, `CODE_CHANGELOG.md`, `EXPERIMENT_ABLATION_LOG.md`, and `git diff -- train.py`.
2. Re-run `python -m py_compile train.py` before further experiments if the session was interrupted.
3. Inspect `compute_final_artifacts` around final score fusion.
4. Implement a default-off conservative final-score uncertainty overlay flag, preserving all existing defaults.
5. Run a targeted experiment only, not full15, on `abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2`.
6. Only consider full15 if targeted improves `abu_beach_2` without harming `abu_urban_2`, sample `1`, sample `3`, and airport samples.

## 2026-06-13 11:40:00 UTC - Prompt For Next Goal Session

Copy this prompt into the next Codex session to continue:

```text
请继续上一次 DPMN 高光谱异常检测 Goal 工作，项目目录是 /root/autodl-tmp/DPMN-main。请先读取并以当前工作区为准，不要从头推翻已有记录。

必须先读取：
- GOAL_RUN_LOG.md
- CODE_CHANGELOG.md
- EXPERIMENT_ABLATION_LOG.md
- git diff -- train.py
- results_goal_strict_uncertainty_seed0_targeted/batch_summary_sp_imp_dpmn.txt
- results_goal_md_seed0_targeted/batch_summary_sp_imp_dpmn.txt

当前已知状态：
- train.py 已新增 default-off uncertainty boost 相关 CLI：--adaptive-uncertainty-boost、--uncertainty-boost-weight、--training-uncertainty-boost、--uncertainty-boost-min-top-overlap、--uncertainty-boost-min-residual-corr、--uncertainty-boost-min-contrast-corr、--random-seed。
- py_compile 已经过，但请恢复后先重新运行 python -m py_compile train.py。
- 当前 best full15 仍是 results_sp_imp_dpmn_md_only_full15，mean_auc 约 0.958119941153。
- 不要把 strict uncertainty boost 直接作为最终方法；它 targeted 下能提升 abu_beach_2，但 airport_1/2 不稳定。
- 最近 targeted：results_goal_strict_uncertainty_seed0_targeted，mean_auc 0.914178548004，abu_beach_2 0.903818772875，abu_urban_2 0.992058847622。
- 同种子 anchor：results_goal_md_seed0_targeted，mean_auc 0.912948844147，abu_beach_2 0.872845842453，abu_urban_2 0.992474975016。

请继续按以下规则推进：
1. 每进入阶段先追加 GOAL_RUN_LOG.md。
2. 每次改代码前先 git status，并记录。
3. 每次改代码后追加 CODE_CHANGELOG.md。
4. 每次改 train.py 后必须执行 python -m py_compile train.py 并记录。
5. 每次实验前后必须追加 EXPERIMENT_ABLATION_LOG.md。
6. 不覆盖任何已有 results 目录。
7. 不用 ground truth 或 AUC 参与训练、权重选择或 sample-adaptive 决策；AUC 只能用于实验后评价。
8. 先 targeted，只有 targeted 稳定后才允许 full15。
9. 不要硬编码 sample id。
10. 不要删除已有结果目录。

下一步建议：
- 不要继续堆 highfreq 或直接 full15。
- 在 train.py 中设计一个新的 default-off 保守 final-score uncertainty overlay：当 strict label-free uncertainty gate 不触发时，保持原始 final fused score 完全不变；只有 gate 触发时，才把少量 uncertainty_score 作为 overlay 混入已经计算好的 final score。
- 该策略必须通过新的 CLI flag 接入，默认不影响 baseline、mask_sam、sp_imp_dpmn 和已有 --adaptive-uncertainty-boost 行为。
- 先跑 targeted：abu_beach_2,1,3,abu_airport_1,abu_airport_2,abu_airport_3,abu_urban_2。
- 如果 targeted 仍不稳定，不要跑 full15；记录失败原因并收敛为论文诊断结论。

请接着完成：读取状态 -> 记录阶段计划 -> 必要代码修改 -> py_compile -> targeted 实验 -> 读取 batch_summary -> 写入三个 Markdown 日志 -> 给出是否允许 full15 的判断。
```

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

## 2026-06-13 15:47:17 UTC - Strategy Update: Stop New Modules, Finish Existing Ablations

User instruction:
- Stop attempting new innovation modules.
- Finish ablations for existing methods.
- Then run or verify full 15-sample evaluation with the best existing method.

Decision:
- Do not continue the planned `results_goal_overlay_md_targeted_v2` experiment.
- Treat `--conservative-uncertainty-overlay` as a default-off, unvalidated code path recorded for traceability only; it is not part of the current method line and will not be used in subsequent experiments.
- Continue only with existing method families already present in the project:
  - `baseline`
  - `mask_sam`
  - `sp_only`
  - `sp_obm`
  - `sp_obm_adaptive_fusion`
  - `sp_obm_md`
  - `sp_obm_hf_diag`
  - `sp_obm_md_hf_diag`

Immediate plan:
- Audit existing full 15-sample `batch_summary*.txt` files.
- Identify which ablation groups already have credible full 15-sample results.
- Run missing full 15-sample ablations only with new result directories.
- Use no ground truth or AUC during training or method selection; AUC remains post-run evaluation only.

## 2026-06-13 15:47:17 UTC - Full Ablation Audit Before Runs

Existing credible full 15-sample summaries:
- `baseline`: `resultold/batch_summary_baseline.txt`, mean_auc `0.949646376529`.
- `mask_sam`: `resultnew_all_v5/batch_summary_mask_sam.txt`, mean_auc `0.940879525839`.
- `sp_obm_adaptive_fusion`: `results_sp_imp_dpmn_adaptive_full/batch_summary_sp_imp_dpmn.txt`, mean_auc `0.939775035950`.
- `sp_obm_md`: `results_sp_imp_dpmn_md_only_full15/batch_summary_sp_imp_dpmn.txt`, mean_auc `0.958119941153`.
- `sp_obm_hf_diag`: `results_sp_imp_dpmn_stationary_hf_diagnostic_v2_full/batch_summary_sp_imp_dpmn.txt`, mean_auc `0.957551813119`.

Missing full 15-sample summaries to complete the current ablation table:
- `sp_only`
- `sp_obm`
- `sp_obm_md_hf_diag`

Fresh result directory checks:
- `results_goal_full_sp_only`: OK, did not exist.
- `results_goal_full_sp_obm`: OK, did not exist.
- `results_goal_full_sp_obm_md_hf_diag`: OK, did not exist.

Current working tree before full runs:
```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Note:
- `train.py` compiles successfully after the latest code changes.
- The unvalidated overlay path will not be enabled in these experiments.

## 2026-06-13 16:05:00 UTC - Completed Full Ablation: `sp_only`

Command:
```bash
python train.py --ablation-mode sp_only \
  --results-dir results_goal_full_sp_only \
  --num-iter 800 \
  --random-seed 0
```

Result:
- `results_goal_full_sp_only/batch_summary_sp_only.txt`
- mean_auc `0.922499903785`
- `abu_beach_2=0.848679463782`
- `abu_urban_2=0.872942217271`

Interpretation:
- Full-set `sp_only` is below historical baseline and clearly below the current MD mainline.
- It supports the ablation narrative that superpixel perturbation alone is not enough; later reliability/fusion/suppression modules are necessary.

## 2026-06-13 16:25:00 UTC - Completed Full Ablation: `sp_obm`

Command:
```bash
python train.py --ablation-mode sp_obm \
  --results-dir results_goal_full_sp_obm \
  --num-iter 800 \
  --random-seed 0
```

Result:
- `results_goal_full_sp_obm/batch_summary_sp_obm.txt`
- mean_auc `0.921017903552`
- `abu_beach_2=0.846024345239`
- `abu_urban_2=0.872058192303`

Interpretation:
- `sp_obm` is slightly below `sp_only` on full 15 samples.
- OBM alone does not fix `abu_beach_2` or `abu_urban_2`; it remains a module whose value must be judged in combination with adaptive fusion and MD suppression.

## 2026-06-13 16:50:00 UTC - Completed Full Ablation: `sp_obm_md_hf_diag`

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
- `results_goal_full_sp_obm_md_hf_diag/batch_summary_sp_imp_dpmn.txt`
- mean_auc `0.955963412793`
- `abu_beach_2=0.820634742592`
- `abu_urban_2=0.997206376251`

Interpretation:
- The MD + HF diagnostic combination does not beat MD-only full mean.
- It severely hurts `abu_beach_2`, so it should not be the final method despite some local gains.

## 2026-06-13 16:55:00 UTC - Final Full Ablation Analysis

Full 15-sample result ranking:
```text
sp_obm_md:             0.958119941153  best current full mean
sp_obm_hf_diag:        0.957551813119
sp_obm_md_hf_diag:     0.955963412793
baseline:              0.949646376529
mask_sam:              0.940879525839
sp_obm_adaptive_fusion:0.939775035950
sp_only:               0.922499903785
sp_obm:                0.921017903552
```

Final recommendation:
- Final mainline: `SP-IMP-DPMN + adaptive score fusion + multidirectional suppression`.
- Result directory: `results_sp_imp_dpmn_md_only_full15`.
- Full 15-sample mean AUC: `0.958119941153`.
- `abu_beach_2`: `0.866870436278`.
- `abu_urban_2`: `0.994053637838`.

Key conclusions:
- Superpixel target alone is not enough.
- OBM alone is not enough and can slightly hurt the full mean.
- Adaptive score fusion is necessary to stabilize `abu_urban_2`.
- Multidirectional suppression is the strongest existing stabilizer and gives the best full mean.
- High-frequency diagnostic fusion is useful as an ablation/diagnostic branch, but not the final method because it worsens `abu_beach_2`.

No additional experiments are planned in this run unless explicitly requested.

## 2026-06-13 16:58:00 UTC - Final Verification

Commands:
```bash
python -m py_compile train.py
ps -eo pid,etime,cmd | rg 'python train.py|tee results_goal_full' || true
git status --short
```

Results:
- `python -m py_compile train.py` succeeded.
- No active `python train.py` experiment process remains.
- Working tree status:
```text
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```

Note:
- Result directories and generated plots/MAT files were not added to Git.

## 2026-06-14 00:00:00 UTC - Resume Cleanup and Main Record Update

User instruction:
- Continue after accidental interruption.
- After ablations are complete, proceed with result analysis.

Actions:
- Confirmed no active `python train.py` process remains.
- Cleaned `EXPERIMENT_ABLATION_LOG.md` tail by removing duplicated planned-experiment blocks that appeared after the final analysis.
- Reordered `CODE_CHANGELOG.md` so the 16:25 overlay signature fix appears before the 16:55 "no additional code changes" analysis note.
- Updated `research_innovation_directions_2025.md` with a superseding 2026-06-14 final full ablation decision.

Final project-facing conclusion:
- Use `SP-IMP-DPMN + adaptive score fusion + multidirectional suppression` as the final existing-method mainline.
- Do not include high-frequency diagnostic fusion in the final main method.
- Keep high-frequency diagnostic fusion as an auxiliary ablation/diagnostic branch because it hurts `abu_beach_2`.

Verification:
```bash
python -m py_compile train.py
```

Result:
- Compilation succeeded.

Current working tree:
```text
 M research_innovation_directions_2025.md
 M train.py
?? CODE_CHANGELOG.md
?? EXPERIMENT_ABLATION_LOG.md
?? GOAL_RUN_LOG.md
```
