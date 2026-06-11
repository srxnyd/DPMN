$ErrorActionPreference = "Stop"

$improvedRoot = $PSScriptRoot
$workspaceRoot = Split-Path (Split-Path (Split-Path $improvedRoot -Parent) -Parent) -Parent
$baselineRoot = Join-Path $workspaceRoot "DPMN-main\DPMNold\DPMN-main"

$baselineResultDir = Join-Path $baselineRoot "resultold"
$improvedResultDir = Join-Path $improvedRoot "resultfinal_v3vis"
$comparisonDir = Join-Path $improvedRoot "ppt_visuals"

Push-Location $baselineRoot
python train_batch_baseline.py --dataset-dir "./dataset" --results-dir "./resultold"
Pop-Location

Push-Location $improvedRoot
python train.py --dataset-dir "./dataset" --dataset-prefix urban --normalize-inputs --results-dir "./resultfinal_v3vis" --ablation-mode mask_sam

python make_ppt_visuals.py `
  --baseline-csv "$baselineResultDir\batch_results_baseline.csv" `
  --improved-csv "$improvedResultDir\batch_results_mask_sam.csv" `
  --baseline-dataset-dir "$baselineRoot\dataset" `
  --improved-dataset-dir "$improvedRoot\dataset" `
  --baseline-detection-dir "$baselineRoot\dataset\urban_detection" `
  --improved-detection-dir "$improvedRoot\dataset\urban_detection" `
  --output-dir $comparisonDir `
  --dataset-prefix urban `
  --exclude-samples abu_urban_5
Pop-Location
