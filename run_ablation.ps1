$modes = @("baseline", "mask_only", "sam_only", "mask_sam")
$resultsDir = ".\results_ablation"

foreach ($mode in $modes) {
    Write-Host "Running mode: $mode"
    python .\train.py --ablation-mode $mode --results-dir $resultsDir
}
