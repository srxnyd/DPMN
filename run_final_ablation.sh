#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_final_ablation"
SAM_WEIGHT="0.01"

cd "${SCRIPT_DIR}"

for MODE in baseline mask_only sam_only mask_sam; do
  echo "Running mode=${MODE}, sam_weight=${SAM_WEIGHT}"
  python train.py --ablation-mode "${MODE}" --sam-weight "${SAM_WEIGHT}" --results-dir "${RESULTS_DIR}"
done
