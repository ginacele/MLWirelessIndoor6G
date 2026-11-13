#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="${1:-}"
[ -n "$LOGDIR" ] && mkdir -p "$LOGDIR"

STEPS=(
    pool_data
    offset_diagnosis
    fit_models
    a_vs_e
    baseline_fits
    transfer
    scenarios
    ml_models
    local_mean
    coverage
    qfunction
    sse_los_nlos
)

for s in "${STEPS[@]}"; do
    echo "=============================================================="
    echo ">>> $s.py"
    echo "=============================================================="
    if [ -n "$LOGDIR" ]; then
        python3 "scripts/$s.py" 2>&1 | tee "$LOGDIR/$s.txt"
    else
        python3 "scripts/$s.py"
    fi
    echo
done

echo "All ${#STEPS[@]} scripts completed."
