#!/bin/bash
# Runs predictions sequentially for budget sizes 128, 256, and 512.
# Usage: bash run_budgets.sh [--model MODEL] [--max_length MAX_LEN] [--mode MODE] [--out_prefix PREFIX]

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH=60000
MODE="dyn"
PREFIX="mistral_all_datasets-dyn"  # output name prefix; budget size is appended automatically

for BUDGET in 128 256 512; do
    OUT_NAME="${PREFIX}-budget${BUDGET}"
    echo ""
    echo "========================================"
    echo "  Starting: budget=${BUDGET}  out=${OUT_NAME}"
    echo "========================================"
    python new_pred.py \
        --model_name_or_path "$MODEL" \
        --max_length "$MAX_LENGTH" \
        --out_name "$OUT_NAME" \
        --mode "$MODE" \
        --budget "$BUDGET"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: budget=${BUDGET} failed (exit code: ${EXIT_CODE}). Stopping."
        exit $EXIT_CODE
    fi
    echo "  Completed: budget=${BUDGET}"
done

echo ""
echo "========================================"
echo "  All budget runs completed!"
echo "========================================"
