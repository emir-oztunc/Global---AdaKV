#!/bin/bash
# Budget 128, 256, 512 sırasıyla çalıştırır.
# Kullanım: bash run_budgets.sh [--model MODEL] [--max_length MAX_LEN] [--mode MODE] [--out_prefix PREFIX]

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH=60000
MODE="dyn"
PREFIX="mistral_all_datasets-dyn"  # out_name ön eki, budget boyutu otomatik eklenir

for BUDGET in 128 256 512; do
    OUT_NAME="${PREFIX}-budget${BUDGET}"
    echo ""
    echo "========================================"
    echo "  Başlatılıyor: budget=${BUDGET}  out=${OUT_NAME}"
    echo "========================================"
    python new_pred.py \
        --model_name_or_path "$MODEL" \
        --max_length "$MAX_LENGTH" \
        --out_name "$OUT_NAME" \
        --mode "$MODE" \
        --budget "$BUDGET"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "HATA: budget=${BUDGET} çalışırken hata oluştu (exit code: ${EXIT_CODE}). Durduruluyor."
        exit $EXIT_CODE
    fi
    echo "  Tamamlandı: budget=${BUDGET}"
done

echo ""
echo "========================================"
echo "  Tüm budget testleri tamamlandı!"
echo "========================================"
