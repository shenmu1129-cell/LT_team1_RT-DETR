#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-5}"
ATTACK="${ATTACK:-tog}"  # tog, daedalus, osfd
DATA="${DATA:-data/tt100k.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_tt100k/weights/best.pt}"
OUTPUT="${OUTPUT:-rtdetr_paper_experiments/adv_outputs/${ATTACK}/tt100k_whitebox}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-2}"
STEPS="${STEPS:-10}"
EPS="${EPS:-0.031372549}"
STEP_SIZE="${STEP_SIZE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/generate_${ATTACK}_rtdetr_tt100k_$(date +%Y%m%d_%H%M%S).log}"

ARGS=(
  --attack "${ATTACK}"
  --data "${DATA}"
  --weights "${WEIGHTS}"
  --split test
  --output "${OUTPUT}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --steps "${STEPS}"
  --eps "${EPS}"
  --step-size "${STEP_SIZE}"
  --max-samples "${MAX_SAMPLES}"
  --device 0
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/generate_rtdetr_baseline_adv.py "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started RT-DETR white-box baseline generation on TT100K."
echo "PID: $!"
echo "Attack: ${ATTACK}"
echo "GPU_ID: ${GPU_ID}"
echo "Output: ${OUTPUT}"
echo "Log: ${LOG_FILE}"
echo "Resumable: existing output images are skipped unless OVERWRITE=1."
echo "Watch: tail -f ${LOG_FILE}"
