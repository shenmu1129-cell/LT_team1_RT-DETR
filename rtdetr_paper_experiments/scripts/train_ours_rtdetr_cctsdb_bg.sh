#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-2}"
DATA="${DATA:-data/cctsdb.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_cctsdb/weights/best.pt}"
OUTPUT="${OUTPUT:-rtdetr_paper_experiments/runs/ours_rtdetr_cctsdb}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-4}"
EPS="${EPS:-0.031372549}"
LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/train_ours_rtdetr_cctsdb_$(date +%Y%m%d_%H%M%S).log}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/train_ours_rtdetr_advgan.py \
  --data "${DATA}" \
  --weights "${WEIGHTS}" \
  --output "${OUTPUT}" \
  --split train \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMGSZ}" \
  --workers "${WORKERS}" \
  --eps "${EPS}" > "${LOG_FILE}" 2>&1 &

echo "Started Ours/AdvGAN-AdaAD RT-DETR training on CCTSDB."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
echo "Watch: tail -f ${LOG_FILE}"
