#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-2}"
DATA="${DATA:-data/tt100k.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_tt100k/weights/best.pt}"
OUTPUT="${OUTPUT:-rtdetr_paper_experiments/runs/ours_rtdetr_tt100k}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-4}"
EPS="${EPS:-0.031372549}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2000}"
LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/train_ours_rtdetr_tt100k_$(date +%Y%m%d_%H%M%S).log}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/train_ours_rtdetr_advgan.py \
  --data "${DATA}" \
  --weights "${WEIGHTS}" \
  --output "${OUTPUT}" \
  --split train \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMGSZ}" \
  --workers "${WORKERS}" \
  --eps "${EPS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" > "${LOG_FILE}" 2>&1 &

echo "Started Ours/AdvGAN-AdaAD RT-DETR training on TT100K."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
echo "Max train samples per epoch: ${MAX_TRAIN_SAMPLES}"
echo "Watch: tail -f ${LOG_FILE}"
