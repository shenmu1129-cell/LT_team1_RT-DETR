#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

GPU_ID="${GPU_ID:-2}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
LOG_FILE="${LOG_FILE:-logs/rtdetr_cctsdb_$(date +%Y%m%d_%H%M%S).log}"

nohup env \
  GPU_ID="${GPU_ID}" \
  EPOCHS="${EPOCHS}" \
  BATCH="${BATCH}" \
  IMGSZ="${IMGSZ}" \
  WORKERS="${WORKERS}" \
  bash scripts/train_rtdetr_cctsdb.sh > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "Started RT-DETR CCTSDB training in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Watch with: tail -f ${LOG_FILE}"
