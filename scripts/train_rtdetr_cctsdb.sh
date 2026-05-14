#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA="${DATA:-data/cctsdb.yaml}"
MODEL="${MODEL:-rtdetr-l.pt}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-8}"
PROJECT="${PROJECT:-outputs}"
NAME="${NAME:-rtdetr_cctsdb}"
PATIENCE="${PATIENCE:-30}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/train_rtdetr.py \
  --data "${DATA}" \
  --model "${MODEL}" \
  --imgsz "${IMGSZ}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device 0 \
  --project "${PROJECT}" \
  --name "${NAME}" \
  --patience "${PATIENCE}"
