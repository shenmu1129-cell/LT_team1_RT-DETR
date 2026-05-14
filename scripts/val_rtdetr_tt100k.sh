#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA="${DATA:-data/tt100k.yaml}"
WEIGHTS="${WEIGHTS:-outputs/rtdetr_tt100k/weights/best.pt}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-4}"
PROJECT="${PROJECT:-outputs}"
NAME="${NAME:-rtdetr_tt100k}"
SPLIT="${SPLIT:-test}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/val_rtdetr.py \
  --weights "${WEIGHTS}" \
  --data "${DATA}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device 0 \
  --split "${SPLIT}" \
  --project "${PROJECT}" \
  --name "${NAME}"
