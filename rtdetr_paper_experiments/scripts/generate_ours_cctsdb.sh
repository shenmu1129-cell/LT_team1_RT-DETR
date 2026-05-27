#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA="${DATA:-data/cctsdb.yaml}"
GENERATOR="${GENERATOR:-rtdetr_paper_experiments/runs/ours_rtdetr_cctsdb/weights/netG_best_asr.pth}"
OUTPUT="${OUTPUT:-rtdetr_paper_experiments/adv_outputs/ours/cctsdb}"
IMGSZ="${IMGSZ:-640}"
EPS="${EPS:-0.031372549}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/generate_ours_adv_images.py \
  --data "${DATA}" \
  --split test \
  --generator "${GENERATOR}" \
  --output "${OUTPUT}" \
  --imgsz "${IMGSZ}" \
  --eps "${EPS}" \
  --max-samples "${MAX_SAMPLES}" \
  --device 0
