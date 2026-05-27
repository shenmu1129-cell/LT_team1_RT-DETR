#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
CONFIG="${CONFIG:-rtdetr_paper_experiments/configs/attacks_cctsdb.yaml}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rtdetr_paper_experiments/results/table1_cctsdb}"

python rtdetr_paper_experiments/tools/eval_attack_tables.py \
  --config "${CONFIG}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --gpu "${GPU_ID}" \
  --target-detector RT-DETR
