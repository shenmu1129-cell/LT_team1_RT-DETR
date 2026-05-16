#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
DATA="${DATA:-data/cctsdb.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_cctsdb/weights/best.pt}"
ADV_IMAGES="${ADV_IMAGES:-}"
ADV_LABELS="${ADV_LABELS:-}"
ADV_STRIP_SUFFIX="${ADV_STRIP_SUFFIX:-_adv}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
CONF="${CONF:-0.25}"
PRED_IOU="${PRED_IOU:-0.7}"
MATCH_IOU="${MATCH_IOU:-0.5}"
PROJECT="${PROJECT:-outputs}"
NAME="${NAME:-rtdetr_clean_adv_eval}"
SOURCE_MODEL="${SOURCE_MODEL:-YOLOv9}"
TARGET_DETECTOR="${TARGET_DETECTOR:-RT-DETR}"

if [[ -z "${ADV_IMAGES}" ]]; then
  echo "Please set ADV_IMAGES=/path/to/adversarial/images"
  exit 1
fi

ARGS=(
  --weights "${WEIGHTS}"
  --data "${DATA}"
  --adv-images "${ADV_IMAGES}"
  --adv-strip-suffix "${ADV_STRIP_SUFFIX}"
  --max-samples "${MAX_SAMPLES}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --device 0
  --conf "${CONF}"
  --pred-iou "${PRED_IOU}"
  --match-iou "${MATCH_IOU}"
  --project "${PROJECT}"
  --name "${NAME}"
  --source-model "${SOURCE_MODEL}"
  --target-detector "${TARGET_DETECTOR}"
)

if [[ -n "${ADV_LABELS}" ]]; then
  ARGS+=(--adv-labels "${ADV_LABELS}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/eval_rtdetr_clean_adv.py "${ARGS[@]}"
