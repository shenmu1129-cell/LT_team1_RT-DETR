#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"
DATA="${DATA:-data/tt100k.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_tt100k/weights/best.pt}"
GENERATOR="${GENERATOR:-rtdetr_paper_experiments/runs/ours_rtdetr_tt100k/weights/netG_best_asr_79_03.pth}"
ADV_OUTPUT="${ADV_OUTPUT:-rtdetr_paper_experiments/adv_outputs/ours/tt100k_best_asr_79_03}"

IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-4}"
EPS="${EPS:-0.031372549}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONF="${CONF:-0.25}"
PRED_IOU="${PRED_IOU:-0.5}"
MATCH_IOU="${MATCH_IOU:-0.5}"
PROJECT="${PROJECT:-rtdetr_paper_experiments/results/raw_eval}"
NAME="${NAME:-tt100k_ours_best_asr7903}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_CLEAN_VAL="${SKIP_CLEAN_VAL:-1}"
SKIP_PAIRED_ASR="${SKIP_PAIRED_ASR:-1}"

# RT-DETR clean metrics on TT100K test. Override these if you change weights/data/imgsz.
CLEAN_MAP50="${CLEAN_MAP50:-82.09}"
CLEAN_MAP50_95="${CLEAN_MAP50_95:-62.97}"
CLEAN_PRECISION="${CLEAN_PRECISION:-84.08}"
CLEAN_RECALL="${CLEAN_RECALL:-80.35}"
CLEAN_F1="${CLEAN_F1:-82.17}"

if [[ ! -f "${GENERATOR}" ]]; then
  echo "Generator file does not exist: ${GENERATOR}" >&2
  exit 1
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "RT-DETR weights file does not exist: ${WEIGHTS}" >&2
  exit 1
fi

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  echo "Generating TT100K adversarial images with: ${GENERATOR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/generate_ours_adv_images.py \
    --data "${DATA}" \
    --split test \
    --generator "${GENERATOR}" \
    --output "${ADV_OUTPUT}" \
    --imgsz "${IMGSZ}" \
    --eps "${EPS}" \
    --max-samples "${MAX_SAMPLES}" \
    --device 0
else
  echo "SKIP_GENERATE=1, using existing adversarial images: ${ADV_OUTPUT}/images"
fi

echo "Evaluating RT-DETR on clean/adversarial TT100K samples..."
ARGS=(
  --weights "${WEIGHTS}" \
  --data "${DATA}" \
  --adv-images "${ADV_OUTPUT}/images" \
  --adv-labels "${ADV_OUTPUT}/labels" \
  --max-samples "${MAX_SAMPLES}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device 0 \
  --conf "${CONF}" \
  --pred-iou "${PRED_IOU}" \
  --match-iou "${MATCH_IOU}" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  --source-model "Ours-AdvGAN-AdaAD" \
  --target-detector "RT-DETR"
)

if [[ "${SKIP_CLEAN_VAL}" == "1" ]]; then
  ARGS+=(
    --skip-clean-val
    --clean-map50 "${CLEAN_MAP50}"
    --clean-map50-95 "${CLEAN_MAP50_95}"
    --clean-precision "${CLEAN_PRECISION}"
    --clean-recall "${CLEAN_RECALL}"
    --clean-f1 "${CLEAN_F1}"
  )
fi

if [[ "${SKIP_PAIRED_ASR}" == "1" ]]; then
  ARGS+=(--skip-paired-asr)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/eval_rtdetr_clean_adv.py "${ARGS[@]}"
