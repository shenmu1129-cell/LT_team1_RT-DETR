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
LR_G="${LR_G:-5e-5}"
LR_D="${LR_D:-5e-5}"
ALPHA_DET="${ALPHA_DET:-10.0}"
ALPHA_SMOOTH="${ALPHA_SMOOTH:-0.001}"
ALPHA_GAN="${ALPHA_GAN:-0.02}"
ALPHA_TRAJ="${ALPHA_TRAJ:-2.0}"
ADAAD_STEPS="${ADAAD_STEPS:-5}"
EVAL_SAMPLES="${EVAL_SAMPLES:-64}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.5}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/train_ours_rtdetr_tt100k_$(date +%Y%m%d_%H%M%S).log}"

ARGS=(
  --data "${DATA}" \
  --weights "${WEIGHTS}" \
  --output "${OUTPUT}" \
  --split train \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMGSZ}" \
  --workers "${WORKERS}" \
  --eps "${EPS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --lr-g "${LR_G}" \
  --lr-d "${LR_D}" \
  --alpha-det "${ALPHA_DET}" \
  --alpha-smooth "${ALPHA_SMOOTH}" \
  --alpha-gan "${ALPHA_GAN}" \
  --alpha-traj "${ALPHA_TRAJ}" \
  --adaad-steps "${ADAAD_STEPS}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --conf "${CONF}" \
  --iou "${IOU}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/train_ours_rtdetr_advgan.py "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started Ours/AdvGAN-AdaAD RT-DETR training on TT100K."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
echo "Max train samples per epoch: ${MAX_TRAIN_SAMPLES}"
echo "LR_G: ${LR_G}, LR_D: ${LR_D}, ALPHA_DET: ${ALPHA_DET}, ADAAD_STEPS: ${ADAAD_STEPS}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
fi
echo "Watch: tail -f ${LOG_FILE}"
