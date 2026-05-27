#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-3}"
DATA="${DATA:-data/tt100k.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_tt100k/weights/best.pt}"
OUTPUT="${OUTPUT:-rtdetr_paper_experiments/runs/ours_rtdetr_tt100k}"

EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
# TT100K train has about 6105 images, so 0 means using the full training split.
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"

EPS="${EPS:-0.031372549}"
LR_G="${LR_G:-2e-4}"
LR_D="${LR_D:-5e-5}"
ALPHA_DET="${ALPHA_DET:-30}"
ALPHA_TRAJ="${ALPHA_TRAJ:-0.5}"
ALPHA_GAN="${ALPHA_GAN:-0.005}"
ALPHA_SMOOTH="${ALPHA_SMOOTH:-0.0003}"
ADAAD_STEPS="${ADAAD_STEPS:-10}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.5}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/train_ours_rtdetr_tt100k_strong_$(date +%Y%m%d_%H%M%S).log}"

ARGS=(
  --data "${DATA}"
  --weights "${WEIGHTS}"
  --output "${OUTPUT}"
  --split train
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --imgsz "${IMGSZ}"
  --workers "${WORKERS}"
  --eps "${EPS}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --lr-g "${LR_G}"
  --lr-d "${LR_D}"
  --alpha-det "${ALPHA_DET}"
  --alpha-smooth "${ALPHA_SMOOTH}"
  --alpha-gan "${ALPHA_GAN}"
  --alpha-traj "${ALPHA_TRAJ}"
  --adaad-steps "${ADAAD_STEPS}"
  --eval-samples "${EVAL_SAMPLES}"
  --conf "${CONF}"
  --iou "${IOU}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/train_ours_rtdetr_advgan.py "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started strong Ours/AdvGAN-AdaAD RT-DETR training on TT100K."
echo "PID: $!"
echo "GPU_ID: ${GPU_ID}"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT}"
echo "Max train samples per epoch: ${MAX_TRAIN_SAMPLES} (0 means full split)"
echo "Batch: ${BATCH}, imgsz: ${IMGSZ}, workers: ${WORKERS}"
echo "LR_G: ${LR_G}, LR_D: ${LR_D}, ALPHA_DET: ${ALPHA_DET}, ALPHA_TRAJ: ${ALPHA_TRAJ}"
echo "ALPHA_GAN: ${ALPHA_GAN}, ALPHA_SMOOTH: ${ALPHA_SMOOTH}, ADAAD_STEPS: ${ADAAD_STEPS}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
fi
echo "Watch: tail -f ${LOG_FILE}"
