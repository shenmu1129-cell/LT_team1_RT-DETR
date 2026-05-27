#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-3}"
DATA="${DATA:-data/cctsdb.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_cctsdb/weights/best.pt}"
OUTPUT="${OUTPUT:-rtdetr_paper_experiments/runs/ours_rtdetr_cctsdb}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${OUTPUT}/weights/checkpoint_latest.pth}"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2000}"

EPS="${EPS:-0.031372549}"
LR_G="${LR_G:-0.001}"
LR_D="${LR_D:-0.001}"
ALPHA_DET="${ALPHA_DET:-20.0}"
ALPHA_SMOOTH="${ALPHA_SMOOTH:-0.001}"
ALPHA_GAN="${ALPHA_GAN:-0.02}"
ALPHA_TRAJ="${ALPHA_TRAJ:-2.0}"
ADAAD_STEPS="${ADAAD_STEPS:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-64}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.5}"

LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/resume_ours_rtdetr_cctsdb_lr001_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  echo "请先确认已经拉取了新版代码，并且之前的训练至少跑完 1 个 epoch。" >&2
  exit 1
fi

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
  --resume-checkpoint "${RESUME_CHECKPOINT}"
)

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/train_ours_rtdetr_advgan.py "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &

echo "Resumed Ours/AdvGAN-AdaAD RT-DETR training on CCTSDB."
echo "PID: $!"
echo "GPU_ID: ${GPU_ID}"
echo "Log: ${LOG_FILE}"
echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
echo "Target total epochs: ${EPOCHS}"
echo "Batch: ${BATCH}, imgsz: ${IMGSZ}, workers: ${WORKERS}, max samples: ${MAX_TRAIN_SAMPLES}"
echo "LR_G: ${LR_G}, LR_D: ${LR_D}, ALPHA_DET: ${ALPHA_DET}, ADAAD_STEPS: ${ADAAD_STEPS}"
echo "Watch: tail -f ${LOG_FILE}"
