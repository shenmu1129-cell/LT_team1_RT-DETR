#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-3}"
CONFIG="${CONFIG:-rtdetr_paper_experiments/configs/attacks_cctsdb.yaml}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-rtdetr_paper_experiments/results/table1_cctsdb_baselines}"
ATTACKS="${ATTACKS:-TOG,Daedalus,OSFD}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-960}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORCE="${FORCE:-0}"
LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/eval_baselines_cctsdb_$(date +%Y%m%d_%H%M%S).log}"

ARGS=(
  --config "${CONFIG}"
  --output-prefix "${OUTPUT_PREFIX}"
  --gpu "${GPU_ID}"
  --target-detector RT-DETR
  --attacks "${ATTACKS}"
  --batch "${BATCH}"
  --imgsz "${IMGSZ}"
  --max-samples "${MAX_SAMPLES}"
  --skip-clean-val
  --skip-paired-asr
)

if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi

nohup python rtdetr_paper_experiments/tools/eval_attack_tables.py "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &

echo "Started resumable RT-DETR baseline attack evaluation on CCTSDB."
echo "PID: $!"
echo "GPU_ID: ${GPU_ID}"
echo "Attacks: ${ATTACKS}"
echo "Log: ${LOG_FILE}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Resume cache: rtdetr_paper_experiments/results/attack_cache"
echo "Watch: tail -f ${LOG_FILE}"
