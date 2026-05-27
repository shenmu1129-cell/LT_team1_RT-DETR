#!/usr/bin/env bash
set -euo pipefail

mkdir -p rtdetr_paper_experiments/logs

GPU_ID="${GPU_ID:-5}"
ATTACKS="${ATTACKS:-tog,daedalus,osfd}"
DATA="${DATA:-data/tt100k.yaml}"
WEIGHTS="${WEIGHTS:-runs/detect/outputs/rtdetr_tt100k/weights/best.pt}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-2}"
STEPS="${STEPS:-10}"
EPS="${EPS:-0.031372549}"
STEP_SIZE="${STEP_SIZE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
LOG_FILE="${LOG_FILE:-rtdetr_paper_experiments/logs/generate_rtdetr_baselines_tt100k_all_$(date +%Y%m%d_%H%M%S).log}"

run_one() {
  local attack="$1"
  local output="rtdetr_paper_experiments/adv_outputs/${attack}/tt100k_whitebox"
  local args=(
    --attack "${attack}"
    --data "${DATA}"
    --weights "${WEIGHTS}"
    --split test
    --output "${output}"
    --imgsz "${IMGSZ}"
    --batch "${BATCH}"
    --steps "${STEPS}"
    --eps "${EPS}"
    --step-size "${STEP_SIZE}"
    --max-samples "${MAX_SAMPLES}"
    --device 0
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi

  echo
  echo "===== Generating ${attack} ====="
  echo "Output: ${output}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python rtdetr_paper_experiments/tools/generate_rtdetr_baseline_adv.py "${args[@]}"
}

(
  IFS=',' read -r -a attack_array <<< "${ATTACKS}"
  for attack in "${attack_array[@]}"; do
    run_one "${attack}"
  done
) > "${LOG_FILE}" 2>&1 &

echo "Started all RT-DETR white-box baseline generation on TT100K."
echo "PID: $!"
echo "Attacks: ${ATTACKS}"
echo "GPU_ID: ${GPU_ID}"
echo "Log: ${LOG_FILE}"
echo "Resumable: existing output images are skipped unless OVERWRITE=1."
echo "Watch: tail -f ${LOG_FILE}"
