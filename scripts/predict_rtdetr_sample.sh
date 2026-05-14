#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
WEIGHTS="${WEIGHTS:-outputs/rtdetr_cctsdb/weights/best.pt}"
SOURCE="${SOURCE:-/home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021/images/test}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"
PROJECT="${PROJECT:-outputs}"
NAME="${NAME:-rtdetr_predict_sample}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
WEIGHTS="${WEIGHTS}" \
SOURCE="${SOURCE}" \
IMGSZ="${IMGSZ}" \
CONF="${CONF}" \
PROJECT="${PROJECT}" \
NAME="${NAME}" \
python - <<'PY'
import os
from ultralytics import RTDETR

weights = os.environ["WEIGHTS"]
source = os.environ["SOURCE"]
imgsz = int(os.environ["IMGSZ"])
conf = float(os.environ["CONF"])
project = os.environ["PROJECT"]
name = os.environ["NAME"]

model = RTDETR(weights)
model.predict(
    source=source,
    imgsz=imgsz,
    conf=conf,
    device=0,
    project=project,
    name=name,
    save=True,
    exist_ok=True,
)
print(f"Predictions saved to: {project}/{name}")
PY
