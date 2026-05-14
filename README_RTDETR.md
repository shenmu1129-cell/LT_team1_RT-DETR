# RT-DETR Training With Ultralytics

This project uses Ultralytics RT-DETR as the fourth detector for cross-detector generalization experiments.

## Why RT-DETR Instead of ATSS/MMDetection

ATSS was originally considered, but the current server setup makes the OpenMMLab route expensive and fragile:

- `mmcv` compilation failed.
- Disk space is limited, so creating another conda environment is undesirable.
- This project should avoid OpenMMLab, MMDetection, `mmcv`, and `mmengine`.

The fourth detector is therefore changed to Ultralytics RT-DETR, for example `rtdetr-l.pt`, using the existing YOLO-format datasets directly.

## Detector Set

Final cross-detector combination:

- Faster R-CNN
- RetinaNet
- FCOS
- RT-DETR

The four models cover complementary detection paradigms:

- Faster R-CNN: two-stage detector
- RetinaNet: anchor-based one-stage detector
- FCOS: anchor-free one-stage detector
- RT-DETR: transformer-based detector

Suggested paper wording:

> To evaluate cross-detector generalization, we select Faster R-CNN, RetinaNet, FCOS, and RT-DETR, covering two-stage, anchor-based one-stage, anchor-free one-stage, and transformer-based detection paradigms.

## Files Added

- `data/cctsdb.yaml`
- `data/tt100k.yaml`
- `tools/check_yolo_dataset.py`
- `tools/train_rtdetr.py`
- `tools/val_rtdetr.py`
- `scripts/train_rtdetr_cctsdb.sh`
- `scripts/train_rtdetr_tt100k.sh`
- `scripts/val_rtdetr_cctsdb.sh`
- `scripts/val_rtdetr_tt100k.sh`
- `scripts/predict_rtdetr_sample.sh`

No Faster R-CNN, RetinaNet, or FCOS training files are modified.

## Important TT100K Note

`data/tt100k.yaml` currently contains `TODO_CLASS_0` to `TODO_CLASS_44` placeholders because no existing TT100K yaml or `classes.txt` file was found in this checkout.

Before training or reporting metrics on TT100K, replace those 45 names with the real TT100K class names in exactly the same order as the YOLO label class ids. Otherwise class-wise metrics and result interpretation will be wrong.

The checker and training scripts intentionally fail while TODO class names remain.

## Dataset Check

Run the checker before formal training:

```bash
python tools/check_yolo_dataset.py --data data/cctsdb.yaml
python tools/check_yolo_dataset.py --data data/tt100k.yaml
```

The checker validates:

- yaml existence and required keys
- `path`, `train`, `val`, and `test` image paths
- corresponding YOLO label directories
- image and label counts
- YOLO label format: `class x_center y_center width height`
- class id range `[0, nc-1]`
- empty label files as warnings, not hard errors
- per-split image counts, label counts, and class statistics

For TT100K, the scripts automatically detect either of these layouts:

- `train/images`, `test/images`, `train/labels`, `test/labels`
- `images/train`, `images/test`, `labels/train`, `labels/test`

## 1 Epoch Debug

```bash
GPU_ID=2 EPOCHS=1 BATCH=2 bash scripts/train_rtdetr_cctsdb.sh
```

## Formal Training

CCTSDB:

```bash
GPU_ID=2 bash scripts/train_rtdetr_cctsdb.sh
```

TT100K:

```bash
GPU_ID=2 bash scripts/train_rtdetr_tt100k.sh
```

Default output directories:

- `outputs/rtdetr_cctsdb`
- `outputs/rtdetr_tt100k`

Each run writes a resolved data yaml to the output directory and saves:

- `outputs/rtdetr_cctsdb/weights/best.pt`
- `outputs/rtdetr_cctsdb/weights/last.pt`
- `outputs/rtdetr_tt100k/weights/best.pt`
- `outputs/rtdetr_tt100k/weights/last.pt`

## Validation

CCTSDB:

```bash
GPU_ID=2 bash scripts/val_rtdetr_cctsdb.sh
```

TT100K:

```bash
GPU_ID=2 bash scripts/val_rtdetr_tt100k.sh
```

Validation prints precision, recall, mAP50, and mAP50-95. It also saves:

- `outputs/rtdetr_cctsdb/metrics_summary.txt`
- `outputs/rtdetr_tt100k/metrics_summary.txt`

## Sample Prediction

```bash
GPU_ID=2 bash scripts/predict_rtdetr_sample.sh
```

Useful overrides:

```bash
GPU_ID=2 WEIGHTS=outputs/rtdetr_tt100k/weights/best.pt SOURCE=/path/to/images bash scripts/predict_rtdetr_sample.sh
```

## If GPU Memory Is Not Enough

Use smaller settings:

```bash
GPU_ID=2 BATCH=2 IMGSZ=512 WORKERS=4 bash scripts/train_rtdetr_cctsdb.sh
```

## If TT100K Small-Object Performance Is Weak

Try a larger image size and longer training:

```bash
GPU_ID=2 IMGSZ=800 BATCH=2 EPOCHS=100 bash scripts/train_rtdetr_tt100k.sh
```

## Dependencies

Use the existing `wwt310` environment if possible:

```bash
conda activate wwt310
pip install ultralytics
```

No OpenMMLab, MMDetection, `mmcv`, or `mmengine` dependency is used.
