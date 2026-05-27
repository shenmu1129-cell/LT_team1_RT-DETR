#!/usr/bin/env python3
"""Evaluate RT-DETR on matched clean/adversarial YOLO-format samples."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - ultralytics normally installs PyYAML
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - ultralytics normally installs Pillow
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc

from check_yolo_dataset import (
    list_images,
    load_yaml,
    resolve_image_dir,
    resolve_label_dir,
)
from val_rtdetr import extract_metrics, resolve_weights_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RT-DETR clean/adv mAP50, recall, and ASR."
    )
    parser.add_argument("--weights", required=True, help="Path to RT-DETR best.pt/last.pt.")
    parser.add_argument("--data", required=True, help="Clean YOLO dataset yaml.")
    parser.add_argument("--adv-images", required=True, help="Adversarial image directory.")
    parser.add_argument(
        "--adv-strip-suffix",
        default="_adv",
        help="Suffix to strip from adversarial filename stems before matching clean files.",
    )
    parser.add_argument(
        "--adv-labels",
        default=None,
        help="Adversarial label directory. If omitted, clean labels are reused.",
    )
    parser.add_argument("--clean-images", default=None, help="Optional clean image dir override.")
    parser.add_argument("--clean-labels", default=None, help="Optional clean label dir override.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Clean split used for matching.",
    )
    parser.add_argument("--max-samples", type=int, default=200, help="Max matched samples.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for paired object-level ASR.",
    )
    parser.add_argument(
        "--pred-iou",
        type=float,
        default=0.7,
        help="NMS IoU used by model.predict for paired object-level ASR.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching detections to GT in paired object-level ASR.",
    )
    parser.add_argument("--project", default="outputs", help="Output project directory.")
    parser.add_argument("--name", default="rtdetr_clean_adv_eval", help="Evaluation run name.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--source-model", default="Unknown", help="Attack/source model name.")
    parser.add_argument("--target-detector", default="RT-DETR", help="Target detector name.")
    return parser.parse_args()


def read_names(data: Dict[str, Any]) -> List[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(item) for item in names]
    raise ValueError("Dataset yaml must contain names as a list or dict.")


def label_map(label_dir: Path) -> Dict[str, Path]:
    if not label_dir.is_dir():
        return {}
    return {path.stem: path for path in sorted(label_dir.rglob("*.txt"))}


def image_map(image_dir: Path) -> Dict[str, Path]:
    return {path.stem: path for path in list_images(image_dir)}


def normalize_adv_stem(stem: str, strip_suffix: str) -> str:
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def normalized_image_map(image_dir: Path, strip_suffix: str) -> Dict[str, Path]:
    mapped: Dict[str, Path] = {}
    for path in list_images(image_dir):
        mapped[normalize_adv_stem(path.stem, strip_suffix)] = path
    return mapped


def normalized_label_map(label_dir: Optional[Path], strip_suffix: str) -> Dict[str, Path]:
    if label_dir is None or not label_dir.is_dir():
        return {}
    mapped: Dict[str, Path] = {}
    for path in sorted(label_dir.rglob("*.txt")):
        mapped[normalize_adv_stem(path.stem, strip_suffix)] = path
    return mapped


def count_gt_boxes(label_files: List[Path]) -> int:
    total = 0
    for label_file in label_files:
        if not label_file.exists():
            continue
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                total += 1
    return total


def yolo_to_xyxy(line: str, image_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    parts = line.split()
    if len(parts) != 5:
        return None
    cls, x_center, y_center, width, height = parts
    img_w, img_h = image_size
    cls_id = int(float(cls))
    x_center = float(x_center) * img_w
    y_center = float(y_center) * img_h
    width = float(width) * img_w
    height = float(height) * img_h
    return {
        "cls": cls_id,
        "box": [
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
        ],
    }


def read_gt_objects(label_file: Path, image_file: Path) -> List[Dict[str, Any]]:
    if not label_file.exists():
        return []
    with Image.open(image_file) as image:
        image_size = image.size
    objects: List[Dict[str, Any]] = []
    for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = yolo_to_xyxy(line, image_size)
        if obj is not None:
            objects.append(obj)
    return objects


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def has_matching_detection(
    gt_object: Dict[str, Any],
    detections: List[Dict[str, Any]],
    match_iou: float,
) -> bool:
    for detection in detections:
        if detection["cls"] != gt_object["cls"]:
            continue
        if box_iou(gt_object["box"], detection["box"]) >= match_iou:
            return True
    return False


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_eval_dataset(
    root: Path,
    samples: List[Dict[str, Path]],
    image_key: str,
    label_key: str,
    nc: int,
    names: List[str],
) -> Path:
    if root.exists():
        shutil.rmtree(root)
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        src_image = sample[image_key]
        src_label = sample[label_key]
        image_name = f"{index:06d}_{src_image.name}"
        dst_image = image_dir / image_name
        dst_label = label_dir / f"{dst_image.stem}.txt"
        symlink_or_copy(src_image, dst_image)
        shutil.copy2(src_label, dst_label)
        sample[f"{image_key}_eval"] = dst_image
        sample[f"{label_key}_eval"] = dst_label

    yaml_path = root / "data.yaml"
    yaml_data = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "nc": nc,
        "names": names,
    }
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path


def run_val(model, data_yaml: Path, args: argparse.Namespace, name_suffix: str):
    return model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
        project=args.project,
        name=f"{args.name}_{name_suffix}",
        exist_ok=True,
    )


def format_percent(value: float) -> float:
    return value * 100.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0 else 0.0


def collect_predictions(model, image_dir: Path, args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.pred_iou,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path)
        detections: List[Dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            for box, cls_id, conf in zip(xyxy, classes, confs):
                detections.append(
                    {"box": [float(v) for v in box], "cls": int(cls_id), "conf": float(conf)}
                )
        predictions[image_path.name] = detections
    return predictions


def compute_paired_object_asr(model, samples: List[Dict[str, Path]], args: argparse.Namespace) -> Dict[str, float]:
    clean_dir = samples[0]["clean_image_eval"].parent
    adv_dir = samples[0]["adv_image_eval"].parent
    clean_predictions = collect_predictions(model, clean_dir, args)
    adv_predictions = collect_predictions(model, adv_dir, args)

    clean_detected = 0
    adv_missed = 0
    total_gt = 0
    for sample in samples:
        gt_objects = read_gt_objects(sample["clean_label_eval"], sample["clean_image_eval"])
        total_gt += len(gt_objects)
        clean_dets = clean_predictions.get(sample["clean_image_eval"].name, [])
        adv_dets = adv_predictions.get(sample["adv_image_eval"].name, [])
        for gt_object in gt_objects:
            if not has_matching_detection(gt_object, clean_dets, args.match_iou):
                continue
            clean_detected += 1
            if not has_matching_detection(gt_object, adv_dets, args.match_iou):
                adv_missed += 1

    paired_asr = adv_missed / clean_detected * 100.0 if clean_detected else 0.0
    clean_detect_rate = clean_detected / total_gt * 100.0 if total_gt else 0.0
    return {
        "paired_asr": paired_asr,
        "clean_detected": float(clean_detected),
        "adv_missed": float(adv_missed),
        "total_gt": float(total_gt),
        "clean_detect_rate": clean_detect_rate,
    }


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data)
    data = load_yaml(data_yaml)
    root = Path(str(data["path"])).expanduser()
    warnings: List[str] = []

    clean_image_dir = (
        Path(args.clean_images).expanduser()
        if args.clean_images
        else resolve_image_dir(root, args.split, str(data[args.split]), warnings)
    )
    clean_label_dir = (
        Path(args.clean_labels).expanduser()
        if args.clean_labels
        else resolve_label_dir(root, clean_image_dir, args.split)
    )
    adv_image_dir = Path(args.adv_images).expanduser()
    adv_label_dir: Optional[Path] = Path(args.adv_labels).expanduser() if args.adv_labels else None

    clean_labels = label_map(clean_label_dir)
    clean_images = image_map(clean_image_dir)
    adv_images = normalized_image_map(adv_image_dir, args.adv_strip_suffix)
    adv_labels = normalized_label_map(adv_label_dir, args.adv_strip_suffix)

    samples: List[Dict[str, Path]] = []
    for stem, clean_image in sorted(clean_images.items()):
        if stem not in clean_labels or stem not in adv_images:
            continue
        adv_label = adv_labels.get(stem, clean_labels[stem])
        if not adv_label.exists():
            continue
        samples.append(
            {
                "clean_image": clean_image,
                "clean_label": clean_labels[stem],
                "adv_image": adv_images[stem],
                "adv_label": adv_label,
            }
        )

    if not samples:
        raise SystemExit(
            "No matched clean/adversarial samples found. Check --adv-images, "
            "--adv-strip-suffix, and filenames."
        )

    if args.max_samples > 0 and len(samples) > args.max_samples:
        random.seed(args.seed)
        samples = sorted(random.sample(samples, args.max_samples), key=lambda item: item["clean_image"].name)

    names = read_names(data)
    nc = int(data["nc"])
    work_dir = Path(args.project) / args.name / "matched_eval_data"
    clean_yaml = prepare_eval_dataset(work_dir / "clean", samples, "clean_image", "clean_label", nc, names)
    adv_yaml = prepare_eval_dataset(work_dir / "adv", samples, "adv_image", "adv_label", nc, names)

    weights_path = resolve_weights_path(args.weights, args.project, args.name)
    if not weights_path.exists():
        raise SystemExit(f"Weights file does not exist: {args.weights}")

    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for RT-DETR. Install it with: pip install ultralytics"
        ) from exc

    print(f"Device: cuda" if args.device != "cpu" else "Device: cpu")
    print(f"Classes: {nc} | Clean samples: {len(samples)}")
    print(f"Weights: {weights_path}")
    print(f"Clean images: {clean_image_dir}")
    print(f"Adv images: {adv_image_dir}")
    print(f"Adv strip suffix: {args.adv_strip_suffix!r}")

    model = RTDETR(str(weights_path))
    clean_metrics = extract_metrics(run_val(model, clean_yaml, args, "clean"))
    gt_boxes = count_gt_boxes([sample["clean_label"] for sample in samples])
    clean_map50 = format_percent(clean_metrics["mAP50"])
    clean_map = format_percent(clean_metrics["mAP50-95"])
    clean_precision = format_percent(clean_metrics["precision"])
    clean_recall = format_percent(clean_metrics["recall"])
    clean_f1 = format_percent(
        f1_from_precision_recall(clean_metrics["precision"], clean_metrics["recall"])
    )
    print(
        f"Clean mAP50={clean_map50:.2f}, Clean mAP50-95={clean_map:.2f}, "
        f"Clean Precision={clean_precision:.2f}, Clean Recall={clean_recall:.2f}, "
        f"Clean F1={clean_f1:.2f}, GT boxes={gt_boxes}"
    )

    print(f"Adv samples: {len(samples)} matched from {len(samples)} clean samples")
    adv_metrics = extract_metrics(run_val(model, adv_yaml, args, "adv"))
    adv_map50 = format_percent(adv_metrics["mAP50"])
    adv_map = format_percent(adv_metrics["mAP50-95"])
    adv_precision = format_percent(adv_metrics["precision"])
    adv_recall = format_percent(adv_metrics["recall"])
    adv_f1 = format_percent(
        f1_from_precision_recall(adv_metrics["precision"], adv_metrics["recall"])
    )
    asr = ((clean_recall - adv_recall) / clean_recall * 100.0) if clean_recall > 0 else 0.0
    print(
        f"Adv mAP50={adv_map50:.2f}, Adv mAP50-95={adv_map:.2f}, "
        f"Adv Precision={adv_precision:.2f}, Adv Recall={adv_recall:.2f}, "
        f"Adv F1={adv_f1:.2f}, ASR={asr:.2f}"
    )

    paired = compute_paired_object_asr(model, samples, args)
    print(
        "Paired object ASR="
        f"{paired['paired_asr']:.2f} "
        f"(adv missed {int(paired['adv_missed'])}/"
        f"{int(paired['clean_detected'])} clean-detected GT boxes; "
        f"total GT {int(paired['total_gt'])}; "
        f"conf={args.conf}, match_iou={args.match_iou})"
    )

    table = "\n".join(
        [
            "",
            "| Source Model | Target Detector | Clean mAP50 | Clean mAP50-95 | Clean Recall | Clean F1 | Adv mAP50 | Adv mAP50-95 | Adv Recall | Adv F1 | Recall-drop ASR | Paired Object ASR |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            (
                f"| {args.source_model} | {args.target_detector} | {clean_map50:.1f} | "
                f"{clean_map:.1f} | {clean_recall:.1f} | {clean_f1:.1f} | "
                f"{adv_map50:.1f} | {adv_map:.1f} | {adv_recall:.1f} | {adv_f1:.1f} | "
                f"{asr:.1f} | {paired['paired_asr']:.1f} |"
            ),
            "",
            "| Attack | mAP50 (%) | mAP50-95 (%) | Recall (%) | F1 (%) | ASR (%) |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| {args.source_model} | {adv_map50:.1f} | {adv_map:.1f} | "
                f"{adv_recall:.1f} | {adv_f1:.1f} | {asr:.1f} |"
            ),
        ]
    )
    print(table)

    summary_path = Path(args.project) / args.name / "clean_adv_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(table.strip() + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
