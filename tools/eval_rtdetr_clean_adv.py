#!/usr/bin/env python3
"""Evaluate RT-DETR on matched clean/adversarial YOLO-format samples."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - ultralytics normally installs PyYAML
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

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


def count_gt_boxes(label_files: List[Path]) -> int:
    total = 0
    for label_file in label_files:
        if not label_file.exists():
            continue
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                total += 1
    return total


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
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
    adv_images = image_map(adv_image_dir)
    adv_labels = label_map(adv_label_dir) if adv_label_dir else {}

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
            "No matched clean/adversarial samples found. Check --adv-images and filenames."
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

    model = RTDETR(str(weights_path))
    clean_metrics = extract_metrics(run_val(model, clean_yaml, args, "clean"))
    gt_boxes = count_gt_boxes([sample["clean_label"] for sample in samples])
    clean_map50 = format_percent(clean_metrics["mAP50"])
    clean_recall = format_percent(clean_metrics["recall"])
    print(f"Clean mAP50={clean_map50:.2f}, Clean Recall={clean_recall:.2f}, GT boxes={gt_boxes}")

    print(f"Adv samples: {len(samples)} matched from {len(samples)} clean samples")
    adv_metrics = extract_metrics(run_val(model, adv_yaml, args, "adv"))
    adv_map50 = format_percent(adv_metrics["mAP50"])
    adv_recall = format_percent(adv_metrics["recall"])
    asr = ((clean_recall - adv_recall) / clean_recall * 100.0) if clean_recall > 0 else 0.0
    print(f"Adv mAP50={adv_map50:.2f}, Adv Recall={adv_recall:.2f}, ASR={asr:.2f}")

    table = "\n".join(
        [
            "",
            "| Source Model | Target Detector | Clean mAP50 | Adv mAP50 | Clean Recall | Adv Recall | ASR |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            (
                f"| {args.source_model} | {args.target_detector} | {clean_map50:.1f} | "
                f"{adv_map50:.1f} | {clean_recall:.1f} | {adv_recall:.1f} | {asr:.1f} |"
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
