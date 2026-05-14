#!/usr/bin/env python3
"""Validate or test an Ultralytics RT-DETR checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from check_yolo_dataset import check_dataset, format_result, write_resolved_yaml


def default_name_from_data(data_path: str) -> str:
    stem = Path(data_path).stem.lower()
    if "cctsdb" in stem:
        return "rtdetr_cctsdb"
    if "tt100k" in stem:
        return "rtdetr_tt100k"
    return f"rtdetr_{stem}"


def resolve_weights_path(weights: str, project: str, run_name: str) -> Path:
    """Resolve best.pt/last.pt across Ultralytics save-dir variants."""
    candidates = [
        Path(weights),
        Path(project) / run_name / "weights" / Path(weights).name,
        Path("runs") / "detect" / project / run_name / "weights" / Path(weights).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(weights)


def get_ultralytics_save_dir(model, fallback: Path) -> Path:
    validator = getattr(model, "validator", None)
    save_dir = getattr(validator, "save_dir", None)
    return Path(save_dir) if save_dir else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RT-DETR with Ultralytics.")
    parser.add_argument("--weights", required=True, help="Path to best.pt/last.pt.")
    parser.add_argument("--data", required=True, help="YOLO dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--project", default="outputs", help="Output project directory.")
    parser.add_argument("--name", default=None, help="Run name under project.")
    return parser.parse_args()


def extract_metrics(metrics: Any) -> Dict[str, float]:
    box = getattr(metrics, "box", None)
    values = {
        "precision": float(getattr(box, "mp", 0.0)) if box is not None else 0.0,
        "recall": float(getattr(box, "mr", 0.0)) if box is not None else 0.0,
        "mAP50": float(getattr(box, "map50", 0.0)) if box is not None else 0.0,
        "mAP50-95": float(getattr(box, "map", 0.0)) if box is not None else 0.0,
    }

    results_dict = getattr(metrics, "results_dict", None) or {}
    for key, value in results_dict.items():
        lowered = key.lower()
        if "precision" in lowered and "precision" in values:
            values["precision"] = float(value)
        elif "recall" in lowered and "recall" in values:
            values["recall"] = float(value)
        elif "map50-95" in lowered or "map50_95" in lowered:
            values["mAP50-95"] = float(value)
        elif lowered.endswith("map50") or "map50(b)" in lowered:
            values["mAP50"] = float(value)
    return values


def save_metrics_summary(path: Path, metrics: Dict[str, float], split: str, weights: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"weights: {weights}",
        f"split: {split}",
        f"precision: {metrics['precision']:.6f}",
        f"recall: {metrics['recall']:.6f}",
        f"mAP50: {metrics['mAP50']:.6f}",
        f"mAP50-95: {metrics['mAP50-95']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_name = args.name or default_name_from_data(args.data)
    output_dir = Path(args.project) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checking YOLO dataset before RT-DETR validation...")
    result = check_dataset(args.data, sample_count=20, strict=False)
    print(format_result(result))
    if not result.ok:
        raise SystemExit(
            "Dataset check failed. Fix the errors above before launching RT-DETR validation."
        )

    resolved_yaml = write_resolved_yaml(result, output_dir / "resolved_data.yaml")
    print(f"Using resolved data yaml: {resolved_yaml}")

    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for RT-DETR. Install it with: pip install ultralytics"
        ) from exc

    weights_path = resolve_weights_path(args.weights, args.project, run_name)
    if not weights_path.exists():
        raise SystemExit(f"Weights file does not exist: {args.weights}")

    print(f"Using weights: {weights_path}")
    model = RTDETR(str(weights_path))
    split_used = args.split
    try:
        metrics_obj = model.val(
            data=str(resolved_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            project=args.project,
            name=run_name,
            exist_ok=True,
        )
    except Exception:
        if args.split != "test":
            raise
        print("Validation with split='test' failed; falling back to split='val'.")
        split_used = "val"
        metrics_obj = model.val(
            data=str(resolved_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split="val",
            project=args.project,
            name=run_name,
            exist_ok=True,
        )

    metrics = extract_metrics(metrics_obj)
    print("RT-DETR metrics:")
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        print(f"{key}: {metrics[key]:.6f}")

    save_dir = get_ultralytics_save_dir(model, output_dir)
    summary_path = save_dir / "metrics_summary.txt"
    save_metrics_summary(summary_path, metrics, split_used, str(weights_path))
    print(f"Saved metrics summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
