#!/usr/bin/env python3
"""Train Ultralytics RT-DETR on a YOLO-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from check_yolo_dataset import check_dataset, format_result, write_resolved_yaml


def parse_resume(value: str | None):
    if value is None:
        return False
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return value


def default_name_from_data(data_path: str) -> str:
    stem = Path(data_path).stem.lower()
    if "cctsdb" in stem:
        return "rtdetr_cctsdb"
    if "tt100k" in stem:
        return "rtdetr_tt100k"
    return f"rtdetr_{stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR with Ultralytics.")
    parser.add_argument("--data", required=True, help="YOLO dataset yaml.")
    parser.add_argument("--model", default="rtdetr-l.pt", help="RT-DETR checkpoint/model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="outputs", help="Output project directory.")
    parser.add_argument("--name", default=None, help="Run name under project.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="true",
        default=None,
        help="Resume training. Use --resume or --resume path/to/last.pt.",
    )
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_name = args.name or default_name_from_data(args.data)
    output_dir = Path(args.project) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checking YOLO dataset before RT-DETR training...")
    result = check_dataset(args.data, sample_count=20, strict=False)
    print(format_result(result))
    if not result.ok:
        raise SystemExit(
            "Dataset check failed. Fix the errors above before launching RT-DETR training."
        )

    resolved_yaml = write_resolved_yaml(result, output_dir / "resolved_data.yaml")
    print(f"Using resolved data yaml: {resolved_yaml}")

    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for RT-DETR. Install it with: pip install ultralytics"
        ) from exc

    model = RTDETR(args.model)
    model.train(
        data=str(resolved_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=run_name,
        device=args.device,
        patience=args.patience,
        resume=parse_resume(args.resume),
        exist_ok=True,
    )

    best_pt = output_dir / "weights" / "best.pt"
    last_pt = output_dir / "weights" / "last.pt"
    print(f"Training finished. best.pt: {best_pt}")
    print(f"Training finished. last.pt: {last_pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
