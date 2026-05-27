#!/usr/bin/env python3
"""Run RT-DETR clean/adv evaluation for multiple attacks and write paper tables."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate attacks and generate RT-DETR table.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--source-model", default="")
    parser.add_argument("--target-detector", default="RT-DETR")
    parser.add_argument("--attacks", default="", help="Comma-separated attack names to run.")
    parser.add_argument("--force", action="store_true", help="Re-run attacks even if cached results exist.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--pred-iou", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=None)
    parser.add_argument("--skip-clean-val", action="store_true")
    parser.add_argument("--run-clean-val", action="store_true")
    parser.add_argument("--skip-paired-asr", action="store_true")
    parser.add_argument("--run-paired-asr", action="store_true")
    return parser.parse_args()


def cfg_value(cfg: Dict[str, Any], args: argparse.Namespace, key: str, default: Any) -> Any:
    value = getattr(args, key.replace("-", "_"), None)
    return cfg.get(key, default) if value is None else value


def selected_attacks(args: argparse.Namespace) -> Optional[set[str]]:
    if not args.attacks:
        return None
    return {item.strip() for item in args.attacks.split(",") if item.strip()}


def cache_path_for(output_prefix: Path, dataset: str, name: str) -> Path:
    return output_prefix.parent / "attack_cache" / f"{dataset}_{name.lower()}.yaml"


def log_path_for(output_prefix: Path, dataset: str, name: str) -> Path:
    return output_prefix.parent / "attack_logs" / f"{dataset}_{name.lower()}.log"


def read_cached_result(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_cached_result(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(row, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_and_log(cmd: List[str], env: Dict[str, str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    collected: List[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                collected.append(line)
        return_code = proc.wait()
    return return_code, "".join(collected)


def run_one_attack(
    name: str,
    attack: Dict[str, Any],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    output_prefix: Path,
):
    cache_path = cache_path_for(output_prefix, str(cfg["dataset"]), name)
    if cache_path.exists() and not args.force:
        print(f"[RESUME] {name}: using cached result: {cache_path}")
        return read_cached_result(cache_path)

    images = attack.get("images")
    labels = attack.get("labels")
    if not images or not Path(str(images)).expanduser().exists():
        print(f"[SKIP] {name}: adversarial image directory missing: {images}")
        return None

    clean_metrics = cfg.get("clean_metrics") or {}
    skip_clean_val = bool(cfg.get("skip_clean_val", False)) or args.skip_clean_val
    skip_paired_asr = bool(cfg.get("skip_paired_asr", False)) or args.skip_paired_asr
    if args.run_clean_val:
        skip_clean_val = False
    if args.run_paired_asr:
        skip_paired_asr = False

    cmd = [
        sys.executable,
        "tools/eval_rtdetr_clean_adv.py",
        "--weights",
        str(cfg["weights"]),
        "--data",
        str(cfg["data"]),
        "--adv-images",
        str(images),
        "--max-samples",
        str(cfg_value(cfg, args, "max_samples", 0)),
        "--imgsz",
        str(cfg_value(cfg, args, "imgsz", 960)),
        "--batch",
        str(cfg_value(cfg, args, "batch", 2)),
        "--device",
        "0",
        "--project",
        "rtdetr_paper_experiments/results/raw_eval",
        "--name",
        f"{cfg['dataset']}_{name.lower()}",
        "--source-model",
        args.source_model or name,
        "--target-detector",
        args.target_detector,
        "--conf",
        str(cfg_value(cfg, args, "conf", 0.35)),
        "--pred-iou",
        str(cfg_value(cfg, args, "pred_iou", 0.7)),
        "--match-iou",
        str(cfg_value(cfg, args, "match_iou", 0.5)),
    ]
    if labels:
        cmd.extend(["--adv-labels", str(labels)])
    if skip_clean_val:
        if "recall" not in clean_metrics:
            print(f"[FAIL] {name}: skip_clean_val=true but clean_metrics.recall is missing.")
            return None
        cmd.extend(
            [
                "--skip-clean-val",
                "--clean-map50",
                str(clean_metrics.get("map50", 0.0)),
                "--clean-map50-95",
                str(clean_metrics.get("map50_95", 0.0)),
                "--clean-precision",
                str(clean_metrics.get("precision", 0.0)),
                "--clean-recall",
                str(clean_metrics["recall"]),
                "--clean-f1",
                str(clean_metrics.get("f1", 0.0)),
            ]
        )
    if skip_paired_asr:
        cmd.append("--skip-paired-asr")

    print(f"\n===== Evaluating {name} =====")
    log_path = log_path_for(output_prefix, str(cfg["dataset"]), name)
    print(f"Log: {log_path}")
    return_code, text = run_and_log(
        cmd,
        env={**dict(os.environ), "CUDA_VISIBLE_DEVICES": args.gpu},
        log_path=log_path,
    )
    if return_code != 0:
        print(f"[FAIL] {name}: evaluator exited with {return_code}")
        return None

    clean = re.search(
        r"Clean mAP50=([0-9.]+), Clean mAP50-95=([0-9.]+), "
        r"Clean Precision=([0-9.]+), Clean Recall=([0-9.]+), Clean F1=([0-9.]+)",
        text,
    )
    adv = re.search(
        r"Adv mAP50=([0-9.]+), Adv mAP50-95=([0-9.]+), "
        r"Adv Precision=([0-9.]+), Adv Recall=([0-9.]+), Adv F1=([0-9.]+), ASR=([0-9.]+)",
        text,
    )
    paired = re.search(r"Paired object ASR=([0-9.]+)", text)
    if not clean or not adv:
        print(f"[FAIL] {name}: could not parse metrics")
        return None
    row = {
        "attack": name,
        "target_detector": args.target_detector,
        "clean_map50": float(clean.group(1)),
        "clean_map50_95": float(clean.group(2)),
        "clean_precision": float(clean.group(3)),
        "clean_recall": float(clean.group(4)),
        "clean_f1": float(clean.group(5)),
        "adv_map50": float(adv.group(1)),
        "adv_map50_95": float(adv.group(2)),
        "adv_precision": float(adv.group(3)),
        "adv_recall": float(adv.group(4)),
        "adv_f1": float(adv.group(5)),
        "recall_drop_asr": float(adv.group(6)),
        "paired_object_asr": float(paired.group(1)) if paired else 0.0,
    }
    write_cached_result(cache_path, row)
    print(f"Cached result: {cache_path}")
    return row


def write_outputs(rows: List[Dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    fields = [
        "attack",
        "target_detector",
        "clean_map50",
        "clean_map50_95",
        "clean_precision",
        "adv_map50",
        "adv_map50_95",
        "adv_precision",
        "clean_recall",
        "adv_recall",
        "clean_f1",
        "adv_f1",
        "recall_drop_asr",
        "paired_object_asr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Attack | Target Detector | mAP50 (%) | mAP50-95 (%) | Recall (%) | F1 (%) | ASR (%) | Paired Object ASR (%) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack']} | {row['target_detector']} | {row['adv_map50']:.1f} | "
            f"{row['adv_map50_95']:.1f} | {row['adv_recall']:.1f} | {row['adv_f1']:.1f} | "
            f"{row['recall_drop_asr']:.1f} | {row['paired_object_asr']:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_prefix = Path(args.output_prefix)
    wanted = selected_attacks(args)
    rows = []
    for name, attack in cfg["attacks"].items():
        if wanted is not None and name not in wanted:
            continue
        row = run_one_attack(name, attack or {}, cfg, args, output_prefix)
        if row:
            rows.append(row)
    if not rows:
        raise SystemExit("No attacks were evaluated. Fill the adv image paths in the config.")
    write_outputs(rows, output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
