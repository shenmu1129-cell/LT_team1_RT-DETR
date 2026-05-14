#!/usr/bin/env python3
"""Check a YOLO-format detection dataset before Ultralytics RT-DETR training.

The checker intentionally avoids OpenMMLab/MMDetection/mmcv/mmengine imports.
It validates image/label layout, label syntax, class id ranges, and produces a
resolved data yaml when a dataset such as TT100K uses one of several common
directory layouts.
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - environment dependent
    yaml = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TT100K_SPLIT_CANDIDATES = {
    "train": ["train/images", "images/train"],
    "val": ["test/images", "images/test"],
    "test": ["test/images", "images/test"],
}


@dataclass
class SplitStats:
    split: str
    image_dir: Path
    label_dir: Path
    image_count: int
    label_count: int
    empty_label_count: int
    class_counts: Counter


@dataclass
class DatasetCheckResult:
    data_path: Path
    root: Path
    yaml_data: Dict[str, Any]
    resolved_yaml_data: Dict[str, Any]
    split_stats: Dict[str, SplitStats]
    warnings: List[str]
    errors: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(data_path: Path) -> Dict[str, Any]:
    """Minimal yaml reader for this repo's simple dataset files.

    PyYAML is still preferred when installed. This fallback supports top-level
    scalar keys and block lists such as names: [- item], which keeps the checker
    usable in lightweight environments.
    """
    lines = data_path.read_text(encoding="utf-8").splitlines()
    data: Dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1].isspace() or ":" not in raw:
            raise ValueError(
                f"Unsupported yaml syntax in fallback parser at line {index}: {raw}"
            )

        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = _parse_scalar(value)
            continue

        block_items: List[Any] = []
        while index < len(lines):
            child_raw = lines[index]
            child = child_raw.strip()
            if not child or child.startswith("#"):
                index += 1
                continue
            if not child_raw[:1].isspace():
                break
            if not child.startswith("- "):
                raise ValueError(
                    f"Unsupported yaml list syntax in fallback parser at line "
                    f"{index + 1}: {child_raw}"
                )
            block_items.append(_parse_scalar(child[2:]))
            index += 1
        data[key] = block_items
    return data


def _dump_simple_yaml(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def load_yaml(data_path: Path) -> Dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml does not exist: {data_path}")
    if yaml is not None:
        with data_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = _load_simple_yaml(data_path)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset yaml must contain a mapping: {data_path}")
    return data


def _as_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _rel_to_root(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_tt100k_root(root: Path) -> bool:
    return "tt100k" in root.as_posix().lower()


def resolve_image_dir(
    root: Path,
    split: str,
    configured: Optional[str],
    warnings: List[str],
) -> Path:
    """Resolve image directory, including TT100K's two common layouts."""
    candidates: List[Path] = []
    if configured:
        candidates.append(_as_path(root, configured))
    if _is_tt100k_root(root):
        candidates.extend(_as_path(root, item) for item in TT100K_SPLIT_CANDIDATES[split])

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = candidate.resolve(strict=False)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_dir():
            if configured and candidate != _as_path(root, configured):
                warnings.append(
                    f"{split}: configured image dir was missing; using detected layout "
                    f"{candidate}"
                )
            return candidate

    return unique_candidates[0] if unique_candidates else root / split


def label_dir_candidates(root: Path, image_dir: Path, split: str) -> List[Path]:
    """Build likely YOLO label directories for a resolved image directory."""
    candidates: List[Path] = []

    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        replaced = parts.copy()
        replaced[idx] = "labels"
        candidates.append(Path(*replaced))

    candidates.extend(
        [
            root / split / "labels",
            root / "labels" / split,
            root / "test" / "labels" if split in {"val", "test"} else root / "train" / "labels",
            root / "labels" / "test" if split in {"val", "test"} else root / "labels" / "train",
        ]
    )

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = candidate.resolve(strict=False)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_label_dir(root: Path, image_dir: Path, split: str) -> Path:
    candidates = label_dir_candidates(root, image_dir, split)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def list_labels(label_dir: Path) -> List[Path]:
    if not label_dir.is_dir():
        return []
    return sorted(label_dir.rglob("*.txt"))


def validate_names(data: Dict[str, Any], nc: int, errors: List[str], warnings: List[str]) -> None:
    names = data.get("names")
    if names is None:
        errors.append("Missing 'names' in dataset yaml.")
        return

    if isinstance(names, dict):
        names_list = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        names_list = names
    else:
        errors.append("'names' must be a list or dict.")
        return

    if len(names_list) != nc:
        errors.append(f"'names' length ({len(names_list)}) does not match nc ({nc}).")

    todo_names = [str(name) for name in names_list if "TODO" in str(name).upper()]
    if todo_names:
        errors.append(
            "Dataset yaml still contains TODO class names. Replace them with the real "
            "class names in YOLO class-id order before training/evaluating."
        )

    duplicated = [name for name, count in Counter(map(str, names_list)).items() if count > 1]
    if duplicated:
        warnings.append(f"Duplicate class names found: {duplicated[:10]}")


def _parse_label_line(
    line: str,
    label_file: Path,
    line_no: int,
    nc: int,
    errors: List[str],
    class_counts: Counter,
) -> None:
    parts = line.split()
    if len(parts) != 5:
        errors.append(f"{label_file}:{line_no}: expected 5 YOLO fields, got {len(parts)}")
        return

    class_text, *box_text = parts
    try:
        class_id_float = float(class_text)
        class_id = int(class_id_float)
    except ValueError:
        errors.append(f"{label_file}:{line_no}: class id is not numeric: {class_text!r}")
        return

    if class_id_float != class_id:
        errors.append(f"{label_file}:{line_no}: class id must be an integer: {class_text!r}")
        return

    if class_id < 0 or class_id >= nc:
        errors.append(
            f"{label_file}:{line_no}: class id {class_id} is outside valid range [0, {nc - 1}]"
        )
    else:
        class_counts[class_id] += 1

    try:
        x_center, y_center, width, height = [float(item) for item in box_text]
    except ValueError:
        errors.append(f"{label_file}:{line_no}: box coordinates must be numeric")
        return

    values = [x_center, y_center, width, height]
    if any(value < 0.0 or value > 1.0 for value in values):
        errors.append(f"{label_file}:{line_no}: YOLO coordinates must be normalized to [0, 1]")
    if width <= 0.0 or height <= 0.0:
        errors.append(f"{label_file}:{line_no}: width and height must be positive")


def scan_labels(
    label_files: Iterable[Path],
    nc: int,
    errors: List[str],
    warnings: List[str],
) -> Tuple[Counter, int]:
    class_counts: Counter = Counter()
    empty_label_count = 0

    for label_file in label_files:
        text = label_file.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            empty_label_count += 1
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            _parse_label_line(line, label_file, line_no, nc, errors, class_counts)

    if empty_label_count:
        warnings.append(
            f"Found {empty_label_count} empty label files. This is allowed for images "
            "without targets."
        )
    return class_counts, empty_label_count


def random_label_sanity_check(
    label_files: List[Path],
    nc: int,
    sample_count: int,
    errors: List[str],
) -> None:
    """Run an explicit random sample check in addition to the full label scan."""
    if not label_files or sample_count <= 0:
        return
    sampled = random.sample(label_files, k=min(sample_count, len(label_files)))
    sample_counts: Counter = Counter()
    for label_file in sampled:
        text = label_file.read_text(encoding="utf-8", errors="replace").strip()
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if line:
                _parse_label_line(line, label_file, line_no, nc, errors, sample_counts)


def check_dataset(
    data: str | Path,
    sample_count: int = 20,
    strict: bool = False,
) -> DatasetCheckResult:
    """Check a YOLO dataset yaml and return structured stats.

    Args:
        data: Path to the YOLO data yaml.
        sample_count: Number of label files to randomly re-check per split.
        strict: If True, raise RuntimeError when errors are found.
    """
    data_path = Path(data).expanduser()
    yaml_data = load_yaml(data_path)
    warnings: List[str] = []
    errors: List[str] = []

    if "path" not in yaml_data:
        errors.append("Missing required key: path")
        root = data_path.parent
    else:
        root = Path(str(yaml_data["path"])).expanduser()

    try:
        nc = int(yaml_data.get("nc"))
    except (TypeError, ValueError):
        nc = -1
        errors.append("'nc' must be an integer.")

    if nc > 0:
        validate_names(yaml_data, nc, errors, warnings)

    if not root.exists():
        errors.append(f"Dataset root does not exist: {root}")

    resolved_yaml_data = dict(yaml_data)
    split_stats: Dict[str, SplitStats] = {}

    for split in ("train", "val", "test"):
        configured = yaml_data.get(split)
        if configured is None:
            errors.append(f"Missing required key: {split}")
            continue

        image_dir = resolve_image_dir(root, split, str(configured), warnings)
        label_dir = resolve_label_dir(root, image_dir, split)

        if not image_dir.is_dir():
            errors.append(f"{split}: image directory does not exist: {image_dir}")
        if not label_dir.is_dir():
            errors.append(f"{split}: label directory does not exist: {label_dir}")

        image_files = list_images(image_dir)
        label_files = list_labels(label_dir)

        if image_dir.is_dir() and not image_files:
            warnings.append(f"{split}: no images found in {image_dir}")
        if label_dir.is_dir() and not label_files:
            warnings.append(f"{split}: no label txt files found in {label_dir}")

        diff = abs(len(image_files) - len(label_files))
        tolerance = max(5, int(0.1 * max(1, len(image_files))))
        if diff > tolerance:
            warnings.append(
                f"{split}: image/label count differs noticeably "
                f"({len(image_files)} images vs {len(label_files)} labels)."
            )

        if nc > 0:
            class_counts, empty_count = scan_labels(label_files, nc, errors, warnings)
            random_label_sanity_check(label_files, nc, sample_count, errors)
        else:
            class_counts, empty_count = Counter(), 0

        split_stats[split] = SplitStats(
            split=split,
            image_dir=image_dir,
            label_dir=label_dir,
            image_count=len(image_files),
            label_count=len(label_files),
            empty_label_count=empty_count,
            class_counts=class_counts,
        )
        resolved_yaml_data[split] = _rel_to_root(root, image_dir)

    result = DatasetCheckResult(
        data_path=data_path,
        root=root,
        yaml_data=yaml_data,
        resolved_yaml_data=resolved_yaml_data,
        split_stats=split_stats,
        warnings=warnings,
        errors=errors,
    )

    if strict and not result.ok:
        raise RuntimeError(format_result(result))
    return result


def write_resolved_yaml(result: DatasetCheckResult, output_path: str | Path) -> Path:
    """Write a data yaml with detected train/val/test image dirs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(result.resolved_yaml_data, f, sort_keys=False, allow_unicode=True)
        else:
            f.write(_dump_simple_yaml(result.resolved_yaml_data))
    return output_path


def format_class_counts(counter: Counter, nc: int) -> str:
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{i}: {counter.get(i, 0)}" for i in range(nc) if counter.get(i, 0)) + "}"


def format_result(result: DatasetCheckResult) -> str:
    nc = int(result.yaml_data.get("nc", 0) or 0)
    lines = [
        f"Dataset yaml: {result.data_path}",
        f"Dataset root: {result.root}",
    ]

    for split in ("train", "val", "test"):
        stats = result.split_stats.get(split)
        if not stats:
            continue
        lines.extend(
            [
                "",
                f"[{split}]",
                f"images: {stats.image_dir}",
                f"labels: {stats.label_dir}",
                f"image_count: {stats.image_count}",
                f"label_count: {stats.label_count}",
                f"empty_label_count: {stats.empty_label_count}",
                f"class_counts: {format_class_counts(stats.class_counts, nc)}",
            ]
        )

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)

    if result.errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)

    lines.append("")
    lines.append("Status: OK" if result.ok else "Status: FAILED")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO-format dataset yaml.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset yaml.")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of random label files to sample per split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = check_dataset(args.data, sample_count=args.samples, strict=False)
    except Exception as exc:
        print(f"Dataset check failed before scanning: {exc}", file=sys.stderr)
        return 1

    print(format_result(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
