#!/usr/bin/env python3
"""Generate YOLO-format adversarial image folders from a trained RT-DETR AdvGAN."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from train_ours_rtdetr_advgan import Generator, IMAGE_EXTS, load_data_yaml, resolve_split_dir


def label_dir_from_image_dir(root: Path, image_dir: Path, split: str) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        replaced = parts.copy()
        replaced[idx] = "labels"
        candidate = Path(*replaced)
        if candidate.is_dir():
            return candidate
    candidates = [
        root / split / "labels",
        root / "labels" / split,
        root / "test" / "labels" if split in {"test", "val"} else root / "train" / "labels",
        root / "labels" / "test" if split in {"test", "val"} else root / "labels" / "train",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial images with trained netG.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--generator", required=True, help="Path to netG_*.pth")
    parser.add_argument("--output", required=True, help="Output root with images/ and labels/")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all matched images")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_data_yaml(args.data)
    root = Path(str(data["path"])).expanduser()
    image_dir = resolve_split_dir(data, args.split)
    label_dir = label_dir_from_image_dir(root, image_dir, args.split)
    out_root = Path(args.output)
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.generator, map_location=device))
    generator.eval()

    to_tensor = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)), transforms.ToTensor()])
    to_pil = transforms.ToPILImage()

    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Output: {out_root}")
    print(f"Generator: {args.generator}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Generating adv images"):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            original = Image.open(image_path).convert("RGB")
            original_size = original.size
            tensor = to_tensor(original).unsqueeze(0).to(device)
            perturb = generator(tensor) * args.eps
            adv = torch.clamp(tensor + perturb, 0.0, 1.0).squeeze(0).cpu()
            adv_image = to_pil(adv).resize(original_size, Image.BILINEAR)
            adv_image.save(out_images / image_path.name, quality=95)
            shutil.copy2(label_path, out_labels / f"{image_path.stem}.txt")

    print(f"Saved adversarial images to: {out_images}")
    print(f"Saved labels to: {out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
