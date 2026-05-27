#!/usr/bin/env python3
"""Generate RT-DETR white-box adversarial images for baseline attacks.

The original TOG/Daedalus/OSFD repositories are left untouched. This adapter
uses RT-DETR gradients and attack-specific objectives to produce comparable
white-box adversarial images in YOLO-format folders.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from train_ours_rtdetr_advgan import IMAGE_EXTS, RTDETRTarget, collect_tensors, load_data_yaml, resolve_split_dir
from generate_ours_adv_images import label_dir_from_image_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RT-DETR white-box baseline adversarial images.")
    parser.add_argument("--attack", required=True, choices=["tog", "daedalus", "dae", "osfd", "osfp"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", required=True, help="Output root containing images/ and labels/.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.0, help="0 means eps / 4.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all matched images.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing images.")
    return parser.parse_args()


def normalize_attack(name: str) -> str:
    lowered = name.lower()
    if lowered == "dae":
        return "daedalus"
    if lowered == "osfp":
        return "osfd"
    return lowered


def list_matched_images(image_dir: Path, label_dir: Path) -> List[Path]:
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    return [path for path in images if (label_dir / f"{path.stem}.txt").exists()]


def load_batch(paths: List[Path], imgsz: int, device: torch.device):
    to_tensor = transforms.Compose([transforms.Resize((imgsz, imgsz)), transforms.ToTensor()])
    tensors: List[torch.Tensor] = []
    original_sizes: List[Tuple[int, int]] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        original_sizes.append(image.size)
        tensors.append(to_tensor(image))
    return torch.stack(tensors, dim=0).to(device), original_sizes


def save_batch(
    adv: torch.Tensor,
    image_paths: List[Path],
    original_sizes: List[Tuple[int, int]],
    label_dir: Path,
    out_images: Path,
    out_labels: Path,
    quality: int,
) -> None:
    to_pil = transforms.ToPILImage()
    adv_cpu = adv.detach().cpu()
    for tensor, image_path, original_size in zip(adv_cpu, image_paths, original_sizes):
        adv_image = to_pil(tensor).resize(original_size, Image.BILINEAR)
        adv_image.save(out_images / image_path.name, quality=quality)
        shutil.copy2(label_dir / f"{image_path.stem}.txt", out_labels / f"{image_path.stem}.txt")


def target_tensors(target: RTDETRTarget, images: torch.Tensor) -> List[torch.Tensor]:
    return [t.float() for t in collect_tensors(target.raw_forward(images)) if t.is_floating_point()]


def sigmoid_mean(tensors: List[torch.Tensor]) -> torch.Tensor:
    values = [torch.sigmoid(t).mean() for t in tensors if t.numel() > 0]
    if not values:
        return torch.zeros((), device=tensors[0].device if tensors else "cpu")
    return torch.stack(values).mean()


def sigmoid_entropy(tensors: List[torch.Tensor]) -> torch.Tensor:
    values = []
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        probs = torch.sigmoid(tensor).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log()).mean()
        values.append(entropy)
    if not values:
        return torch.zeros((), device=tensors[0].device if tensors else "cpu")
    return torch.stack(values).mean()


def feature_distance(adv_tensors: List[torch.Tensor], clean_tensors: List[torch.Tensor]) -> torch.Tensor:
    values = []
    for adv, clean in zip(adv_tensors, clean_tensors):
        if adv.shape != clean.shape or adv.numel() == 0:
            continue
        values.append(torch.mean(torch.abs(torch.sigmoid(adv) - torch.sigmoid(clean))))
    if not values:
        return torch.zeros((), device=adv_tensors[0].device if adv_tensors else "cpu")
    return torch.stack(values).mean()


def attack_objective(
    target: RTDETRTarget,
    attack: str,
    adv: torch.Tensor,
    clean_tensors: Optional[List[torch.Tensor]],
) -> tuple[torch.Tensor, float]:
    adv_tensors = target_tensors(target, adv)
    mean_score = sigmoid_mean(adv_tensors)

    if attack == "tog":
        # TOG-style vanishing: suppress detector activations.
        return mean_score, -1.0

    if attack == "daedalus":
        # Daedalus-style confusion: maximize output uncertainty while suppressing confident detections.
        return sigmoid_entropy(adv_tensors) - 0.25 * mean_score, 1.0

    if clean_tensors is None:
        raise RuntimeError("OSFD-style attack requires clean tensors.")
    # OSFD-style feature disruption: push RT-DETR internal outputs away from clean responses.
    return feature_distance(adv_tensors, clean_tensors) - 0.50 * mean_score, 1.0


def pgd_attack(
    target: RTDETRTarget,
    clean: torch.Tensor,
    attack: str,
    eps: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    adv = clean.detach().clone()
    clean_tensors = None
    if attack == "osfd":
        with torch.no_grad():
            clean_tensors = [tensor.detach() for tensor in target_tensors(target, clean)]

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        objective, direction = attack_objective(target, attack, adv, clean_tensors)
        grad = torch.autograd.grad(objective, adv, create_graph=False, retain_graph=False)[0]
        grad = torch.nan_to_num(grad)
        with torch.no_grad():
            adv = adv + direction * step_size * grad.sign()
            delta = torch.clamp(adv - clean, min=-eps, max=eps)
            adv = torch.clamp(clean + delta, 0.0, 1.0)
    return adv.detach()


def main() -> int:
    args = parse_args()
    attack = normalize_attack(args.attack)
    data = load_data_yaml(args.data)
    root = Path(str(data["path"])).expanduser()
    image_dir = resolve_split_dir(data, args.split)
    label_dir = label_dir_from_image_dir(root, image_dir, args.split)

    out_root = Path(args.output)
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = list_matched_images(image_dir, label_dir)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]
    if not args.overwrite:
        image_paths = [
            path
            for path in image_paths
            if not (out_images / path.name).exists() or not (out_labels / f"{path.stem}.txt").exists()
        ]

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    target = RTDETRTarget(args.weights, device)
    step_size = args.step_size if args.step_size > 0 else args.eps / 4.0

    print(f"Attack: {attack}")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Output: {out_root}")
    print(f"RT-DETR weights: {args.weights}")
    print(f"Remaining images to generate: {len(image_paths)}")
    print(f"eps={args.eps}, steps={args.steps}, step_size={step_size}, batch={args.batch}, imgsz={args.imgsz}")

    for start in tqdm(range(0, len(image_paths), args.batch), desc=f"Generating {attack}"):
        batch_paths = image_paths[start : start + args.batch]
        clean, original_sizes = load_batch(batch_paths, args.imgsz, device)
        adv = pgd_attack(target, clean, attack, args.eps, args.steps, step_size)
        save_batch(adv, batch_paths, original_sizes, label_dir, out_images, out_labels, args.quality)

    print(f"Saved adversarial images to: {out_images}")
    print(f"Saved labels to: {out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
