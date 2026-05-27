#!/usr/bin/env python3
"""Train the paper's AdvGAN-AdaAD attack against Ultralytics RT-DETR.

This is intentionally self-contained so the original TOG/Daedalus/OSFD folders
and the previous YOLO-target AdvGAN code remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.ops import box_iou
from tqdm import tqdm

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_dir(data: Dict[str, Any], split: str) -> Path:
    root = Path(str(data["path"])).expanduser()
    split_value = data[split]
    split_path = Path(str(split_value)).expanduser()
    return split_path if split_path.is_absolute() else root / split_path


class YoloImageDataset(Dataset):
    def __init__(self, data_yaml: str | Path, split: str, imgsz: int):
        data = load_data_yaml(data_yaml)
        self.image_dir = resolve_split_dir(data, split)
        self.image_paths = sorted(
            p for p in self.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        self.transform = transforms.Compose(
            [transforms.Resize((imgsz, imgsz)), transforms.ToTensor()]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_path.name


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            ResnetBlock(256),
            ResnetBlock(256),
            ChannelAttention(256),
            ResnetBlock(256),
            ResnetBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).flatten()


def collect_tensors(output: Any) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    if isinstance(output, torch.Tensor):
        tensors.append(output)
    elif isinstance(output, (list, tuple)):
        for item in output:
            tensors.extend(collect_tensors(item))
    elif isinstance(output, dict):
        for item in output.values():
            tensors.extend(collect_tensors(item))
    return tensors


class RTDETRTarget:
    def __init__(self, weights: str, device: torch.device):
        from ultralytics import RTDETR

        self.device = device
        self.model = RTDETR(weights)
        self.nn = self.model.model.to(device)
        self.nn.eval()
        for param in self.nn.parameters():
            param.requires_grad = False

    def raw_forward(self, images: torch.Tensor):
        was_training = self.nn.training
        self.nn.train()
        for module in self.nn.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
        output = self.nn(images)
        self.nn.train(was_training)
        return output

    def detection_loss(self, images: torch.Tensor) -> torch.Tensor:
        tensors = [t.float() for t in collect_tensors(self.raw_forward(images)) if t.is_floating_point()]
        if not tensors:
            return torch.zeros((), device=images.device)
        losses = []
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            # Vanishing objective: push object/class activations toward low confidence.
            losses.append(torch.sigmoid(tensor).mean())
        return torch.stack(losses).mean() if losses else torch.zeros((), device=images.device)

    @torch.no_grad()
    def predict_boxes(self, images: torch.Tensor, imgsz: int, conf: float, iou: float):
        results = self.model.predict(
            source=images,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=0 if images.is_cuda else "cpu",
            verbose=False,
        )
        boxes = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                boxes.append(torch.zeros((0, 6), device=images.device))
                continue
            xyxy = result.boxes.xyxy.to(images.device)
            confs = result.boxes.conf.to(images.device).view(-1, 1)
            cls = result.boxes.cls.to(images.device).view(-1, 1)
            boxes.append(torch.cat([xyxy, confs, cls], dim=1))
        return boxes


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.mean(
        torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    )


def vanishing_stats(clean_boxes: List[torch.Tensor], adv_boxes: List[torch.Tensor], iou_thr: float):
    total = 0
    vanished = 0
    fp = 0
    for clean, adv in zip(clean_boxes, adv_boxes):
        if len(clean) == 0:
            fp += len(adv)
            continue
        total += len(clean)
        if len(adv) == 0:
            vanished += len(clean)
            continue
        ious = box_iou(clean[:, :4], adv[:, :4])
        vanished += (ious.max(dim=1).values < iou_thr).sum().item()
        fp += (ious.max(dim=0).values < iou_thr).sum().item()
    return vanished, total, fp


@dataclass
class TrainConfig:
    data: str
    weights: str
    output: Path
    split: str
    imgsz: int
    epochs: int
    batch: int
    workers: int
    eps: float
    alpha_det: float
    alpha_smooth: float
    alpha_gan: float
    alpha_traj: float
    adaad_steps: int
    lr_g: float
    lr_d: float
    conf: float
    iou: float
    eval_samples: int
    max_train_samples: int
    seed: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train AdvGAN-AdaAD against RT-DETR.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha-det", type=float, default=10.0)
    parser.add_argument("--alpha-smooth", type=float, default=0.001)
    parser.add_argument("--alpha-gan", type=float, default=0.02)
    parser.add_argument("--alpha-traj", type=float, default=2.0)
    parser.add_argument("--adaad-steps", type=int, default=5)
    parser.add_argument("--lr-g", type=float, default=5e-5)
    parser.add_argument("--lr-d", type=float, default=5e-5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Use at most this many training images per epoch. 0 means full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(**{**vars(args), "output": Path(args.output)})


def adaad_search(
    target: RTDETRTarget,
    clean: torch.Tensor,
    adv: torch.Tensor,
    eps: float,
    steps: int,
) -> torch.Tensor:
    step_size = eps / 2.0
    x_temp = adv.detach()
    for _ in range(steps):
        x_temp = x_temp.detach().requires_grad_(True)
        loss = target.detection_loss(x_temp)
        grad = torch.autograd.grad(loss, x_temp, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            x_temp = x_temp - step_size * grad.sign()
            delta = torch.clamp(x_temp - clean, -eps, eps)
            x_temp = torch.clamp(clean + delta, 0.0, 1.0)
    return x_temp.detach()


def evaluate_quick(
    target: RTDETRTarget,
    generator: Generator,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    generator.eval()
    total = vanished = fp = images_seen = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            delta = generator(images) * cfg.eps
            adv = torch.clamp(images + delta, 0.0, 1.0)
            clean_boxes = target.predict_boxes(images, cfg.imgsz, cfg.conf, cfg.iou)
            adv_boxes = target.predict_boxes(adv, cfg.imgsz, cfg.conf, cfg.iou)
            v, t, f = vanishing_stats(clean_boxes, adv_boxes, cfg.iou)
            vanished += v
            total += t
            fp += f
            images_seen += images.shape[0]
            if images_seen >= cfg.eval_samples:
                break
    generator.train()
    return {
        "quick_asr": 100.0 * vanished / total if total else 0.0,
        "quick_fp_per_img": fp / max(1, images_seen),
        "quick_objects": float(total),
    }


def main() -> int:
    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.output / "weights").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {cfg.output}")
    print(f"Target RT-DETR: {cfg.weights}")

    full_dataset = YoloImageDataset(cfg.data, cfg.split, cfg.imgsz)
    if cfg.max_train_samples > 0 and cfg.max_train_samples < len(full_dataset):
        rng = random.Random(cfg.seed)
        indices = list(range(len(full_dataset)))
        rng.shuffle(indices)
        indices = sorted(indices[: cfg.max_train_samples])
        dataset = Subset(full_dataset, indices)
        print(f"Using training subset: {len(dataset)}/{len(full_dataset)} images")
    else:
        dataset = full_dataset
        print(f"Using full training split: {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_base = full_dataset if isinstance(dataset, Subset) else dataset
    eval_indices = list(range(min(len(eval_base), max(cfg.eval_samples, cfg.batch))))
    eval_loader = DataLoader(
        Subset(eval_base, eval_indices),
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=max(0, min(2, cfg.workers)),
    )

    target = RTDETRTarget(cfg.weights, device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = cfg.output / f"training_log_{timestamp}.csv"
    best_asr = -1.0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss_d",
                "loss_g",
                "loss_det",
                "loss_smooth",
                "loss_traj",
                "linf",
                "quick_asr",
                "quick_fp_per_img",
                "quick_objects",
            ],
        )
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            sums = {k: 0.0 for k in ["loss_d", "loss_g", "loss_det", "loss_smooth", "loss_traj", "linf"]}
            steps = 0
            for clean, _ in tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
                clean = clean.to(device)
                perturb = generator(clean) * cfg.eps
                adv = torch.clamp(clean + perturb, 0.0, 1.0)

                opt_d.zero_grad(set_to_none=True)
                pred_real = discriminator(clean)
                pred_fake = discriminator(adv.detach())
                loss_d = F.mse_loss(pred_real, torch.ones_like(pred_real) * 0.9) + F.mse_loss(
                    pred_fake, torch.zeros_like(pred_fake) + 0.1
                )
                loss_d.backward()
                opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                x_target = adaad_search(target, clean, adv, cfg.eps, cfg.adaad_steps)
                perturb = generator(clean) * cfg.eps
                adv = torch.clamp(clean + perturb, 0.0, 1.0)
                pred_fake = discriminator(adv)
                loss_gan = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
                loss_det = target.detection_loss(adv)
                loss_smooth = tv_loss(perturb)
                loss_traj = F.l1_loss(perturb, x_target - clean)
                loss_g = (
                    cfg.alpha_det * loss_det
                    + cfg.alpha_traj * loss_traj
                    + cfg.alpha_smooth * loss_smooth
                    + cfg.alpha_gan * loss_gan
                )
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
                opt_g.step()

                sums["loss_d"] += float(loss_d.detach())
                sums["loss_g"] += float(loss_g.detach())
                sums["loss_det"] += float(loss_det.detach())
                sums["loss_smooth"] += float(loss_smooth.detach())
                sums["loss_traj"] += float(loss_traj.detach())
                sums["linf"] += float(perturb.detach().abs().max())
                steps += 1

            row = {key: value / max(1, steps) for key, value in sums.items()}
            row.update(evaluate_quick(target, generator, eval_loader, device, cfg))
            row["epoch"] = epoch
            writer.writerow(row)
            f.flush()
            print(
                f"Epoch {epoch}: loss_g={row['loss_g']:.4f}, "
                f"quick_asr={row['quick_asr']:.2f}, fp/img={row['quick_fp_per_img']:.3f}"
            )

            torch.save(generator.state_dict(), cfg.output / "weights" / "netG_latest.pth")
            if row["quick_asr"] > best_asr:
                best_asr = row["quick_asr"]
                torch.save(generator.state_dict(), cfg.output / "weights" / "netG_best_asr.pth")
            if epoch % 10 == 0:
                torch.save(generator.state_dict(), cfg.output / "weights" / f"netG_epoch_{epoch}.pth")

    print(f"Training log: {csv_path}")
    print(f"Best generator: {cfg.output / 'weights' / 'netG_best_asr.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
