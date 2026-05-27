# RT-DETR 论文补表实验

这个目录用于给论文 Table 1 / Table 2 补充 RT-DETR 结果。代码只放在当前仓库内，服务器上使用时位于：

```bash
/home/sutongtong/wwt/code/LT_team1_RT-DETR/rtdetr_paper_experiments
```

## 目标

在 CCTSDB 和 TT100K 上，用统一流程得到 RT-DETR 在四类攻击下的指标：

- TOG
- Daedalus
- OSFD
- Ours / AdvGAN-AdaAD

输出指标包括：

- Clean mAP50
- Adv mAP50
- Clean Recall
- Adv Recall
- Recall-drop ASR
- Paired Object ASR

## 实验路径约定

RT-DETR 项目路径：

```bash
/home/sutongtong/wwt/code/LT_team1_RT-DETR
```

攻击代码或攻击样本路径：

```bash
TOG:      /home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/TOG
Daedalus: /home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/Dae
OSFD:     /home/sutongtong/LanTu_team1/OSFD
Ours:     本目录提供 RT-DETR 版本 AdvGAN-AdaAD 训练与生成脚本
```

## 1. 训练 Ours / AdvGAN-AdaAD for RT-DETR

CCTSDB：

```bash
GPU_ID=2 bash rtdetr_paper_experiments/scripts/train_ours_rtdetr_cctsdb_bg.sh
```

TT100K：

```bash
GPU_ID=2 bash rtdetr_paper_experiments/scripts/train_ours_rtdetr_tt100k_bg.sh
```

默认每个 epoch 随机固定抽取 `MAX_TRAIN_SAMPLES=2000` 张图训练生成器，避免全量训练过慢。可以按需覆盖：

```bash
GPU_ID=2 MAX_TRAIN_SAMPLES=4000 EPOCHS=30 BATCH=16 IMGSZ=640 bash rtdetr_paper_experiments/scripts/train_ours_rtdetr_cctsdb_bg.sh
```

如果要使用全量训练集，设置：

```bash
MAX_TRAIN_SAMPLES=0
```

默认会输出到：

```bash
rtdetr_paper_experiments/runs/ours_rtdetr_cctsdb
rtdetr_paper_experiments/runs/ours_rtdetr_tt100k
```

训练过程中会保存两类权重：

```bash
weights/netG_latest.pth          # 最新生成器，用于生成对抗样本
weights/netG_best_asr.pth        # quick_asr 最高的生成器，用于生成对抗样本
weights/checkpoint_latest.pth    # 完整恢复点，包含生成器、判别器、优化器、epoch、best_asr
weights/checkpoint_best_asr.pth  # quick_asr 最高时的完整恢复点
```

如果训练中途断了，可以从完整 checkpoint 继续。注意 `EPOCHS` 表示最终要训练到的总 epoch 数，不是额外再训练多少轮：

```bash
GPU_ID=3 \
EPOCHS=30 \
BATCH=32 \
IMGSZ=640 \
WORKERS=8 \
MAX_TRAIN_SAMPLES=2000 \
RESUME_CHECKPOINT=rtdetr_paper_experiments/runs/ours_rtdetr_cctsdb/weights/checkpoint_latest.pth \
bash rtdetr_paper_experiments/scripts/train_ours_rtdetr_cctsdb_bg.sh
```

如果要调强攻击，可以直接覆盖训练参数，例如：

```bash
GPU_ID=3 \
EPOCHS=30 \
BATCH=32 \
IMGSZ=640 \
WORKERS=8 \
MAX_TRAIN_SAMPLES=3000 \
LR_G=1e-4 \
LR_D=5e-5 \
ALPHA_DET=20 \
ADAAD_STEPS=8 \
bash rtdetr_paper_experiments/scripts/train_ours_rtdetr_cctsdb_bg.sh
```

## 2. 生成 Ours 对抗样本

CCTSDB：

```bash
GPU_ID=2 bash rtdetr_paper_experiments/scripts/generate_ours_cctsdb.sh
```

TT100K：

```bash
GPU_ID=2 bash rtdetr_paper_experiments/scripts/generate_ours_tt100k.sh
```

输出目录：

```bash
rtdetr_paper_experiments/adv_outputs/ours/cctsdb/images
rtdetr_paper_experiments/adv_outputs/ours/cctsdb/labels
rtdetr_paper_experiments/adv_outputs/ours/tt100k/images
rtdetr_paper_experiments/adv_outputs/ours/tt100k/labels
```

## 3. 准备 TOG / Daedalus / OSFD 攻击样本

本目录不修改 TOG、Daedalus、OSFD 的原始代码。你可以用它们自己的脚本生成对抗样本，然后把路径填入：

```bash
rtdetr_paper_experiments/configs/attacks_cctsdb.yaml
rtdetr_paper_experiments/configs/attacks_tt100k.yaml
```

每个攻击需要提供：

```yaml
images: /path/to/adv/images
labels: /path/to/adv/labels   # 如果没有，可留空，评估时复用 clean labels
```

## 4. 生成 Table 1 / Table 2

CCTSDB：

```bash
GPU_ID=2 bash rtdetr_paper_experiments/scripts/eval_table1_cctsdb.sh
```

TT100K：

```bash
GPU_ID=2 bash rtdetr_paper_experiments/scripts/eval_table2_tt100k.sh
```

输出：

```bash
rtdetr_paper_experiments/results/table1_cctsdb.md
rtdetr_paper_experiments/results/table1_cctsdb.csv
rtdetr_paper_experiments/results/table2_tt100k.md
rtdetr_paper_experiments/results/table2_tt100k.csv
```

## 重要说明

RT-DETR 是 transformer-based detector。YOLOv9/YOLOv12 上生成的扰动迁移到 RT-DETR 时，Recall-drop ASR 可能偏低，这是正常现象。建议论文同时报告：

- Recall-drop ASR
- Paired Object ASR

其中 Paired Object ASR 更接近攻击论文中的“clean 中已检测目标在 adv 中消失的比例”。
