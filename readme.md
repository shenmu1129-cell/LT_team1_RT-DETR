# 基于 Ultralytics 的 RT-DETR 训练流程

本项目使用 Ultralytics RT-DETR 作为跨检测器泛化实验中的第四个目标检测器。数据集继续使用 YOLO 格式，不需要转换成 COCO，也不依赖 OpenMMLab 生态。

## 为什么不用 ATSS/MMDetection

原计划使用 ATSS 作为第四个检测器，但当前服务器环境不适合继续走 OpenMMLab 路线：

- `mmcv` 编译失败。
- 服务器硬盘空间有限，不希望再新建 conda 环境。
- 本项目后续不再使用 OpenMMLab、MMDetection、`mmcv`、`mmengine`。

因此第四个检测器改为 Ultralytics RT-DETR，例如 `rtdetr-l.pt`。

## 最终检测器组合

跨检测器泛化实验最终使用：

- Faster R-CNN
- RetinaNet
- FCOS
- RT-DETR

四个模型覆盖的检测范式如下：

- Faster R-CNN：two-stage detector
- RetinaNet：anchor-based one-stage detector
- FCOS：anchor-free one-stage detector
- RT-DETR：transformer-based detector

论文中可以这样描述：

> To evaluate cross-detector generalization, we select Faster R-CNN, RetinaNet, FCOS, and RT-DETR, covering two-stage, anchor-based one-stage, anchor-free one-stage, and transformer-based detection paradigms.

## 新增文件

- `data/cctsdb.yaml`
- `data/tt100k.yaml`
- `tools/check_yolo_dataset.py`
- `tools/train_rtdetr.py`
- `tools/val_rtdetr.py`
- `scripts/train_rtdetr_cctsdb.sh`
- `scripts/train_rtdetr_tt100k.sh`
- `scripts/val_rtdetr_cctsdb.sh`
- `scripts/val_rtdetr_tt100k.sh`
- `scripts/predict_rtdetr_sample.sh`
- `scripts/run_rtdetr_cctsdb_bg.sh`
- `scripts/run_rtdetr_tt100k_bg.sh`

这些文件只服务于 RT-DETR，不会修改 Faster R-CNN、RetinaNet、FCOS 的训练代码。

## TT100K 类别名提醒

当前 `data/tt100k.yaml` 中使用的是 `TODO_CLASS_0` 到 `TODO_CLASS_44` 占位类别名，因为当前项目里没有找到可复用的 TT100K yaml 或 `classes.txt`。

正式训练或汇报 TT100K 结果前，必须把这 45 个类别名替换成真实类别名，并且顺序必须和 YOLO 标签里的 class id 完全一致。否则类别指标和结果解释会对应错误。

只要 `data/tt100k.yaml` 里仍然存在 TODO 类别名，数据集检查脚本和训练脚本会主动失败，避免误训练。

## 环境依赖

建议使用现有 `wwt310` 环境：

```bash
conda activate wwt310
pip install ultralytics
```

本流程不使用 OpenMMLab、MMDetection、`mmcv`、`mmengine`。

## 检查数据集

正式训练前先检查 YOLO 数据集：

```bash
python tools/check_yolo_dataset.py --data data/cctsdb.yaml
python tools/check_yolo_dataset.py --data data/tt100k.yaml
```

检查脚本会验证：

- yaml 文件是否存在。
- `path`、`train`、`val`、`test` 是否配置且路径是否存在。
- 图片目录和标签目录是否存在。
- 图片数量和标签数量是否大致匹配。
- 标签是否为 YOLO 格式：`class x_center y_center width height`。
- class id 是否在 `[0, nc-1]` 范围内。
- 如果发现类别 id 越界，会打印具体文件名和行号。
- 空标签文件只给 warning，因为可能存在无目标图片。
- 输出 train、val、test 的图片数量、标签数量和类别统计。

TT100K 会自动识别以下两种常见目录布局：

- `train/images`、`test/images`、`train/labels`、`test/labels`
- `images/train`、`images/test`、`labels/train`、`labels/test`

## 1 Epoch 调试

先用 CCTSDB 跑 1 个 epoch 确认环境、数据路径和显存都正常：

```bash
GPU_ID=2 EPOCHS=1 BATCH=2 bash scripts/train_rtdetr_cctsdb.sh
```

## 正式训练

训练 CCTSDB：

```bash
GPU_ID=2 bash scripts/train_rtdetr_cctsdb.sh
```

训练 TT100K：

```bash
GPU_ID=2 bash scripts/train_rtdetr_tt100k.sh
```

默认输出目录：

- `outputs/rtdetr_cctsdb`
- `outputs/rtdetr_tt100k`

部分 Ultralytics 版本会把 RT-DETR 的实际结果目录放到 `runs/detect/outputs/rtdetr_cctsdb` 或 `runs/detect/outputs/rtdetr_tt100k`。训练脚本结束时会打印真实的 `best.pt` 和 `last.pt` 路径，以最终日志为准。

每次训练会在输出目录中保存解析后的数据集 yaml，并生成类似：

- `outputs/rtdetr_cctsdb/weights/best.pt`
- `outputs/rtdetr_cctsdb/weights/last.pt`
- `outputs/rtdetr_tt100k/weights/best.pt`
- `outputs/rtdetr_tt100k/weights/last.pt`

如果日志显示保存到了 `runs/detect/outputs/...`，权重路径也以日志中的路径为准。

## 后台训练

后台训练 CCTSDB：

```bash
GPU_ID=2 EPOCHS=50 BATCH=4 bash scripts/run_rtdetr_cctsdb_bg.sh
```

后台训练 TT100K：

```bash
GPU_ID=2 EPOCHS=80 BATCH=4 bash scripts/run_rtdetr_tt100k_bg.sh
```

脚本会输出后台进程 PID 和日志路径。查看训练进度：

```bash
tail -f logs/rtdetr_cctsdb_*.log
```

查看进程是否还在运行：

```bash
ps -f -p PID
```

如果需要停止后台训练：

```bash
kill PID
```

## 验证

验证 CCTSDB：

```bash
GPU_ID=2 bash scripts/val_rtdetr_cctsdb.sh
```

验证 TT100K：

```bash
GPU_ID=2 bash scripts/val_rtdetr_tt100k.sh
```

验证脚本会输出 precision、recall、mAP50、mAP50-95，并保存指标摘要：

- `outputs/rtdetr_cctsdb/metrics_summary.txt`
- `outputs/rtdetr_tt100k/metrics_summary.txt`

## 样例预测

```bash
GPU_ID=2 bash scripts/predict_rtdetr_sample.sh
```

也可以覆盖权重和输入图片目录：

```bash
GPU_ID=2 WEIGHTS=outputs/rtdetr_tt100k/weights/best.pt SOURCE=/path/to/images bash scripts/predict_rtdetr_sample.sh
```

## 显存不足时

可以减小 batch、图像尺寸和 dataloader workers：

```bash
GPU_ID=2 BATCH=2 IMGSZ=512 WORKERS=4 bash scripts/train_rtdetr_cctsdb.sh
```

## TT100K 小目标效果一般时

可以尝试增大输入尺寸并延长训练：

```bash
GPU_ID=2 IMGSZ=800 BATCH=2 EPOCHS=100 bash scripts/train_rtdetr_tt100k.sh
```

## 常用参数覆盖

训练脚本支持通过环境变量覆盖常用参数，例如：

```bash
GPU_ID=2 MODEL=rtdetr-l.pt IMGSZ=640 EPOCHS=80 BATCH=4 WORKERS=8 bash scripts/train_rtdetr_tt100k.sh
```

验证脚本也可以覆盖权重路径：

```bash
GPU_ID=2 WEIGHTS=outputs/rtdetr_cctsdb/weights/best.pt bash scripts/val_rtdetr_cctsdb.sh
```
