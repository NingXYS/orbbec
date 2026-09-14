# GraspNet + Orbbec Astra Pro Plus（阶段一）

在 **Astra Pro Plus** 上跑通 GraspNet-baseline（SJTU, CVPR 2020）推理：对齐 RGB-D → 点云 → 6-DoF 抓取位姿 → 可视化。不接机械臂。

## 环境位置

Conda 环境在**本仓库内**，不占用 `~/miniconda3/envs`：

```text
graspnet_orbbec/.conda/     # Python 3.10 + PyTorch CUDA
```

```bash
cd /path/to/orbbec/graspnet_orbbec
bash scripts/setup_env.sh          # 首次：装依赖并编译 CUDA 扩展
source scripts/activate.sh         # 之后每次进入环境
sudo bash scripts/install_udev.sh  # 必须：否则深度报 OB_USB_STATUS_ACCESS
bash scripts/download_weights.sh   # RealSense 预训练权重；GDrive 配额满时请手动放到 weights/
```

干净机器按 [docs/DEPLOY.md](docs/DEPLOY.md) 操作，目标 1 小时内跑通官方 demo。

## 模块

| 目录 | 职责 | 独立入口 |
|------|------|----------|
| `capture/` | Astra Pro Plus 取流、软件 RGB-D 对齐 | `python capture/run_capture.py --preview` |
| `pointcloud/` | 反投影、工作空间、尺寸/对齐评估 | `python pointcloud/run_pointcloud.py --live --save` |
| `infer/` | GraspNet 推理 + 规则过滤 | `python infer/run_infer.py --live` |
| `viz/` | RGB / 点云 / 抓取框三图 | `python viz/run_viz.py --from-dir output/...` |

端到端：

```bash
python scripts/run_live_gui.py                   # 连续规划 GUI（推荐）
python scripts/run_pipeline.py --live            # 同上
python scripts/run_pipeline.py --once --live     # 只拍一帧并保存
python infer/run_demo.py                         # 官方 example_data 单次
```

连续界面按键：`q` 退出，空格暂停推理，`s` 存盘，`3` 打开可旋转的 3D 抓取窗口。

产物在 `output/<时间戳>/`：

- `01_rgb.png` 原始 RGB
- `02_cloud.png` 着色点云
- `03_cloud_grasps.png` 叠加抓取框
- `grasps.json` / `report.json`

## 相机驱动

Astra Pro Plus **不在** OrbbecSDK_ROS2 v2 支持列表。本仓库使用 **PyOrbbecSDK v1.x**（`pyorbbecsdk==1.3.2`），软件 D2C。若 SDK 打不开独立 UVC 彩色头（PID `0x050f`），自动回退 OpenCV V4L2 + numpy 软件配准。

## 阶段边界

- 阶段一（本仓库）：取流、点云、GraspNet 推理、可视化、测试报告
- 阶段二：SO-ARM100、手眼标定、MoveIt 2（不做）
- 阶段三：VLM / 语音（不做）
- 不重新训练 GraspNet，只用 `checkpoint-rs.tar`

## 文档

- [docs/DEPLOY.md](docs/DEPLOY.md) 部署
- [docs/STAGE1_TEST_REPORT.md](docs/STAGE1_TEST_REPORT.md) 阶段一测试报告（Astra）
- [docs/phase2_report.md](docs/phase2_report.md) 阶段二整体验收（336L 点名闭环，进臂前）
