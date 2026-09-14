# 阶段一部署文档

目标：一台干净的 Ubuntu 22.04 + NVIDIA GPU（≥8 GB）机器，1 小时内跑通官方 GraspNet demo，并在 Astra Pro Plus 上取到对齐 RGB-D。

## 0. 硬件清单

| 项目 | 要求 |
|------|------|
| 系统 | Ubuntu 22.04 |
| GPU | NVIDIA，显存 ≥ 8 GB（本仓库在 RTX 4060 Laptop 8 GB 上验证） |
| 驱动 | NVIDIA 驱动，CUDA 能力 ≥ 12.x（`nvidia-smi` 能看到 GPU） |
| 相机 | Orbbec Astra Pro Plus（`lsusb` 可见 `2bc5:050f` 彩色 + `2bc5:060f` 深度） |
| 磁盘 | 建议剩余 ≥ 12 GB（PyTorch 轮子约 2.5 GB + 环境） |

ROS 2 Humble 阶段一**不需要**。不要用 OrbbecSDK_ROS2 v2 驱动这台相机。

## 1. 系统包

```bash
sudo apt update
sudo apt install -y git build-essential cmake libusb-1.0-0-dev \
    python3-dev pkg-config
```

安装 Miniconda（若尚无）：

```bash
# 已有 conda 可跳过
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

## 2. 获取代码

```bash
# 本仓库已包含 third_party/graspnet-baseline、graspnetAPI、pyorbbecsdk-v1
cd /path/to/orbbec/graspnet_orbbec
```

若 `third_party` 为空，再执行：

```bash
mkdir -p third_party && cd third_party
git clone --depth 1 https://github.com/H-Freax/GraspNet-PointNet2-Pytorch-General-Upgrade.git graspnet-baseline
git clone --depth 1 https://github.com/graspnet/graspnetAPI.git
git clone --depth 1 --branch main https://github.com/orbbec/pyorbbecsdk.git pyorbbecsdk-v1
cd ..
```

这里的 GraspNet 是官方 baseline 的现代 PyTorch 移植，权重与网络结构不变。

## 3. 创建项目内 conda 环境

环境前缀固定为 **`graspnet_orbbec/.conda`**，不要 `conda create -n ...`。

```bash
bash scripts/setup_env.sh
source scripts/activate.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`setup_env.sh` 会：

1. `conda create --prefix .conda python=3.10`
2. 安装 PyTorch **cu126** 轮子（官网失败时可改阿里云 pytorch-wheels）
3. `pip install -r requirements.txt` 和本地 `graspnetAPI`
4. **从源码编译 PyOrbbecSDK v1**（PyPI `pyorbbecsdk==1.3.2` 的 manylinux 轮子实际是 macOS dylib，不能用）
5. 安装 CUDA **12.6** nvcc（NVIDIA redistributable，与 cu126 对齐）
6. 给 PointNet2 / knn 打 PyTorch 2.x 补丁并编译 `sm_89`

编译针对 RTX 40 系默认 `TORCH_CUDA_ARCH_LIST=8.9`。其它 GPU 请覆盖该变量。

## 4. USB 权限（必须 sudo，否则深度打不开）

Astra Pro Plus 深度节点默认 `root:root 0664`，无写权限时 SDK 报 `OB_USB_STATUS_ACCESS`。

```bash
sudo bash scripts/install_udev.sh
# 拔插一次相机
lsusb | grep -i 2bc5
```

应看到：

```text
2bc5:050f  USB 2.0 Camera          # 彩色 UVC
2bc5:060f  ORBBEC Depth Sensor     # 深度
```

## 5. 预训练权重

```bash
bash scripts/download_weights.sh
ls -lh weights/checkpoint-rs.tar
```

若 Google Drive 报 “Too many users have viewed or downloaded this file recently”，改用百度网盘或浏览器登录后下载，把文件放到 `weights/checkpoint-rs.tar`。没有这份权重就无法跑 GraspNet 推理（取流和点云不受影响）。

Kinect 权重 `checkpoint-kn.tar` 可选，桌面场景更推荐 RealSense 权重。

## 6. 冒烟测试（不接相机）

```bash
source scripts/activate.sh
python -m pytest tests/ -q
python infer/run_demo.py
```

`output/official_demo/<时间戳>/` 下应有 `01_rgb.png`、`02_cloud.png`、`03_cloud_grasps.png`。

## 7. 接上 Astra Pro Plus

```bash
source scripts/activate.sh
python capture/run_capture.py --preview          # q 退出，s 存盘
python capture/run_capture.py --fps-seconds 30
python pointcloud/run_pointcloud.py --live --save --expect-block-mm 50
python scripts/run_pipeline.py --live
```

工作距离建议 **0.6–0.8 m**。Astra Pro Plus 官方近距约 0.6 m，0.5 m 处深度空洞会明显增加。

## 8. 稳定性（验收：30 分钟 ≥10 FPS）

```bash
python scripts/run_stability.py --seconds 1800
```

冒烟可用 `--seconds 60`。

## 9. 常见问题

**`No device found` / 深度打不开**  
udev 未生效或当前用户打不开 USB。确认 `sudo bash scripts/install_udev.sh` 后拔插相机。

**只有彩色没有深度**  
深度是 OpenNI 协议设备，必须用 PyOrbbecSDK v1，不要装 `pyorbbecsdk2`。

**只有深度没有彩色**  
Astra Pro Plus 彩色是独立 UVC。代码会自动按 VID/PID 打开 `/dev/video*`。不要抢占笔记本内置摄像头。

**编译 PointNet2 失败**  
确认 `nvcc --version` 来自 `.conda`，且 `echo $CUDA_HOME` 指向 `.conda`。不要用系统 CUDA 11.5。

**`Unknown layout` / `type().is_cuda`**  
未跑 `scripts/patch_cuda_ext.py`。重新执行 `setup_env.sh`。

**显存不足**  
把 `configs/default.yaml` 里 `graspnet.num_point` 降到 12000，`num_view` 降到 120。

## 10. 卸载

```bash
rm -rf graspnet_orbbec/.conda
# udev：sudo rm /etc/udev/rules.d/99-obsensor-libusb.rules /etc/udev/rules.d/99-orbbec-all.rules
```
