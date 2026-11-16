> [!IMPORTANT]
> **Current status (2026-10-28):** This subsystem has progressed beyond the older stage boundaries below. The repository now includes a MoveIt 2 dual-pose execution toolkit with real AR6E-R700 validation, plus SAM2 wire segmentation and a 3D arc-length wire grasp policy. See the [repository root README](../README.md), [MoveIt validation](docs/MoveIt双位姿验证记录_20260914.md), and [wire grasp policy](docs/WIRE_GRASP_POLICY.md). Sections describing MoveIt as “not done” are retained only as historical stage context.

# GraspNet + Orbbec RGB-D 相机

## Piper X 当前真机入口

官方启动脚本已统一到 `bash ../bringup/piper_x/start_real_control.sh`。采图、识别和MoveIt抓取适配仍在原位置，完整操作见 [真机启动手册](docs/REAL_GRASP_STARTUP.md)。旧“视觉”入口仅兼容转发；仿真和历史实验未迁移。

当前默认相机为 **DaBai DC1**，保留 Gemini 336L 和 Astra Pro Plus 后端。GraspNet-baseline（SJTU, CVPR 2020）推理链路：对齐 RGB-D → 点云 → 6-DoF 抓取位姿 → 可视化。相机接入与推理入口不发送机械臂运动命令。

## 线材手眼联调

Piper X 当前为 **Phase 1.8离线准备，Phase 2 BLOCKED**。新预览从金属/绝缘原始交界总后退14mm，保留141.5mm名义TCP；完整夹爪模型投影与22mm接触带分开检查。原始分割和Shift交界确认保持，端子可用多边形人工标注并绑定原图哈希保存；人工标注不冒充自动检测或物理安全验收。

新版文件图像入口：`python scripts/run_wire_offline.py --snapshot enabled --gui --out /tmp/wire_review_new`，另一张用`--snapshot raised`。用法、保存/重载与已知限制见 [Phase1.8离线复核](docs/WIRE_OFFLINE_PHASE18.md)。原10mm仅保留历史比较；旧手眼试验记录仍见 [线材手眼联调](docs/WIRE_HANDEYE_TRIAL.md)，不代表当前实机许可。事故锁、实机执行限制保持，不能由离线预览发运动指令。

视觉基座坐标到官方 MoveIt 的应用适配已新增：

```bash
recognition="output/wire_grasp/real_20261103_164546_base-12_grasp135_01/recognition.json"
candidate_id="A"

scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id"
```

默认只规划；只有视觉目标、实测工具和执行安全检查全部通过后，才可使用 `--execute`。接口、门槛和当前阻塞见 [视觉坐标到官方MoveIt抓取适配](docs/MOVEIT_VISION_GRASP.md)。

完整的双终端启动、采图、候选确认、手动使能、只规划和真机执行步骤见 [Piper X 视觉抓取真机启动手册](docs/REAL_GRASP_STARTUP.md)。

当前支持固定桌面平面模式：使用 `configs/support_plane_piper_x_02.json` 后，新采图不再要求棋盘入镜。默认识别采用SAM2中心线并由操作者人工确认交界；YOLO端子检测默认仅作辅助证据。

## DaBai DC1

本机环境已就绪，关闭 OrbbecViewer 后运行：

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
python scripts/run_live_gui.py --no-target
```

默认绑定序列号 `CC1N162037K`；更换另一台 DC1 时修改 `configs/default.yaml` 的 `camera.serial`，程序会从实际设备重新读取参数。DC1 的 RGB 与深度不支持同步曝光，请固定相机并保持场景静止。采集时开启激光发射器，退出恢复之前的开关状态。

取流自检：`python capture/run_capture.py --preview`。单帧完整推理：`python scripts/run_pipeline.py --live --once`。使用原 Gemini 配置：`python scripts/run_live_gui.py --config configs/gemini336l.yaml`。详细约束见 [DC1 接入说明](docs/DABAI_DC1.md)。


离线 RGB-D 回放保留深度图原始尺寸、内参、对齐标记及深度比例。未对齐帧可读取，彩色点云与基于彩色框的三维定位要求已对齐帧。`FileCamera.get_intrinsics()` 可在读帧前调用。重新保存回放数据时沿用原文件的深度单位；写图失败会报错。

## YOLO 权重效果检验

根目录 `best.pt` 为单类别 YOLOv8x 实例分割权重。直接运行：

```bash
cd /home/ning/orbbec
bash graspnet_orbbec/scripts/run_yolo.sh
```

默认使用当前 Orbbec 的彩色流和可用 GPU；支持 `--source 图片/目录/视频` 或 `--source 0` 使用普通摄像头。窗口可调置信度，`s` 保存、`q` 退出。结果及逐帧记录保存在 `output/yolo/`。使用说明、依赖隔离和实测限制见 [YOLO 检验说明](docs/YOLO_TEST.md)。

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
| `capture/` | DC1 / Gemini / Astra 取流与 RGB-D 对齐 | `python capture/run_capture.py --preview` |
| `pointcloud/` | 反投影、工作空间、尺寸/对齐评估 | `python pointcloud/run_pointcloud.py --live --save` |
| `infer/` | GraspNet 推理 + 规则过滤 | `python infer/run_infer.py --live` |
| `segmentation/` | SAM2 自动线材掩码、端部候选及本地端子模型 | `python scripts/run_live_gui.py --wire-auto --no-target` |
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

DaBai DC1 使用项目现有 **PyOrbbecSDK v1 / Orbbec SDK 1.10.16**，彩色 `640×480 MJPG`、原始深度 `640×400 Y11`，硬件 D2C 后两路输出 `640×480`。SDK 解包深度为 `Y16`，按每帧 scale 换算为米。当前默认后端 `dabai_dc1`；SDK v1 与 Gemini 使用的 v2 不能在同一进程混用。

Astra Pro Plus **不在** OrbbecSDK_ROS2 v2 支持列表。本仓库使用 **PyOrbbecSDK v1.x**（`pyorbbecsdk==1.3.2`），软件 D2C。若 SDK 打不开独立 UVC 彩色头（PID `0x050f`），自动回退 OpenCV V4L2 + numpy 软件配准。

## 阶段边界

- 阶段一（本仓库）：取流、点云、GraspNet 推理、可视化、测试报告
- 阶段二：SO-ARM100、手眼标定、MoveIt 2（不做）
- 阶段三：VLM / 语音（不做）
- 不重新训练 GraspNet，只用 `checkpoint-rs.tar`

## 文档

- [docs/WIRE_SEGMENTATION.md](docs/WIRE_SEGMENTATION.md) 线材分割、端部候选与原图复现
- [docs/DEPLOY.md](docs/DEPLOY.md) 部署
- [docs/STAGE1_TEST_REPORT.md](docs/STAGE1_TEST_REPORT.md) 阶段一测试报告（Astra）
- [docs/phase2_report.md](docs/phase2_report.md) 阶段二整体验收（336L 点名闭环，进臂前）
- [docs/controller_recon.md](docs/controller_recon.md) 厂家控制器（Jetson Thor）只读侦察
- [docs/cross_host_ros.md](docs/cross_host_ros.md) 笔记本 Humble ↔ 控制器 Jazzy 跨机连通实测
- [docs/moveit_safe_exec.md](docs/moveit_safe_exec.md) MoveIt 双位姿安全执行步骤
- [docs/MoveIt双位姿验证记录_20260914.md](docs/MoveIt双位姿验证记录_20260914.md) 2026-09-14 真机双位姿验收记录
- [docs/task3_closeout.md](docs/task3_closeout.md) 任务 3 结项（MoveIt 首次程序下发）

迁移到其他电脑时的目录布局、环境与重新编译步骤见 [迁移路径说明](docs/PORTABLE_DEPLOYMENT.md)。
