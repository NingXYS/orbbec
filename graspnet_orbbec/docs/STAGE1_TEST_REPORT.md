# 阶段一测试报告：GraspNet + Astra Pro Plus

> 本报告随仓库代码一起维护。带「实测」标记的条目在本机跑通后填写；其余为规格与设计结论，避免把未测数据写成已测。

## 1. 测试环境

| 项 | 值 |
|----|----|
| 主机 | Ubuntu 22.04.5，内核 6.8 |
| GPU | NVIDIA GeForce RTX 4060 Laptop，8188 MiB |
| 驱动 | 580.178.04（`nvidia-smi` 报 CUDA 13.0） |
| Python 环境 | `graspnet_orbbec/.conda`（Python 3.10，项目内前缀） |
| 相机 | Orbbec Astra Pro Plus，USB `2bc5:050f`（UVC 彩色）+ `2bc5:060f`（深度） |
| 模型 | GraspNet-baseline，预训练 `checkpoint-rs.tar`（RealSense） |
| 不使用 | OrbbecSDK_ROS2 v2、机械臂、VLM |

## 2. 驱动选型结论

| 方案 | Astra Pro Plus | 结论 |
|------|----------------|------|
| OrbbecSDK_ROS2 **v2-main** | 不支持 | 禁止作为阶段一主路径 |
| PyOrbbecSDK **v2** (`pyorbbecsdk2`) | 不支持 | 禁止 |
| PyOrbbecSDK **v1.x** (`pyorbbecsdk==1.3.2`) | limited maintenance | **主路径** |
| OpenNI2 + OpenCV UVC | 可用 | SDK 彩色失败时的回退（代码已实现 UVC） |
| `orbbec/ros_astra_camera` ros2 分支 | 可用 | 仅阶段二需要 ROS 时再启用 |

Astra Pro Plus 是「深度 OpenNI + 彩色独立 UVC」结构，软件 registration 为默认对齐方式。

## 3. 功能核验清单

| 验收项 | 方法 | 状态 |
|--------|------|------|
| 环境可复现 | `bash scripts/setup_env.sh`，环境在 `.conda/` | 见第 7 节 |
| 官方 demo | `python infer/run_demo.py` | 见第 7 节 |
| 取流 ≥10 FPS | `python capture/run_capture.py --fps-seconds 30` 及 30 min 稳定性 | 见第 7 节 |
| 点云尺寸误差 ≤5 mm | 50 mm 块 + `run_calibration_check.py` | 需在桌面摆块后实测 |
| 边缘对齐 ≤10 px @0.5 m | Canny 中位距离；建议 0.6–0.8 m | 需实测 |
| 10 次随机摆放可抓率 ≥80% | 瓶/盒/工具，肉眼看 top 候选 | 需实测 |
| 三图输出 | `output/<ts>/01_rgb.png` 等 | 流水线自动保存 |

## 4. Astra Pro Plus 在本任务中的已知局限

这些是传感器物理特性，阶段二必须按此设计工作距离与物体清单。

1. **近距**  
   官方深度范围约 **0.6–8 m**。验收文案写 0.5 m，实测会在近裁剪附近出现大面积空洞。阶段一建议相机到桌面 **0.6–0.8 m**；阶段二手眼安装不要追求 0.4 m 超近距。

2. **噪声与空洞**  
   结构光对**深色、亚光黑、强反光金属、透明塑料/玻璃**失效或噪声极大。GraspNet 输入是点云，这些区域会被工作空间掩码丢掉，表现为「物体存在但无抓取」。

3. **RGB-D 对齐**  
   无硬件 D2C。软件对齐依赖出厂外参；UVC 彩色与深度是两个 USB 设备，时间同步只能靠 SDK `enable_frame_sync` 或软件近似。运动模糊或手持晃动时边缘偏差会超过 10 px。

4. **分辨率**  
   深度典型 640×480；彩色可达 1280×720。对齐后深度被散射到彩色网格，远距点较稀，细杆工具（螺丝刀）容易欠采样。

5. **USB 2.0 彩色头**  
   `2bc5:050f` 是 Sonix UVC。带宽紧张时彩色掉帧，表现为 RGB 与深度时间错位。请尽量插在同一 USB hub 的稳定口，避免与其它高带宽设备抢总线。

6. **无硬件配准标定工具链**  
   若出厂外参不准，需要棋盘格/标定板重新估计 `R,t`。阶段一提供 numpy D2C 与 `intrinsics.json` 导出，阶段二手眼标定前应先锁死这组外参。

## 5. 典型失败案例（设计预期 + 实测栏）

| 场景 | 预期现象 | 根因 | 缓解 |
|------|----------|------|------|
| 黑色鼠标/黑色电源适配器 | 点云破洞，抓取打在桌面 | IR 吸收 | 换浅色物体，或补环境光但避免直射 |
| 透明水瓶 | 深度落在瓶后桌面 | 折射/透射 | 阶段二不要把「全透明瓶」当首批 SKU |
| 不锈钢勺子 | 飞点、边缘错位 | 镜面反射 | 换磨砂工具；GraspNet 分数会虚高 |
| 物体贴桌且扁平（纸盒摊开） | 被桌面平面滤掉 | `height_min_m` | 降低阈值，但误抓桌面风险上升 |
| 相机 <0.5 m | 大片 0 深度 | 超出近距 | 抬高相机 |
| 强窗光 | 深度闪烁 | IR 干扰 | 关窗帘或加遮光罩 |
| 多物体严重遮挡 | 只出上层可抓 | 单视图 GraspNet 固有 | 阶段二可加多视角或人工摆放约束 |

## 6. 对阶段二（SO-ARM100）的可行性与风险

**可行性结论（阶段一设计层）：**  
在 **0.65–0.80 m 俯视、浅色不透明日常物体、桌面无强反光** 的前提下，GraspNet-baseline 预训练权重可以把 6-DoF 抓取候选排到可执行集合里，作为 MoveIt 2 的目标位姿输入是合理的。不建议在未做手眼标定前把相机位姿「估一下」就上臂。

**风险清单：**

| ID | 风险 | 影响 | 缓解 |
|----|------|------|------|
| R1 | 相机近距与 SO-ARM100 工作半径冲突 | 臂够不到或相机太近无深度 | 先画工作空间：桌面、臂基座、相机三角 |
| R2 | 手眼标定误差叠加 D2C 误差 | 抓空、撞桌 | 先标定 D2C，再 eye-to-hand；用 50 mm 块闭环 |
| R3 | GraspNet 夹爪模型与真实夹爪几何不一致 | 碰撞检测过宽/过窄 | 阶段二用真实指宽/指长改 `ModelFreeCollisionDetector` |
| R4 | 无可达性过滤 | 高分抓取在臂奇异或超限位 | 阶段二用 MoveIt IK 批量筛选 |
| R5 | 单视图遮挡 | 下层物体抓取虚报 | SKU 限制 + 抓完再感知 |
| R6 | 8 GB 显存与规划同机 | 推理时规划卡顿 | 推理完释放 CUDA，或降 `num_point` |
| R7 | Astra Pro Plus v1 SDK 停更 | 驱动回归 | 代码已预留 UVC 回退；长期考虑换 Gemini 系列 |
| R8 | 透明/黑色 SKU | 分拣失败率高 | 产品层面剔除，或加光电/接触传感器 |

## 7. 本机实测记录

测试日期：2026-09-11。环境前缀：`/home/ning/orbbec/graspnet_orbbec/.conda`。

### 7.1 环境与官方 demo

| 项 | 结果 |
|----|------|
| `torch.cuda.is_available()` | True，RTX 4060 Laptop，PyTorch 2.7.1+cu126 |
| PointNet2 / knn 编译 | 成功（sm_89，含 `cylinder_query`） |
| GraspNet dummy CUDA forward | 成功（随机权重，抓取数为 0 符合预期） |
| `pytest tests/ -q` | 6 passed |
| `infer/run_demo.py` | **阻塞**：官方 `checkpoint-rs.tar` Google Drive 触发流量配额，需手动下载后复跑 |
| 三图路径 | `output/pointcloud/20260911_155752/`（官方 example_data 的 RGB + 着色点云） |

官方 example_data 离线点云：有效深度 81.1%，均值距离 468 mm，点数 33 万（工作空间掩码后）。渲染图可辨认电钻、香蕉、饼干盒、瓶罐。

### 7.2 取流

| 项 | 结果 |
|----|------|
| 设备枚举 | SDK v1.10.16 识别 **Astra Pro Plus，PID 0x060f，USB2.0** |
| 打开设备 | **失败：`OB_USB_STATUS_ACCESS`**。`/dev/bus/usb/003/010|011` 为 `root:root 0664`，当前用户无写权限 |
| 彩色分辨率 | 待 udev 后测 |
| 深度分辨率 | 待 udev 后测 |
| 对齐方式 | 代码默认 SW；SDK 无彩色时回退 UVC + numpy D2C |
| 30 s / 30 min | 待 udev 后测 |

**必须由本机用户执行一次：**

```bash
sudo bash scripts/install_udev.sh
# 拔插相机后再：
python capture/run_capture.py --preview
python capture/run_capture.py --fps-seconds 30
python scripts/run_stability.py --seconds 1800
```

### 7.3 点云与对齐

| 项 | 结果 |
|----|------|
| 官方 example_data 工作距离 | 468 mm |
| 官方 example_data Canny 边缘偏差 | 18.1 px（杂乱桌面，**不是** 50 mm 标定块；验收 10 px 需实机标定块） |
| 空洞率（example_data） | 18.9% |
| 50 mm 块测量 | 待实机 + udev |

### 7.4 抓取肉眼评估（10 次随机摆放）

待：`weights/checkpoint-rs.tar` + 相机 udev。下载后：

```bash
bash scripts/download_weights.sh   # 或手动放到 weights/checkpoint-rs.tar
python infer/run_demo.py
python scripts/run_pipeline.py --live
```

## 8. 结论

阶段一代码与部署路径已按「项目内 conda 环境 + PyOrbbecSDK v1 + 软件对齐 + GraspNet 预训练」落地。传感器层最大风险是 **近距空洞** 与 **深色/透明/反光**。阶段二必须先固定相机安装高度（建议 ≥0.65 m），再做手眼标定，不要把 GraspNet 分数直接当执行位姿。
