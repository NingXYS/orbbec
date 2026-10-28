# DaBai DC1 接入

DC1 通过 `capture/dabai_dc1.py` 接入现有 `open_camera`，返回原有 `RGBDFrame`，复用点云、GraspNet、可视化与点名接口。此入口只做相机采集和抓取识别，不连接或控制机械臂。默认配置已切到 DC1，旧的完整 Gemini 配置保存在 `configs/gemini336l.yaml`。

## 启动

关闭 OrbbecViewer 和其他相机采集进程，保持桌面场景静止，镜头到目标约 0.5～1 米：

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh

# 仅相机预览，q 退出
python capture/run_capture.py --preview

# 连续 GraspNet 推理；先禁用终端点名，避免调用外部 VLM
python scripts/run_live_gui.py --no-target

# 单帧推理并保存 RGB、点云、抓取可视化和报告
python scripts/run_pipeline.py --live --once

# 连续取流性能、边缘对齐诊断（各自独立运行，不同时占用相机）
python capture/run_capture.py --fps-seconds 10 --min-fps 10
python capture/run_capture.py --align-check --no-window
```

GUI：`q/Esc` 退出，空格暂停/继续推理，`s` 保存，`3` 打开可旋转点云。第四行提示 `RGB-D unsynced`。如需原有点名功能，去掉 `--no-target`，并沿用已有 VLM 配置。

2026-10-26：原配置 `qwen3-vl-flash-2026-01-22` 已被[官方列入 10 月 10 日下线清单](https://platform.qianwenai.com/docs/changelog/model-deprecations/snapshot-models)。默认 VLM 已改为官方推荐替代系列的固定版本 `qwen3.6-flash-2026-04-16`，继续关闭思考并要求 JSON 输出。替代模型定价与旧版不同，以百炼控制台为准。已运行的程序需退出并重启才能读取新配置。

HTTP 400/401/403/404/422 现在归类为配置错误并停止重试；需核对模型名、地域、密钥及权限。`torch.cross` 提示是第三方代码的弃用警告，不是点名失败原因。Open3D 只在第一份非空点云到来后建立包围盒，并只在推理结果更新时重建网格；队列满丢帧仍可能在渲染慢于采集时出现，不能据此判定相机故障。如实时显示较慢，可临时增加 `--no-o3d` 关闭旁路 3D 窗口。

退出时会等待正在进行的一次推理完成；加载模型失败或启动被中断时也会释放相机。实测记录见 [DC1 测试报告](DABAI_DC1_TEST_REPORT.md)。

## 参数与数据约定

| 项目 | 当前配置/行为 |
|---|---|
| 设备 | DaBai DC1，`CC1N162037K`，深度 PID `0x0657`、彩色 PID `0x0557` |
| 环境 | 项目 `.conda` Python 3.10；已有 SDK v1 原生库 1.10.16；不需要安装 ROS 驱动 |
| 彩色请求 | 640×480 MJPG，30 FPS |
| 原始深度请求 | 640×400 Y11，30 FPS；SDK 解包为 uint16/Y16 |
| 对齐 | `camera.align_mode: HW`；检查设备对齐开关、启动后的 D2C 参数以及实际帧尺寸 |
| 对齐输出 | 640×480，彩色坐标系；深度 Z 采用 SDK 硬件 D2C 输出定义 |
| 深度单位 | `raw × get_depth_scale()` 为毫米；`depth_m` 为米；保存的 `depth.png` 固定为毫米 |
| 内参来源 | `pipeline.start(config)` 后读取 `get_camera_param()`，不使用其他相机的参数 |
| 帧同步 | 不调用不支持的 SDK 帧同步；只接受两路完整、都已更新且主机接收差 ≤100 ms 的帧 |
| 激光 | 采集期间按 `camera.laser` 设置发射器，退出或打开失败后尝试恢复原值 |

30 FPS 是请求流配置，不代表完整且更新的 RGB-D 对也达到 30 FPS。应通过实际计数测量；主机接收差也不代表曝光同步精度。设备时间戳属于不同的时钟，不能相减作为同步误差。

配置指定序列号时，会在打开设备后核对实际序列号，避免 OpenNI 枚举阶段序列号为空造成误选。多台 DC1 未指定序列号会报错。`camera.backend: auto` 支持 DC1 PID 检测，但流尺寸仍由配置决定；Gemini 与 DC1 同时连接时 auto 优先 Gemini，实际切换请选对应的完整配置。

## Viewer 文件

`configs/dabai_dc1_viewer_reference.ini` 是本次用户导出的原始参考副本，仅用于核对，不是运行时回退参数。RGB 为：

```text
fx=fy=490.743073
cx=319.486542
cy=211.561279
width=640 height=480
```

文件内部 DepthIntrinsic 为 640×480，且内参与 RGB 相同、D2C 外参为单位变换。它对应对齐参数，不能用于原始 640×400 深度的反投影。程序从正在运行的硬件 D2C 管线读取相应参数，不重复应用原始 Depth→RGB 变换，也不按尺寸比例猜内参。

如果 D2C 参数或帧尺寸不匹配、深度格式/scale 异常，会明确报错，不把 resize 当配准；持续没有有效深度时超时报错。镜头图像保持设备原始朝向，不做翻转、裁剪或额外去畸变。

## 保存和验证边界

保存文件遵循现有 `color.png`、毫米 `depth.png`、`intrinsics.json` 格式。额外记录 `capture_metadata`（序列号、原始流配置、HW 对齐、不同步声明、两路主机接收时间和接收差）、`source_depth_scale_mm` 及 SDK 畸变参数。

`aligned=true` 表示 SDK 硬件 D2C 配置、参数和帧尺寸检查通过；`geometry_verified=false` 表示尚未用独立物理基准验收几何精度。现有点云代码使用针孔反投影，保存的畸变参数不意味着已执行去畸变。边缘诊断和模型候选不能替代深度绝对距离、物体尺寸、D2C Z 语义和畸变影响的独立验收。

历史手眼对话后期使用另一台 `CC1N162032Y`。其内参、手眼矩阵和深度偏差不得直接用于本次 `CC1N162037K`。当前接入不采用历史深度补偿，也不输出机器人基座坐标。

## 切回 Gemini 336L

```bash
python scripts/run_live_gui.py --config configs/gemini336l.yaml
```

切换相机应退出并重启进程，避免 SDK v1/v2 以相同模块名混载。
