# DaBai DC1 接入测试记录（2026-10-26）

结论：真实 DC1 → HW D2C RGB-D → 点云 → CUDA GraspNet → 保存结果及连续界面已接通。当前完成相机和推理链路验证，未完成毫米级几何精度验收，未连接或控制机械臂。

## 实机环境和修复

- 设备为 `DaBai DC1 / CC1N162037K / RD1001 / USB2.0`，与此次 Viewer 导出文件一致。历史手眼对话后期使用的 `CC1N162032Y` 参数没有用于本次接入。
- 使用现有项目 `.conda`、PyTorch `2.7.1+cu126`、CUDA GPU 与 SDK v1 原生库 `1.10.16`。临时对比官方 Viewer 的 `1.10.37` 后发现零深度在两个版本均存在，未替换原环境。
- 初始彩色近乎全黑；用户调整后恢复图像。深度仍全零，随后实读 `OB_PROP_LASER_BOOL=False`。采集期间开启激光后，深度恢复。程序在退出/打开失败后恢复原开关。
- DC1 不支持 SDK `enable_frame_sync`，且设备时间戳不共时钟。新后端不调用此接口，仅筛选完整、两路均更新且主机接收差不超过 100 ms 的帧。界面显示不同步提示。
- 原始配置为 RGB 640×480 MJPG、Depth 640×400 Y11；实际硬件 D2C 输出 640×480 Y16，`get_depth_scale()=1.0`（毫米/单位）。内参为 `fx=fy=490.7430725097656, cx=319.4865417480469, cy=211.561279296875`，与 Viewer 六位小数舍入一致。
- GUI 中断退出曾暴露旧的后台推理线程清理竞态，导致异常终止；已增加等待当前推理结束，并在模型加载失败时释放相机。修复后复测无 traceback/terminate，SDK 正常完成停流与设备销毁日志。

## 验证结果

| 检查 | 实测结果 | 证据 |
|---|---|---|
| 真实 RGB-D 存盘 | 成功，内参/畸变/不同步元数据齐全 | `output/dc1_first_capture/` |
| 桌面真实 GPU 推理 | 有效深度 67.8%，过滤前 42 个、过滤后 4 个候选 | `output/dc1_tabletop/report.json` |
| 真实点云与抓取图 | RGB、点云、3D 抓取与 RGB 投影图已保存 | `output/dc1_tabletop/01_rgb.png` 至 `04_rgb_grasps.png` |
| 连续取流 | 10 秒 300 对完整且均更新的帧，29.92 FPS | `output_dc1_fps.log` |
| 连续 GUI | 进入实时界面，未报读帧/推理失败；15 秒后主动 SIGINT 清理通过 | `output_dc1_gui_clean_exit.log` |
| 退出恢复 | 设备可重新打开；`LASER_AFTER_CLOSE False` | `output_dc1_device_release.log` |
| 单元和现有回归测试 | 65 项通过 | `python -m pytest -q tests` |

测试覆盖：0.5 mm 原始 scale 到米/毫米文件的换算，保存读回，尺寸不匹配、非法 scale、未解包格式拒绝，过旧帧与重复帧过滤，全零深度超时，序列号不匹配拒绝，启动失败/KeyboardInterrupt 的停流及发射器恢复，以及 GUI 等待未完成推理。

## 对齐诊断与未验收项

不得将下面两种指标混为同一精度结论：

1. 桌面单帧 `report.json` 的旧全图 RGB 边缘距离为 **11.91 px**，超过其 10 px 阈值。该指标也包含没有对应深度几何边的 RGB 纹理/背景边，报告中的告警保留。
2. 最后另采 8 帧，使用项目已有的内部有效区、15 px 搜索范围匹配几何边方法，7 帧获得指标，中位数 **2.7386 px**（单帧约 1.37～3.69 px），见 `output/dc1_alignment/align.json` 和 `align_panel.png`。其 `pass_align_3px=true` 仅表示当前图像诊断中位数小于 3 px，不代表已知 0.5 米距离下的精度验收，也不消除第一项告警。

帧率测试的全图平均深度变化范围约 198 mm；没有恒定距离参考物，不能将它当作传感器漂移或精度。上述采样场景并非统一受控测量场景。

SDK 硬件对齐的开关、后处理参数和输出尺寸已验证；硬件 D2C 的 Z 定义、畸变对针孔反投影的影响、独立距离/物体尺寸误差仍需物理基准验收。保存数据保留 `geometry_verified=false`。本轮没有采用历史约 9 mm 偏差补偿，没有使用手眼矩阵，也没有进行机器人抓取测试。

SDK 1.10.16 退出时会打印 OpenNI stop/cancel 警告；连续多次重新打开和开关恢复已实测成功。GUI 的旧中断崩溃日志保留在 `output_dc1_gui.log`，最终修复后的记录是 `output_dc1_gui_clean_exit.log`，不能把前者当作最终测试结果。

## 文件和运行方式

Viewer 参考副本为 `configs/dabai_dc1_viewer_reference.ini`，SHA256：`e71f3feaf314de5b1e8116a8c8990294ad5fb2b91ca955c15f66dff1b9455176`。程序使用当前设备启动后的 SDK 参数，该 INI 不作为原始 Depth 或运行时回退标定。

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
python scripts/run_live_gui.py --no-target
```

详细参数、保存约定和切回 Gemini 的方法见 [接入说明](DABAI_DC1.md)。本轮未安装系统包、修改 udev/CAN 或提交/推送 Git。

## 16:00 后续点名故障修复

用户运行点名时百炼返回 HTTP 404 / `model_not_found`。配置中的 `qwen3-vl-flash-2026-01-22` 已列入[官方 2026-10-10 下线清单](https://platform.qianwenai.com/docs/changelog/model-deprecations/snapshot-models)。根据官方推荐迁移到固定版本 `qwen3.6-flash-2026-04-16`，默认配置、Gemini 配置和客户端缺省值均已更新。保留 `enable_thinking=false`、JSON 输出和 norm1000 坐标约定。

经用户明确授权，使用项目已有百炼凭据向原 DashScope 地址进行一次模型列表查询和一次生成图视觉调用，没有发送相机画面，也没有保存凭据。模型列表含 `qwen3.6-flash-2026-04-16`，不含已下线的 Flash 快照；生成白色矩形的识别结果 `found=true`、JSON 解析成功、bbox IoU 约 0.918、耗时约 0.732 秒、0 次重试。证据：`output/vlm_migration_verification.json`。此结果证明账号可调用新模型和现有客户端协议可用，不等于实拍白色鼠标识别与 3D 抓取匹配已验收。

HTTP 400/401/403/404/422 现在立即报告配置错误，不再提示成网络失败或重复提交；关闭 OpenAI SDK 的内部重试，以现有总超时/重试控制为准。日志按实际凭据值额外脱敏，兼容没有 `sk-` 前缀的密钥。

旁路 3D 窗口改为首次非空点云才建立包围盒、只在点云/抓取结果更新时重建几何，避免在每个相机帧中重复创建相同夹爪。已使用保存帧做包含 Open3D 窗口的离线 GUI 复测，未出现空包围盒警告和异常退出，见 `output_vlm_fix_gui.log`；没有占用用户正在运行的相机。终端输入改用支持 readline 的 input，支持方向键编辑。

SDK 队列满意味着消费端暂时跟不上输入，不能仅凭该日志诊断相机故障。此轮没有重启用户正在运行的程序，真实 DC1 同时打开 3D 时的丢帧频率需由重启后的运行进一步观察；可用 `--no-o3d` 对比。第三方 `torch.cross` 弃用提示保留，未改变其抓取几何算法。
