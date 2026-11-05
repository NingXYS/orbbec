# 静止场景多帧RGB采集（可选）

此功能仅采集、保存和验证独立原始帧，不运行YOLO/SAM2融合，不使能或运动机械臂。默认仍为原来的单帧采集；历史单帧目录无需转换。

## 使用

先完成[日常手册](REAL_GRASP_STARTUP.md)的CAN、官方链路及真实反馈检查。终端B工作目录为 `~/orbbec/graspnet_orbbec`。相机只能由一个程序占用；采集期间不要拖动机器人、相机、桌面或线材，勿触发其他控制程序运动。保持急停可触达。

确认整个场景保持静止后，可将日常单帧采图步骤替换为：

```bash
sequence_dir="output/wire_grasp/sequence_$(date +%Y%m%d_%H%M%S_%N)"
bash scripts/run_wire_grasp.sh capture \
  --frames 5 --scene-stationary-confirmed --out "$sequence_dir"
```

支持2–10帧，推荐5–10帧。每帧重新调用现有SDK原始RGB-D采图程序，包含原有预热，**不是固定帧率的连续视频burst**；总耗时随设备、预热和反馈收尾而变。每次调用必须获得新的独立RGB，不能复制单张旧图片补足帧数。

默认在完整RGB上检查静止。可选 `--scene-roi X1 Y1 X2 Y2`，使用完整原始RGB的整数坐标、右下角不包含；区域必须同时覆盖所有目标线材及固定背景，不能通过裁掉移动目标使验证通过。ROI只用于静止检查，保存的RGB不裁剪。

看到“多帧序列已保存”后，根目录须有 `COMPLETE`，manifest状态须为 `validated`。可不打开设备进行复核：

```bash
bash scripts/run_wire_grasp.sh verify-sequence --snapshot "$sequence_dir"
```

任一帧失败则整个序列失败，保留 `FAILED.json` 和现场文件，不生成根目录 `COMPLETE`；用新目录重采，不手改时间戳、配对或校验值。程序只回收自己启动的被动反馈子进程，不停止用户的官方链路，也不自动关闭CAN或下电。

## 验收范围与限制

- 机器人：复用已审计的只读CAN反馈工具，每段10秒、每8秒开始下一段。合并时只对同一内核接收时间戳去重；位姿冲突、时间倒退、反馈间隙超过20ms均拒绝。每张RGB及深度接收时刻须有前后反馈，且覆盖前后各0.5秒；**整个序列及帧间间隙**均须满足0.5mm/0.1°停稳筛选，不能用单帧局部窗口放宽。
- 配对：每帧独立采用与其RGB主机接收时间最近的原始法兰反馈样本，保存该帧 `T_base_flange`。不跨帧平均，不插值成曝光位姿，不声称硬件同步或定位精度。
- 可见静止：每张与第一张、上一张比较。至少8个可追踪特征、保留率≥80%、往返误差≤0.5px、最大位移≤0.75px、平均灰度变化≤3、灰度变化>20的像素比例≤2%。纹理不足、明显移动、遮挡或曝光变化会保守拒绝。
- 相机、线材：须由操作者明确确认静止，并通过上述可见图像检查。该检查**不能证明遮挡线材、微小局部移动或曝光间运动完全不存在**，也不能排除机器人稳定但相机支架缓慢挠动。验证结果只表示采集合同通过，不产生真机执行许可。
- 原始RGB像素哈希重复、主机接收时间不递增、设备时间重复均拒绝。完全相同的真实静态图像也可能被保守拒绝；SDK跨重启帧编号可重置，因此以独立采集会话、时间戳和帧ID共同标识。设备时钟若重置导致时间戳重复，本序列失败，不伪造新时间。
- 深度作为每次RGB-D采集的原始配对附件保留并校验文件类型、尺寸及反馈覆盖；本任务不校验深度几何精度，也不把深度时间误称为RGB曝光同步。

## 目录与manifest schema v1

```text
sequence/
  manifest.json
  robot.json                  # 整段反馈，保留所有实际样本
  verification.json
  COMPLETE
  feedback/segment_000.json   # 各段原始报告与同名log
  frames/frame_000/
    capture/{rgb.png,depth.png,camera.json,frames.json,COMPLETE}
    robot.json               # 根报告的硬链接，只有反馈复用；RGB不复用
    pair.json
    camera.log               # 真实采集时SDK日志
  frames/frame_001/...
```

根manifest字段：

| 字段 | 含义 |
|---|---|
| `schema_version`, `kind` | `1`、`stationary_rgb_sequence` |
| `status` | `capturing → pending → validated`；异常为 `failed` |
| `data_origin` | 原生采集为 `physical_camera`；注入测试驱动强制为 `simulated` |
| `requested_frames`, `frames` | 请求帧数和有序逐帧清单 |
| `camera_serial`, `calibration_sha256` | 设备及冻结标定合同 |
| `scene_stationary_confirmed`, `scene_roi_xyxy` | 操作者确认及静止检查范围 |
| `robot_report`, `robot_report_sha256`, `feedback_segments` | 全程反馈及原始分段路径/指纹 |
| `verification`, `verification_sha256` | 成功后的验证报告路径/指纹 |
| `robot_motion_commands_sent`, `pose_aggregation` | `0`、`none` |
| `hardware_sync_verified`, `exposure_pose_verified` | 均为 `false` |

每个 `frames[]` 包含：唯一 `frame_id`；相对根目录的 `snapshot` 和 `rgb`；字符串形式的微秒 `host_receive_timestamp_us`、`device_timestamp_us`；原始 `sdk_frame_index`；独立 `capture_session`；设备/标定；解码像素 `rgb_pixels_sha256`；7个必要文件的 `sha256`。原始RGB/深度各自的完整时间信息保留在该帧 `capture/frames.json` 中。

下一任务应先重新验证整个manifest，再按 `frames[]` 顺序读取每帧 `rgb`。不得把序列根目录当作单帧快照；需要旧流程分析某一帧时，指定对应 `frames/frame_XXX`。每帧保持原有 `capture/`、`robot.json`、`pair.json` 合同。选一帧分析不会产生跨帧检测融合。

序列的独立YOLO检测、端子关联和参考帧SAM2分析见[离线多帧稳定性验证](MULTIFRAME_STABILITY.md)。

新pair版本、16×16局部静止检查和旧数据执行资格变化见[修复说明](AUDIT_FIXES_20261104.md)。旧序列仍可分析，不通过手补字段升级为可执行数据。

测试注入接口 `_capture_frame` / `_feedback` 必须成对提供，产物始终标记 `simulated`；复核模拟序列必须显式加 `--allow-simulated`。这个选项不会将模拟数据变成真实数据。测试图片及报告禁止用于真机执行。普通SHA256用于完整性检查，不构成防伪签名。

## 离线测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda/bin/python -m pytest tests/test_stationary_sequence.py -q
```

测试覆盖2/5/10帧、逐帧实际样本配对、重复图片与时间戳、错误设备/深度/配对、文件篡改、ROI、全程运动及反馈缺口、重叠去重、失败清理和旧单帧命令分派。离线通过不代表物理设备实采已验收；首次实采仍需核对设备时间戳行为、光照及场景静止阈值。
