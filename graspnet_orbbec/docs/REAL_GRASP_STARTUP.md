# Piper X 视觉抓取真机启动手册

更新日期：2026-11-04，Asia/Taipei。

本文是实际操作的运行手册。每个新终端先进入复制后的 `orbbec` 根目录，再执行 `orbbec_root="$(pwd -P)"`；下文的 `$orbbec_root` 由实际位置得到，不依赖用户名。`orbbec` 与 `PIPER X` 应放在同一父目录。不要直接把尖括号占位符输入 Bash；本文命令使用终端变量和真实路径。

## 入口位置（2026-11-04归位）

- 官方真机启动：`bringup/piper_x/start_real_control.sh`。
- 采图与识别：`scripts/run_wire_grasp.sh`。
- 官方MoveIt抓取适配：`scripts/run_moveit_wire_grasp.sh`。
- 新启动的ROS日志和缓存：`bringup/piper_x/runtime/`。
- 官方工作空间和`PIPER X`标定工程保持原位。当前正式启动与抓取入口不再从“视觉”加载程序；旧启动路径仅作兼容转发。

[历史官方部署记录](PIPER_X_OFFICIAL_BRINGUP_HISTORY.md)和[历史交接记录](交接文档_官方MoveIt接视觉_20261103.md)仅用于追溯。迁移不代表重新验证真机执行或抓取效果。

## 1. 先分清两个终端

终端 A 只负责官方真机进程：

```bash
bash "$orbbec_root/bringup/piper_x/start_real_control.sh"
```

终端 A 启动后保持运行。除了最后按 `Ctrl+C`，其余检查、识别、规划和执行都在终端 B 完成。

当前官方入口会启动 RViz。如果不需要 RViz，只关闭 RViz 窗口，不要关闭终端 A，也不要结束 `move_group`、`ros2_control_node` 或 `agx_arm_ctrl_single_node`。

## 2. 使用当前已保存结果的快速路径

如果断电期间桌面、线束、机械臂底座和相机安装都没有移动，可以直接使用已修正的冻结结果：

```text
recognition: output/wire_grasp/real_20261103_164546_base-12_grasp135_01/recognition.json
candidate: A
base Z offset: -12 mm
grasp center depth: 135 mm
```

### 2.1 终端 A 启动官方链路

```bash
bash "$orbbec_root/bringup/piper_x/start_real_control.sh"
```

等待出现：

```text
You can start planning now!
```

不要再次执行启动命令。入口有单实例锁，重复启动会失败；若需要完整重启，先在原终端 A 按 `Ctrl+C`，确认全部退出后再启动。

### 2.2 终端 B 配置 ROS 环境

```bash
source /opt/ros/humble/setup.bash
source "$orbbec_root/agilex_official_ws/install_humble/setup.bash"

export ROS_DOMAIN_ID=93
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 2.3 检查控制器和真实反馈

```bash
ros2 control list_controllers -c /piper_x/controller_manager
```

预期三个控制器都是 `active`：

```text
joint_state_broadcaster
arm_controller
gripper_controller
```

检查真实反馈：

```bash
ros2 topic echo /piper_x/feedback/joint_states --once
ros2 topic echo /piper_x/feedback/tcp_pose --once
```

任何一条没有数据，都不要继续执行。

### 2.4 先做只规划

```bash
cd "$orbbec_root/graspnet_orbbec"

recognition="output/wire_grasp/real_20261103_164546_base-12_grasp135_01/recognition.json"
candidate_id="A"

scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id"
```

成功标志：

```text
规划完成：candidate=A stages=2
status: planned_only
physical_motion_performed: false
```

报告路径会打印为类似：

```text
output/wire_grasp/moveit_<时间戳>_A/report.json
```

报告中还必须看到：

```text
execution_allowed: true
execution_blockers: []
```

否则不要加 `--execute`。

真机执行报告会分别记录规划容差和实际到位验收容差。当前默认：

```text
规划位置容差：3 mm
规划姿态容差：0.05 rad
执行后位置验收：10 mm
执行后姿态验收：0.10 rad
```

执行后的验收容差用于判断控制器是否完成抓取动作，不改变视觉/手眼绝对精度，也不能据此声称毫米级精度。

### 2.5 手动使能并检查状态

确认工作区无人、急停可触达后再使能：

```bash
ros2 service call /piper_x/enable_agx_arm \
  std_srvs/srv/SetBool "{data: true}"
```

成功返回：

```text
success=True
message='Agx_arm enabled'
```

检查电机：

```bash
ros2 topic echo /piper_x/feedback/motor_status --once
```

六个关节必须同时满足：

```text
message: enabled
enabled: 'true'
```

检查夹爪：

```bash
ros2 topic echo /piper_x/feedback/gripper_status --once
```

必须满足：

```text
driver_enable_status: 'true'
```

所有故障标志都应为 `false`。

### 2.5.1 机械臂已使能但夹爪仍为disabled

当前官方ROS封装没有单独的夹爪enable服务。机械臂使能不会保证夹爪驱动已经使能；首次发送合法的官方夹爪宽度命令通常会唤醒夹爪。

先短时打开官方控制门控：

```bash
ros2 service call /piper_x/control_enable \
  std_srvs/srv/SetBool "{data: true}"
```

发送一次50 mm开口、1 N限力命令：

```bash
ros2 topic pub --once /piper_x/control/joint_states \
  sensor_msgs/msg/JointState \
  "{name: [gripper], position: [0.05], velocity: [], effort: [1.0]}"
```

等待一秒后检查：

```bash
ros2 topic echo /piper_x/feedback/gripper_status --once
```

预期：

```text
message: enabled
driver_enable_status: 'true'
```

确认后关闭门控：

```bash
ros2 service call /piper_x/control_enable \
  std_srvs/srv/SetBool "{data: false}"
```

该动作会让夹爪实际运动，执行前必须确认夹爪周围安全。若发送后仍为`disabled`，检查`agx_arm_ctrl_single`终端是否出现夹爪控制错误，不要重复高频发送。

### 2.6 执行真机抓取

```bash
scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id" \
  --execute
```

执行顺序：

1. 官方夹爪打开到 100 mm；
2. MoveIt 到预抓取位；
3. MoveIt 下降到夹持位；
4. 官方 `gripper_controller/follow_joint_trajectory` 闭合夹爪，使用驱动默认 `1.0 N`；
5. MoveIt 抬升回预抓取位。

## 3. 场景变化后的完整采图识别流程

如果机械臂、底座、相机、桌面或线束发生过移动，不能复用冻结坐标。

### 3.0 固定桌面平面模式

当前已保存：

```text
configs/support_plane_piper_x_02.json
```

来源快照：

```text
output/wire_grasp/support_plane_20261103_live_01
```

固定平面参数：

```text
frame_id: base_link
base_z_offset_m: -0.012
normal_base: [0.01809, 0.02626, 0.99949]
offset_base_m: -0.01215 m
棋盘重投影RMS: 0.2168 px
```

使用该模式时，新照片不需要棋盘入镜，但机械臂底座、桌面和相机安装必须没有移动。线束移动不影响固定桌面平面。

重新保存固定平面时，只需要完整棋盘，不需要线束入镜：

```bash
bash scripts/run_wire_grasp.sh save-fixed-plane \
  --snapshot "$snapshot_dir" \
  --base-z-offset-mm -12 \
  --out configs/support_plane_piper_x_03.json
```

保存时要求：

- 棋盘四边完整可见；
- 黑白方格清晰、无高光；
- 机械臂和相机在采集期间静止；
- 机械臂底座、桌面和相机安装保持固定。

固定桌面平面的使用命令：

```bash
bash scripts/run_wire_grasp.sh preview \
  --snapshot "$snapshot_dir" \
  --center-height-mm "$wire_center_height_mm" \
  --support-relation-confirmed \
  --same-installation-confirmed \
  --base-z-offset-mm -12 \
  --fixed-plane configs/support_plane_piper_x_02.json \
  --tool configs/wire_tool_piper_x_135mm.json
```

使用固定平面后，普通新采图不再需要棋盘。

### 3.1 采图

在终端 B：

```bash
cd "$orbbec_root/graspnet_orbbec"

snapshot_dir="output/wire_grasp/real_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_wire_grasp.sh capture --out "$snapshot_dir"
printf '\n本次快照目录：%s\n' "$snapshot_dir"
```

约四秒内保持机械臂、相机、桌面和线材静止。该步骤只读，不发送运动。

### 3.2 打开预览并确认交界

先输入实测的线材中心高度：

```bash
read -r -p "线材中心相对棋盘表面的高度(mm): " wire_center_height_mm
```

输入数字本身，例如：

```text
44
```

不要输入 `44mm`、`<44>` 或 `new_snapshot`。

启动预览：

```bash
bash scripts/run_wire_grasp.sh preview \
  --snapshot "$snapshot_dir" \
  --center-height-mm "$wire_center_height_mm" \
  --support-relation-confirmed \
  --same-installation-confirmed \
  --base-z-offset-mm -12 \
  --tool configs/wire_tool_piper_x_135mm.json
  --roi 30 120 520 280
```

默认不传 `--roi`，自动识别整张原图。线束位置经常变化时，应优先使用这种方式，不要复制上一张图专用的窄ROI。

如果整图误检过多，可增加覆盖整个工作区域的大ROI：

```bash
  --roi 0 0 640 350
```

不要把单个端子的窄ROI写进通用启动命令；它只适合在一张固定图片上排查特定端子。

窗口操作：

1. 按 `R`：重新自动识别；
2. `Shift + 左键`：点击端子侧金属/绝缘交界；
3. 确认出现粉色交界和绿色夹持中心；
4. 按一次 `S`；
5. 终端出现 `已保存：...`；
6. 再按 `Q` 退出。

`S` 只保存，不会退出；`Q` 才会关闭窗口。不要在没有看到“已保存”时退出。

GUI 的保存目录通常是：

```text
output/wire/<时间戳>/recognition.json
```

不是 `output/wire_grasp`。

### 3.3 找到最后一次保存结果

```bash
latest_recognition="$(ls -t output/wire/*/recognition.json | head -1)"
printf '最新识别：%s\n' "$latest_recognition"
```

列出已确认且可规划的候选：

```bash
python3 - "$latest_recognition" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
ready = [
    item["candidate_id"]
    for item in data["grasp_plans"]
    if item.get("status") == "base_frame_preview_ready"
]
print("可规划候选：", ready)
print("阻断：", data.get("grasp_policy", {}).get("readiness_blockers"))
PY
```

例如输出：

```text
可规划候选： ['A']
阻断： []
```

把看到的候选写入变量：

```bash
candidate_id="A"
recognition="$latest_recognition"
```

### 3.4 新场景先规划、再执行

只规划：

```bash
scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id"
```

确认报告：

```text
status = planned_only
execution_allowed = true
physical_motion_performed = false
```

然后按本文第 2.5 节手动使能、检查状态，再执行：

```bash
scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id" \
  --execute
```

## 4. 常见错误与处理

### `bash: new_snapshot: No such file or directory`

把 `<new_snapshot>` 原样粘贴了。尖括号是 Bash 重定向，不是路径。使用本文的 `snapshot_dir` 变量。

### `bash: 未预期的记号 "newline" 附近有语法错误`

同上，命令中包含 `<...>` 占位符。不要输入尖括号。

### `反馈未稳定`

失败目录没有`pair.json`，不能继续做手眼预览。

当前线束采图会先使用原有的全程稳定门槛。如果抖动只出现在图像采集完成之后，则回退检查`RGB接收时刻±0.5秒`内的真实曝光窗口：

- 反馈仍必须完整覆盖RGB和depth；
- 图像接收时刻前后必须有20ms内的反馈覆盖；
- 整个监听窗口不能存在超过20ms的反馈缺口；
- RGB接收窗口内位置变化必须≤0.5mm、姿态≤0.1°；
- 全程抖动幅度记录到`pair.json`供审计，但不会混入曝光时刻位姿。

2026-11-04实测中，RGB接收前后400帧完全一致，抖动只发生在图像采集完成后的约3.55秒。修改后重新采图成功，新快照为：

```text
output/wire_grasp/real_20261104_103204_retry01
```

如果RGB接收窗口本身也在抖动，仍然拒绝采图，需要检查伺服保持增益、负载、关节温度和碰撞/过流状态。

### `交界点离端部中心线太远`

在端子侧金属/绝缘交界处重新 `Shift + 左键`。当前版本允许交界点位于主中心线中部，不必恰好落在自动生成的端点上。成功后会生成类似`M10`的人工主路径候选。

如果界面仍拒绝，按`Q`退出，使用当前`640×480`工作区ROI直接指定交界：

```bash
bash scripts/run_wire_grasp.sh recognize \
  --snapshot output/wire_grasp/real_20261103_213751 \
  --center-height-mm 44 \
  --support-relation-confirmed \
  --same-installation-confirmed \
  --base-z-offset-mm -12 \
  --fixed-plane configs/support_plane_piper_x_02.json \
  --tool configs/wire_tool_piper_x_135mm.json \
  --roi 30 120 520 280 \
  --boundary-point 410 150 \
  --out output/wire_grasp/real_20261103_213751_manual_02
```

已验证`[410,150]`会吸附到主中心线`[411,142]`，生成`base_frame_preview_ready`候选。点击只需靠近真实端子交界；不要求先自动生成正确端点。

### `端子模型未检出端子；当前只有线材几何端部候选`

这不是完全识别失败，而是需要人工确认交界。使用同一张冻结快照，不必立刻重拍。

1. 查看终端列出的端点像素，例如：

   ```text
   端点A: [468, 84] 像素
   端点B: [454, 125] 像素
   ```

2. 在正确端点对应的金属/绝缘交界处执行 `Shift + 左键`。
3. 点击必须靠近黄色中心线；点到红色块、背景或离中心线超过约12像素时会报“交界点离端部中心线太远”。
4. 确认粉色交界与绿色夹持中心后，按 `S` 保存，再按 `Q`。

如果误检测到顶部圆孔等无关区域，但正确端点仍可选，可以忽略无关候选。只有正确端子无法选中时才重新采图。

### YOLO已经检出端子时（高级实验，默认关闭）

当前默认流程是“SAM2中心线 + 人工确认交界”。YOLO端子模型仍可作为辅助证据显示，但默认不会自动生成交界，也不会把YOLO端子点加入SAM2提示。本节仅用于以后显式开启实验模式时参考，不应复制到当前默认流程。

如需重新启用YOLO自动关联，可显式设置：

```yaml
wire.auto:
  terminal_auto_boundary: true
  terminal_use_seeds: true
```

启用YOLO自动关联时，成功终端和报告中会出现类似：

```text
端子模型T1: confidence=0.451
YOLO自动交界 Y1
terminal_boundary_suggestions: {'Y1': [460.0, 167.0]}
```

默认回退模式下仍可单独运行`run_yolo.sh`观察端子检测效果；正式识别坐标仍使用SAM2中心线和`Shift + 左键`确认交界。

`Y1` 是YOLO引导的自动交界候选，不依赖A/B等几何端部是否刚好落在端子上。其流程为：

1. YOLO定位金属端子；
2. 找端子掩码附近最近的SAM2主中心线点；
3. 从该点沿较长一侧线身生成抓取中心；
4. 后续仍要求当前棋盘平面和44 mm线材中心高度有效，才能得到毫米级抓取坐标。

如果只出现`T1`而没有`Y1`，说明端子与主中心线距离超过默认12像素。优先重新调整端子可见性，或使用`Shift + 左键`人工确认；也可配置：

```text
wire.auto.terminal_path_max_distance_px
```

不要仅为了“更灵敏”把这个距离无限增大，否则可能把邻近金属块或背景误关联到线束。

### 固定平面与YOLO联合验证结果（历史实验记录）

以下仅记录关闭自动关联前完成过的离线实验，不代表当前默认流程会自动点击或自动确认交界。

已用移动后的线束快照

```text
output/wire_grasp/real_20261103_204658
```

完成离线验证：

```text
YOLO T1 confidence: 0.451
YOLO框: [460,169]-[475,175]
自动交界 Y1: [460,167]
支撑平面来源: fixed_base_plane
抓取候选: base_frame_preview_ready
```

对应的完整识别结果：

```text
output/wire_grasp/real_20261103_204658_fixed-plane-yolo-b02-01/recognition.json
```

隔离MoveIt mock已对候选`Y1`完成只规划，`pregrasp`和`grasp`均成功，`execution_allowed=true`，且未发送真机动作。

### 刷新后仍无法识别正确端子

`R` 只是在同一张冻结原图上重新运行识别，不能恢复拍摄时已经丢失的信息。按下面顺序处理：

1. 按 `Q` 退出；
2. 保持同一快照，缩小右侧端子ROI后重开：

   ```bash
   bash scripts/run_wire_grasp.sh preview \
     --snapshot output/wire_grasp/real_20261103_200055 \
     --center-height-mm 44 \
     --support-relation-confirmed \
     --same-installation-confirmed \
     --base-z-offset-mm -12 \
     --tool configs/wire_tool_piper_x_135mm.json \
     --roi 0 0 640 350
   ```

3. 再按 `R`，并在列出的端点像素附近人工确认交界；
4. 同一快照经过一次ROI调整和一次人工点击仍失败，才重新采图。

`ROI` 是原始640×480图像上的矩形，格式为：

```text
x1 y1 x2 y2
```

- `(0,0)` 是左上角；
- `x` 向右增大，`y` 向下增大；
- `0 0 640 350` 表示覆盖整幅图像的上部工作区，排除下方夹爪和桌外区域；
- 它只是限制识别范围，不会放大图像，输出坐标仍是原图640×480坐标；
- ROI切到端子边缘会导致端子被当成裁剪断端，因此端子周围要留至少10–20像素；
- 线束位置变化时，先删除`--roi`；只有在单张图上误检过多时，才按当前画面重新设置窄ROI。

如果GUI中`Shift + 左键`总是在红块上选错，可直接在CLI指定正确交界并重新识别：

```bash
bash scripts/run_wire_grasp.sh recognize \
  --snapshot output/wire_grasp/real_20261103_200055 \
  --center-height-mm 44 \
  --support-relation-confirmed \
  --same-installation-confirmed \
  --base-z-offset-mm -12 \
  --tool configs/wire_tool_piper_x_135mm.json \
  --roi 260 20 640 230 \
  --boundary-point 416 88 \
  --out output/wire_grasp/real_20261103_200055_manual_j2
```

该命令已在本快照验证，生成候选`J2`并得到`base_frame_preview_ready`。如果输出目录已存在，需要换一个新的输出目录名，不能覆盖旧结果。

重新采图前检查：

- 棋盘四边完整，不遮住棋盘印刷面；
- 线束和正确金属端子完整进入画面；
- 端子不要贴着画面边缘，四周留出至少30像素；
- 端子尽量位于画面中部并占足够像素；
- 夹爪、手腕和线束阴影不能遮挡端子；
- 避免金属端子高光发白或完全融入背景；
- 只保留一组主要线束，避免多根线粘连；
- 采集时机械臂、相机、桌面和线材保持静止；
- 重新采图后必须按新的快照重新测量高度和确认交界。

重新采图：

```bash
snapshot_dir="output/wire_grasp/real_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_wire_grasp.sh capture --out "$snapshot_dir"
printf '新快照：%s\n' "$snapshot_dir"
```

随后使用本文第3.2节的预览命令，并把 `--snapshot` 设置为新生成的 `"$snapshot_dir"`。

### 保存后不知道用哪个文件

按 `S` 后终端会打印真实路径。也可以运行：

```bash
latest_recognition="$(ls -t output/wire/*/recognition.json | head -1)"
printf '%s\n' "$latest_recognition"
```

### 报告显示 `execution_allowed=false`

先看 `execution_blockers` 和 `readiness_blockers`，不要添加执行开关强行运行。

### 夹爪 JTC 报 `Trajectories with effort fields are currently not supported`

当前实现已修复：只发送位置轨迹，夹持力使用驱动的 `gripper_default_effort=1.0 N`。

### 重新上电后是否必须接入机械臂

- 只改代码、识别和离线规划：不需要机械臂。
- 真机执行、夹爪闭合、TCP到位和搬运验证：必须接入机械臂。

## 5. 安全停止和断电

正常退出时，在终端 A 对整个官方 launch 按一次 `Ctrl+C`，等待控制器全部退出。

需要软件急停时：

```bash
ros2 service call /piper_x/emergency_stop std_srvs/srv/Empty "{}"
```

实体急停优先。断电前确认机械臂制动保持、下方无人和无线材，再关闭机械臂电源；不要在驱动仍运行时拔 CAN/USB。
