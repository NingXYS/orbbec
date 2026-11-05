# Piper X 真机视觉抓取：日常操作

更新：2026-11-05，Asia/Taipei。适用于**采集新场景**，按顺序操作；任一步失败即停止，不跳到执行。

当前工程在 `~/orbbec`。迁移后请按实际位置调整 `cd`，保持 `orbbec` 与 `PIPER X` 同级。新终端的目录和变量相互独立；下面每个代码块只在指定终端运行，不需要 `orbbec_root`。

| 终端 | 工作目录 | 用途 |
|---|---|---|
| A | `~/orbbec` | 官方驱动、MoveIt、RViz；启动后保持运行 |
| B | `~/orbbec/graspnet_orbbec` | 反馈检查、新采图、识别、规划及可选执行 |

历史照片只用于[冻结结果复核](REAL_GRASP_HISTORY.md)，不是日常执行入口。异常处理见[故障手册](REAL_GRASP_TROUBLESHOOTING.md)，迁移依赖见[迁移说明](PORTABLE_DEPLOYMENT.md)。

来源门控修复后，旧快照/旧JSON保持可分析，但执行前需用新采集合同重采并重新确认，不要手补来源字段。见[修复与兼容性](AUDIT_FIXES_20261104.md)。

## 0. CAN检查：启动驱动之前

确认机械臂已上电、USB-CAN和机械臂CAN线接好，工作区清空、实体急停可触达。在任意终端查看：

```bash
ip -details -statistics link show can0
```

应包含 `UP`、`bitrate 1000000`，不能为 `STOPPED`、`BUS-OFF` 或仅监听模式。若是 `DOWN / STOPPED`，**确认旧官方launch已在终端A按 Ctrl+C 完整退出**后，手动配置：

```bash
sudo ip link set dev can0 type can bitrate 1000000 listen-only off
sudo ip link set dev can0 up
```

重新用上面的检查命令确认。接口缺失、配置失败或总线报错时停止。启动脚本只检查接口存在及已启用，不代替位率、总线健康和真机反馈检查；不自动配置CAN。

## 1. 终端A：启动官方链路

```bash
cd ~/orbbec
bash bringup/piper_x/start_real_control.sh
```

等待 `You can start planning now!`。入口启动官方驱动、MoveIt和RViz，默认 `auto_enable=false`、速度5%、官方控制门控及单实例锁。**RViz可见不代表真机已连接。** 不重复启动；RViz窗口退出也不等于后台已退出。

## 2. 终端B：加载环境、检查真实反馈

另开终端B，完整执行一次；若关闭该终端，新开后需重新配置：

```bash
cd ~/orbbec
source /opt/ros/humble/setup.bash
source agilex_official_ws/install_humble/setup.bash
export ROS_DOMAIN_ID=93
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd graspnet_orbbec
pwd
```

此后终端B保持在 `graspnet_orbbec`，`scripts/`、`configs/`、`output/` 都相对该目录。检查：

```bash
ros2 control list_controllers -c /piper_x/controller_manager
ros2 topic echo /piper_x/feedback/joint_states --once
ros2 topic echo /piper_x/feedback/tcp_pose --once
```

三个控制器 `joint_state_broadcaster`、`arm_controller`、`gripper_controller` 应为 `active`；两条真实反馈均须收到数据，关节完整、姿态合理。任何一条无数据、反馈异常或陈旧，停止，不反复使能。

## 3. 终端B：采集本次新快照

保持机械臂、相机、桌面和线材静止，采集约4秒的RGB/深度及被动法兰反馈；此步骤不发送运动：

```bash
snapshot_dir="output/wire_grasp/real_$(date +%Y%m%d_%H%M%S_%N)"
bash scripts/run_wire_grasp.sh capture --out "$snapshot_dir"
```

看到“配对快照已保存”，并确认目录内有 `pair.json` 和 `capture/COMPLETE` 才继续。失败时保留原目录，不覆盖或伪造配对文件。

需要保存同一静止场景的5–10张独立原始RGB时，参见[可选多帧采集与manifest](STATIONARY_RGB_SEQUENCE.md)。默认单帧不变；多帧仅采集和验收，不融合识别结果或自动运动。

## 4. 终端B：核实定位条件，SAM2识别并人工确认交界

当前默认启用这次固定摆位的ROI：原始640×480图中的 `[40,160,555,218]`，保留右端端子和线身，避开桌孔、夹爪及左固定块。无需另加参数；已打开的GUI需按Q退出后重开，R不会重新加载配置。相机视角、摆位或分辨率变化后须重选范围，见[固定视角ROI说明](FIXED_VIEW_ROI_037K.md)。

以下固定平面模式仅适用于：底座、桌面和相机安装与保存平面时一致；手眼绑定有效；夹指及夹持中心与工具配置一致。安装关系改变后停止复用，按[重新建立平面](REAL_GRASP_TROUBLESHOOTING.md#重新建立固定平面)处理；相机相对法兰改变还需重新核验手眼标定。

线束移动后可继续使用有效的固定桌面平面，但**必须重新核实线材中心相对棋盘印刷面的高度及局部平行关系**，不能沿用历史44mm。以下确认参数表示操作者已核实这些条件，不能仅为消除报错添加。

```bash
fixed_plane="configs/support_plane_piper_x_02.json"
tool_file="configs/wire_tool_piper_x_135mm.json"
base_z_offset_mm="$(python3 - "$fixed_plane" <<'PY'
import json, sys
print(float(json.load(open(sys.argv[1]))['base_z_offset_m']) * 1000)
PY
)"
read -r -p "本次实测线材中心相对棋盘印刷面的高度(mm，只填数字): " wire_center_height_mm

bash scripts/run_wire_grasp.sh preview \
  --snapshot "$snapshot_dir" \
  --center-height-mm "$wire_center_height_mm" \
  --support-relation-confirmed \
  --same-installation-confirmed \
  --base-z-offset-mm "$base_z_offset_mm" \
  --fixed-plane "$fixed_plane" \
  --tool "$tool_file"
```

先确认黄色中心线沿真实线身，没有把桌沿、夹爪或支架混入；必要时按 `R` 重跑同一张照片。用 **Shift+左键** 点击目标金属/绝缘交界，检查粉色交界、绿色夹持中心及基座坐标预览。默认沿线身后退14mm。端子模型仅为辅助证据，不代替人工确认；提示“待真实三维尺度”或路径歧义时不继续。

端子模型漏检时，只要线材实例与局部路径有效，仍可人工确认交界并继续定位、只规划。模型检出端子后也须完成相同检查；线材实例歧义、尺度缺失或来源校验失败时停止。

若Shift提示“实例归属存在歧义”，先在左侧**目标线身中段普通左键点一下**，等待重新分割；右键可补背景负点。检查新掩码只包含目标线身、没有粘连/交叉/断裂，按 **V复核线身**，再 **Shift+左键确认交界**。新补点会使复核失效，须重新按V；自动分割已清晰时不要求V。详见[手动纠正流程](MANUAL_WIRE_CORRECTION.md)。

## 5. 终端B：保存并选择本次候选

窗口中按 `S`，看到终端“已保存：...”后再按 `Q`。`S`只保存，`Q`才退出。GUI结果通常在 `output/wire/`；采图快照在 `output/wire_grasp/`。

输入**刚才S保存的目录**，不按“全局最新文件”自动挑选，以免误取历史结果：

```bash
read -r -p "粘贴本次“已保存”的目录（不要加引号）: " recognition_dir
recognition="${recognition_dir%/}/recognition.json"
python3 - "$recognition" "$snapshot_dir" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
policy = d.get('grasp_policy', {})
if Path(policy.get('capture_snapshot') or '').resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('停止：识别结果不属于本次快照')
ready = [x['candidate_id'] for x in d.get('grasp_plans', [])
         if x.get('status') == 'base_frame_preview_ready' and x.get('boundary_confirmed')]
print('可规划候选：', ready)
print('定位阻断：', policy.get('readiness_blockers'))
print('候选视觉阻断：', {x['candidate_id']: x.get('readiness_blockers', [])
                       for x in d.get('grasp_plans', []) if x['candidate_id'] in ready})
if not ready:
    raise SystemExit('停止：没有可规划目标')
PY
read -r -p "从上面列表选择本次目标ID: " candidate_id
```

检查命令报错或候选列表为空时，不输入ID、不继续。设置变量不会解除阻断。`base_frame_preview_ready`只表示可送规划，不是执行许可。

## 6. 终端B：只规划，检查报告与现场路径

```bash
plan_report="output/wire_grasp/plan_$(date +%Y%m%d_%H%M%S_%N)/report.json"
bash scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id" \
  --report "$plan_report"
```

这一步不加 `--execute`。程序打印报告路径；检查报告顶层 `status=planned_only`、`physical_motion_performed=false`，以及 **`motion_spec` 内**的 `execution_allowed`、`execution_blockers` 和 `warnings`。只有 `execution_allowed=true`、`execution_blockers=[]`，且警告已人工评估，才考虑真机执行；否则停在规划预览。

新结果还需 `motion_spec.visual_path_review.passed=true` 和 `motion_spec.source_review.passed=true`，分别表示局部路径及实际快照来源复验通过。`execution_source_missing` / `execution_source_invalid` 时按[修复说明](AUDIT_FIXES_20261104.md)重采并重新确认，不手补字段或删除阻断。旧JSON不能因坐标可用或 `path_ambiguous=false`获准执行；[复核步骤](VISUAL_PATH_COMPATIBILITY.md)说明如何重新生成证据。

只规划目前检查预抓取、抓取两段；抬升在执行阶段规划。人工查看目标、接近方向和运动空间，确认桌面、支架、相机及线材不会碰撞。适配器不自动把这些物体加入MoveIt场景，不能仅凭规划成功断言现场安全或视觉绝对精度达标。

## 7. 终端B：人工确认后，可选执行

只观察效果可跳到第8步。需要执行时，先确认本次照片、候选和规划有效；其后目标、线材、底座或安装关系未变，机械臂当前状态可信，工作区清空、实体急停可触达。场景改变须回到新采图，不重放旧目标。

人工使能（可能改变机器人保持状态）：

```bash
ros2 service call /piper_x/enable_agx_arm std_srvs/srv/SetBool "{data: true}"
ros2 topic echo /piper_x/feedback/arm_status --once
ros2 topic echo /piper_x/feedback/motor_status --once
ros2 topic echo /piper_x/feedback/gripper_status --once
```

使能服务须返回 `success=True`；六轴均 `enabled=true`，夹爪 `driver_enable_status=true`，诊断均为OK且无故障、无示教记录状态。反馈缺失/陈旧、夹爪未使能或任何阻断均不得继续；夹爪单独唤醒见[故障手册](REAL_GRASP_TROUBLESHOOTING.md#机械臂已使能但夹爪未使能)，该操作会真实运动。

人工确认上述条件后才运行：

```bash
bash scripts/run_moveit_wire_grasp.sh \
  --recognition "$recognition" \
  --candidate-id "$candidate_id" \
  --execute
```

真实动作顺序为：打开夹爪 → 预抓取 → 到夹持位 → 闭合 → 抬升。程序重新请求规划并检查真实反馈，不直接回放上一步报告；不会自动使能机械臂。失败后不盲目重试、不放宽容差或绕过阻断。控制器成功和`executed`报告仍需结合现场观察确认是否实际夹住、抬起线材。

失败报告中的 `physical_motion_performed=null` 表示已尝试动作但运动证据不足，不能当作“未动”。查看 `motion_commands_attempted`、`motion_evidence`及现场反馈，先确认真实状态再处理。

## 8. 安全退出

关闭预览用 `Q`；正常结束在**终端A对整个官方launch按 Ctrl+C**，等待驱动、MoveIt及控制器退出。只关闭RViz不会结束后台。

异常时实体急停优先；软件急停仅在驱动及服务仍可响应时使用，在已配置环境的终端B运行：

```bash
ros2 service call /piper_x/emergency_stop std_srvs/srv/Empty "{}"
```

退出进程不等于失能或物理急停。确认机械臂稳定支撑、制动保持、人员及线材安全后，按设备断电流程关闭电源；不要在驱动仍运行时拔CAN/USB。
