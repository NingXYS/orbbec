# Piper X 真机视觉抓取：故障排查

更新：2026-11-04。日常命令只以[主手册](REAL_GRASP_STARTUP.md)为准。本页按症状查找；以下操作默认在主手册配置好的终端B、`~/orbbec/graspnet_orbbec` 下进行。不要清空故障、扩大限位或绕过执行阻断来“跑通”。

## 路径、变量与重复启动

- 报 `/bringup/...不存在`：旧命令使用了未设置的 `orbbec_root`。改用主手册第1节完整的目录切换和启动命令。
- 报 `new_snapshot: No such file or directory` 或 `newline`语法错误：把尖括号占位符当命令输入了。使用本次采图建立的 `snapshot_dir`；路径含空格时保留引号。
- 报“会话已在运行”或“遗留官方驱动”：先找到原终端A，Ctrl+C并等整套launch退出，再启动。不要仅关闭RViz后再开第二套。
- 终端B新开后没有ROS命令或看不到服务：重做主手册第2节环境配置，确认domain为93；终端A中的变量不会传给B。

## CAN未连接或反馈为空

`Agx_arm is not connected`、`waiting for service`、两个反馈话题一直无数据时：

1. `Ctrl+C`结束等待中的命令，不反复请求使能。
2. 按主手册第0节只读查看CAN，确认接口存在、UP、1Mbps、不是listen-only/STOPPED/BUS-OFF。
3. 需要改CAN配置时先完整停止官方launch。接口正确仍无反馈，检查机械臂电源、USB-CAN、CAN接线与终端电阻，并查看终端A的初始连接日志。
4. 按主手册恢复单实例，再取得完整、新鲜、合理的反馈。RViz显示模型、控制器active或服务存在，均不单独证明真机已连接。

## 采图失败或“反馈未稳定”

没有 `pair.json`、`capture/COMPLETE` 或目录内出现 `FAILED.json`，都不能继续把该目录用于抓取。保留 `camera.log`、`robot.log`、失败记录供排查，使用新的独占目录重采，不覆盖原文件。

程序先检查全程停稳；只有特定“反馈未稳定”失败才回退检查RGB**主机接收时刻**±0.5秒窗口：

- RGB和depth接收时刻均需被完整位姿前后覆盖，前后间隔各不超过20ms。
- 监听窗口不能有超过20ms的完整反馈缺口，覆盖图像接收前后各0.5秒。
- RGB接收附近位置变化不超过0.5mm、姿态变化不超过0.1°；全程抖动另记。

这是主机接收窗口检查，不是硬件曝光同步或精度验收。若图像附近反馈确实抖动，先处理保持、负载、温度或故障，不放宽采图条件。

## 缺定位平面或“待真实三维尺度”

`current_board_plane_unavailable` 表示当前照片没有取得棋盘定位平面；确认二维交界、输入候选ID都不会生成毫米尺度。

- 固定桌面模式：检查主手册第4节是否带了 `--fixed-plane "$fixed_plane"`，文件、手眼指纹及Z修正是否匹配。
- 底座、桌面或相机安装改变后，不得直接复用原固定平面；先重新核验标定及[重建平面](#重新建立固定平面)。夹指改变还要核实工具文件。
- 改用“当前照片棋盘”模式：需完整、清晰棋盘与线材同时可见，在主手册预览命令中省略 `--fixed-plane`这一行；仍需实测高度、有效工具、安装确认及本次配对快照。
- `measured_wire_center_height_and_support_relation_required`：重新测量中心高度并核实局部平行关系。当前实现只接受相对棋盘±50mm范围，不可把历史44mm当默认。

只漏传参数且场景未变，可用原 `snapshot_dir` 重开预览，不必重采。定位方法或安装改变则重新取得相应有效数据。

## 重新建立固定平面

此操作只保存几何资料，不使能或运动。先核实相机相对法兰的安装和手眼绑定；改变相机安装不能只重建桌面平面了事。

1. 在终端B按主手册第3节采集**新的完整棋盘快照**：四边完整可见、无高光或遮挡，采集时机械臂与相机静止；不要求线束同框。
2. 核实底座Z修正的实际依据，不能默认旧实验值。以下使用刚采集的 `snapshot_dir`，保存到新的文件名：

```bash
read -r -p "核实后的base Z修正(mm，带正负号): " new_base_z_offset_mm
new_plane="configs/support_plane_$(date +%Y%m%d_%H%M%S_%N).json"
bash scripts/run_wire_grasp.sh save-fixed-plane \
  --snapshot "$snapshot_dir" \
  --base-z-offset-mm "$new_base_z_offset_mm" \
  --out "$new_plane"
printf '新固定平面：%s\n' "$new_plane"
```

3. 核对保存结果及现场几何。日常流程中将 `fixed_plane` 设为此新文件；不能只改文件名或SHA去迎合校验。随后对线束采集新快照，并重新测量高度、确认交界。

## 掩码混乱或交界无法确认

端子模型未检出并不等于线身识别失败：只要真实线身中心线有效，可以人工确认金属/绝缘交界。反过来，端子检测分数高也不证明它是正确目标。

- 掩码覆盖桌面、夹爪、支架或大量噪声时，不随意从几何候选中选一个。先按Q退出，在**主手册第4节的预览命令**末尾加 `--roi x1 y1 x2 y2`，按当前照片选范围。此处四个字母是参数说明，不可原样粘贴。
- ROI采用原图像素，左上角为原点，x向右、y向下；裁剪不改变输出坐标。包住端子及足够长的线身，端子四周至少留10–20像素，避免把裁剪断端当物理线头。
- 不将某张照片的窄ROI写成通用默认。线束位置改变后重新选取；不扩大关联距离去强行匹配背景。
- 手眼预览支持普通左键补线身正点、右键补背景负点。对单根直线却报告大量歧义的情况，先在线身中段普通点一下，等待结果，再检查目标掩码并按V复核，随后Shift点交界。窄框可能丢失SAM2上下文，优先试单点；详见[手动纠正与复核](MANUAL_WIRE_CORRECTION.md)。
- “交界点离中心线太远”：Shift+左键点在真实交界附近的黄色主中心线上，不能点红块或背景。人工交界可位于中心线中部，不必落在自动端点。
- `wire_instance_ambiguous`：继续补点或改善场景，不能把不唯一的路径强行转为三维目标。V不会清除分叉/交叉/断线标记，Shift不能代替重新分割。
- `visual_path_manual_mask_review_required`：手动纠正后尚未按V复核，或复核后图像/掩码已改变。核对当前目标后重新复核；不能手补JSON或清空阻断。
- R只重跑同一张冻结照片。照片本身遮挡、高光或目标出画时，整理现场后重采，再测高度、重新确认。

高级CLI支持 `recognize --boundary-point X Y --out 新目录`，其余定位参数与主预览流程一致。坐标必须来自本次照片；`preview`不接受用 `--out`指定保存结果。

## 保存结果、候选与规划阻断

- S保存、Q退出；没有看到“已保存”就不能认为结果已写入。GUI保存目录通常在 `output/wire/`，不是原采图目录。
- 直接粘贴本次保存目录，检查其 `grasp_policy.capture_snapshot` 对应本次 `snapshot_dir`；不要通过全局最新时间自动选择其他窗口或旧任务的结果。
- 候选必须为 `base_frame_preview_ready` 且 `boundary_confirmed=true`。空列表或 `METRIC_SCALE_MISSING`时停止；手动设置candidate_id不会改变状态。
- 报告顶层查看 `status`，许可和阻断在 `motion_spec.execution_allowed`、`motion_spec.execution_blockers`；不存在这些字段不等于允许执行。
- `nominal_finger_tip_reference_not_verified_grasp_center`默认只允许规划。不要把 `--allow-nominal-urdf-tool`当通用排错开关；它代表额外接受名义工具风险，不提供实测TCP证据，日常流程不使用。
- TF末端偏置、诊断陈旧、无效工具、当前姿态或反馈错误需处理原因，不伪造目标、不放宽门槛。

只规划检查预抓取和抓取两段；抬升在真机执行时另行规划。规划目标默认容差为3mm/0.05rad；执行后到位默认验收为10mm/0.10rad。这些是程序阈值，不是独立视觉、手眼或TCP绝对精度。

## 机械臂已使能但夹爪未使能

先完成主手册的反馈、规划及现场人工确认。当前官方封装没有独立夹爪enable服务；机械臂使能不保证夹爪已使能。下面是**会真实运动的专项处理**，不是每次启动必做项。夹爪周围必须安全，驱动保持单实例。

短时打开官方门控，发送一次50mm开口、1N命令：

```bash
ros2 service call /piper_x/control_enable std_srvs/srv/SetBool "{data: true}"
ros2 topic pub --once /piper_x/control/joint_states \
  sensor_msgs/msg/JointState \
  "{name: [gripper], position: [0.05], velocity: [], effort: [1.0]}"
```

等待一秒，再检查并关闭门控（不论唤醒是否成功，都应尝试关闭）：

```bash
ros2 topic echo /piper_x/feedback/gripper_status --once
ros2 service call /piper_x/control_enable std_srvs/srv/SetBool "{data: false}"
```

须看到 `driver_enable_status=true`、诊断OK且无故障。仍disabled或命令失败时停止、看驱动日志，不重复高频发送；门控关闭失败也不得继续。成功后重新完成主手册的执行前检查。

## JTC报错、到位失败与停止

- `Trajectories with effort fields are currently not supported`：现有适配发送位置轨迹，夹持力使用官方驱动默认值；核对是否运行了旧版本，不另写控制器解决。
- `真实TCP未到目标`、夹爪开闭未达标：保留报告和现场状态，检查目标、工具、真实反馈及实际夹持，不仅凭控制器SUCCEEDED认定成功，不反复执行或扩大容差。
- 执行失败报告中的 `physical_motion_performed`不能单独证明没有动作：代码可能在标记更新前已经开夹爪或移动。结合完整动作记录、真实反馈和现场观察判断。
- 关闭RViz、退出ROS进程、软件急停与物理断电不是同一件事。异常停止和正常退出按[主手册第8节](REAL_GRASP_STARTUP.md#8-安全退出)，实体急停优先。

仅离线识别已配对照片可不连接真机；当前只规划入口仍需ROS规划会话与关节反馈。仿真反馈不能当真机当前状态，离线通过不能代替实机验证。
