# 初始夹持中心策略

2026-10-28 用户确认采用以下初始方法，后续以真实夹持测试调整：以金属端子与绝缘线身交界为基准，沿线身后退 10 mm；夹爪闭合方向垂直于局部线材方向。夹持位置、工具 TCP 偏置和插入预留空间分别处理。

## 当前窗口如何使用

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
python scripts/run_live_gui.py --wire-auto --no-target
```

1. 等待分割完成。A/B 现在表示线材端点，作为选择哪一端的标识。
2. 在左侧冻结原图的**金属/绝缘交界处按 Shift+左键**。程序按中心线距离将交界归属到对应端点；接近两个端部或路径粘连时会提示歧义。
3. 粉色菱形表示人工确认的交界。当前彩色取流没有真实三维尺度，会提示“待真实三维尺度”，不绘制冒充 10 mm 的夹持中心。
4. S 保存原图、掩码及策略/交界信息；R 取新快照后重新确认交界，防止相机/线材移动后沿用旧坐标。

绿色十字仅在有真实米制路径输入后表示夹持中心的位置预览。它不表示机械臂可执行位姿。原先的端点内移 12 像素仍保留为底层几何调试数据 `near_tip_xy`；正常窗口及离线 CLI 应用本策略后不再将该调试位置画成夹持中心。

当前默认：`configs/default.yaml` → `wire.grasp_policy.offset_mm: 10`。调整试验距离时修改此值，单位为毫米。没有物理尺度时，该参数只表示目标策略，不会转换为固定像素距离。

## 保存数据

`recognition.json` 增加：

- `grasp_policy`：交界基准、后退距离、闭合方向规则和工具补偿状态。
- `grasp_plans`：按端点记录人工确认、交界像素、当前可计算状态，以及可选的相机系夹持中心/闭合方向。
- 候选的 `inward_path_xy`：从该端点向线身内部定向的像素中心线；`path_ambiguous` 表示实例路径存在歧义。

所有结果仍为 `robot_pose_valid=false`、`tcp_compensation_applied=false`。边界确认只针对当前快照，不宣称端子类别已被模型验证。

离线 CLI 可直接记录已确认的交界：

```bash
python scripts/run_wire_segmentation.py --auto \
  --image /实际原图路径/color.png \
  --boundary A 交界X像素 交界Y像素 \
  --out /新的输出目录
```

坐标应来自这张原图，且候选 A 必须存在；可重复 `--boundary`。此命令依然不会产生毫米坐标或机械臂指令。

## 米制路径接口与后续标定

`segmentation/grasp_policy.py` 的 `plan_candidate()` 接受：

- `candidate`：包含 `inward_path_xy`，从端子侧向线身侧排列。
- `boundary_xy`：人工确认的金属/绝缘交界，投影到上述路径。
- `offset_mm`：从交界沿路径后退的物理距离，默认 10。
- `path_xyz_m`：与每个像素路径点逐点对应的 Nx3 **相机坐标系、米制**真实位置，所有数据必须有限且 Z 为正。
- `plane_normal`：相机系支撑面法向；有此数据才计算三维闭合方向。

算法从交界开始累计三维折线弧长至 0.010 m，插值得到夹持中心；不使用端点到目标的直线距离。闭合轴由支撑面法向与局部线材切向叉乘得到。图像中的绿色点是沿相邻像素路径的插值预览，不是独立重投影精度验证。

没有交界时为 `boundary_required`；没有真实路径时为 `metric_path_required`；路径有歧义时为 `wire_instance_ambiguous`；线身不足 10 mm 时为 `insufficient_wire_length`，不外推。只有位置可算而缺法向时为 `support_plane_required`。完整相机系预览为 `camera_frame_preview_ready`，仍不是可执行机器人位姿。

当前识别程序未自动采集/恢复这些三维路径。装臂后的尺度定位、相机到机械臂坐标变换、实际夹持中心到 TCP 的工具变换仍需独立验证。10 mm 目前是用户选定的试验初值，尚未用夹指尺寸验证插入余量。

## 验证

- 133 项本地测试通过：`output/grasp_policy_tests.log`。
- 新增测试覆盖三维弯线弧长、毫米/像素区分、交界归属、垂直闭合方向、缺尺度/法向/短路径/粘连路径、快照更新、保存及 Shift 点击不会重新启动分割。
- 真实保存帧的离线窗口运行 12 秒后正常退出：`output/grasp_policy_gui.log`、`output/grasp_policy_preview.png`。没有使用虚构尺度展示物理抓取点，没有连接机械臂。
