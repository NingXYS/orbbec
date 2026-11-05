# 局部视觉路径门控兼容性报告

后续新增的真实来源门控也须通过，见[六项审计修复及兼容性](AUDIT_FIXES_20261104.md)。本文中的局部路径通过不单独授予执行许可。

日期：2026-11-04。仅离线纯数据检查，没有运行任何真实运动命令；没有改写历史recognition.json。

## 执行资格变化

新版不再根据 `path_ambiguous=false`、`boundary_confirmed=true`、工具已补偿或坐标有效直接授予执行许可。旧JSON缺少局部实例/米制路径证据及分阶段验证时，仍能被 `load_recognition()`读取，原有几何条件满足时仍可只规划；`execution_allowed=false`，原因是 `visual_path_validation_missing`。

原有 `--allow-nominal-urdf-tool`只影响名义工具阻断，不能补齐或绕过局部视觉证据。旧的“无readiness blockers即允许执行”和部分合成测试预期已更新；坐标值、14mm弧长及手眼/TCP公式未改变。

对本地output/wire与output/wire_grasp中可用的base_frame_preview_ready目标，使用修改前后的纯数据适配器分别复核：11个目标完成比较，8个从允许执行变为仅规划。此处“此前允许”仅指旧数据校验返回值，不表示历史真实运动成功或现场安全已验收。

| 历史文件（相对graspnet_orbbec） | 候选 | 新资格 |
|---|---|---|
| `output/wire/20261103_200723_147905/recognition.json` | `J2` | 允许 → 仅规划 |
| `output/wire/20261104_171506_553946/recognition.json` | `C` | 允许 → 仅规划 |
| `output/wire_grasp/real_20261103_164546_base-12_grasp135_01/recognition.json` | `A` | 允许 → 仅规划 |
| `output/wire_grasp/real_20261103_200055_base-12_grasp135_manual-j2-01/recognition.json` | `J2` | 允许 → 仅规划 |
| `output/wire_grasp/real_20261103_204658_fixed-plane-yolo-01/recognition.json` | `Y1` | 允许 → 仅规划 |
| `output/wire_grasp/real_20261103_204658_fixed-plane-yolo-b02-01/recognition.json` | `Y1` | 允许 → 仅规划 |
| `output/wire_grasp/real_20261103_213751_mainpath-manual-01/recognition.json` | `M10` | 允许 → 仅规划 |
| `output/wire_grasp/real_20261104_104006_roi-manual-01/recognition.json` | `M2` | 允许 → 仅规划 |

其余目标原先已有阻断；新增门控不会解除原有阻断。完整逐目标结果见 [compatibility.json](../output/visual_path_gate_validation_20261104_task5/compatibility.json)。该报告是本次本地文件快照，不自动代表未来新增文件。

## 使用新数据完成复核

1. 按[日常手册](REAL_GRASP_STARTUP.md)重新检查当前CAN、真实反馈、安装、平面、工具及现场安全。
2. 采集新的配对照片，重新测量线材高度。安装/桌面变化时先重建相应有效标定或平面，不直接复用旧坐标。
3. 使用本轮实例分离及路径门控代码重新识别。确认实例清晰、无交叉/断线歧义，Shift点击本次金属/绝缘交界，再S保存。
4. 核对保存结果的 `boundary_verification`、`visual_path_validation`、`verification_stages`及 `readiness_blockers`。模型预测不能代替人工确认；坐标成功不能代替实例验证。
5. 对该新结果执行主手册中的只规划，查看 `motion_spec.visual_path_review.passed=true`、`execution_allowed=true`及空执行阻断。失败时按明确原因处理，不复制旧证据、不手工填passed或清空blockers。
6. 即使纯数据资格通过，也仍需操作者对本次现场、碰撞、真实反馈、机械臂使能和夹爪状态做独立确认。本任务到只规划/离线检查为止，不发真实执行请求。

如果只排查算法，可对历史配对照片重新识别并保存到新目录，但必须标为历史分析；新生成验证字段不使旧场景自动变成当前真机目标。

## 测试与限制

- 全套离线测试：302项通过；新增局部门控33项。
- 覆盖有效路径、交叉、错误实例、断裂、宽度异常、方向突变、其他实例漏写、模型预测、旧JSON及缺失证据。
- 最终层在上游passed被误写为true或阻断列表漏写时，仍从原始局部掩码、完整实例游程和路径数据重新拒绝异常。
- 已使用实际官方适配器所用的系统 `/usr/bin/python3`复核合成有效目标；不依赖视觉Conda的skimage，未加载ROS节点。
- 合成成功样例仅证明数据合同与门控工作，不证明真实SAM识别精度、实际夹持、碰撞安全或绝对定位精度。

格式与拒绝码详见 [视觉路径门控说明](VISUAL_PATH_VALIDATION.md)。
