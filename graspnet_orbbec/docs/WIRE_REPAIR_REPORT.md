# 自动线材识别：闪退与耗时修复记录

更新：2026-10-28。验证使用本机 DC1（CC1N162037K）与项目现有模型；未发送外部视觉 API 请求，未产生机械臂运动指令。

## 已修复的问题

1. 用户日志中的退出是未处理的 `TimeoutError: DC1 取流超时：有效深度全部为零`。二维识别原本复用了 RGB-D 接口，因此彩色正常也可能因深度无效退出。识别窗口现使用 `capture/color_camera.py` 的独立彩色接口，后台连续消费相机帧，只保留最新一帧。彩色超时会显示“取流暂停”，等待恢复；自动识别只使用新帧。原 GraspNet 的 RGB-D 深度检查不变。
2. 真机复测又复现了 SIGFPE（退出码 136），Python 原生堆栈位于后台首次导入 Ultralytics 的阶段。该包的 `utils/__init__.py` 导入时会调用 `cv2.setNumThreads(0)` 并修改部分 OpenCV 函数。OpenCV 官方明确说明 [`setNumThreads` 不允许并发调用](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html)。因此将这部分首次导入移到主线程，在相机、窗口、推理线程启动前完成；模型本身仍由后台工作线程按需加载。首次后台导入现在会被明确拒绝，初始化失败也不会在窗口已运行后重新尝试导入。
3. 自动模式从每帧 1024 个密集采样点改为局部明暗线条上的稀疏提示，默认最多 64 点、每批 8 点，共用一次图像编码。提示预算是否耗尽会写入结果。
4. 放宽旧版对短线/宽度的几何筛选。闭环、出画线身可以保留；多分支中心线逐端计算候选，避免只留主路径两端。原图线条仍向外延伸的疑似掩码断口会被排除。端子模型独立运行，线身没分出来时仍保存端子检测。
5. 未识别出结果时，S 也可保存真实彩色原图和诊断信息。彩色文件回放无需 `depth.png`，不补造 RGB-D 数据。

SIGFPE 堆栈只定位到了导入阶段，没有捕获到导致信号的具体 C++ 指令；上述改动消除了已确认存在的并发全局设置问题，并通过后续普通启动真机复测。

## 最终速度对照

输入为 2026-10-26 用户调整好后采集的真实多束线材图：

`output/wire_repair/prepared_scene/color.png`

2026-10-28 的最终对照在真机窗口稳定性复测结束后单独运行；两种方法使用同一张图、同一模型和相同筛选配置，仅采样方法不同。

| 方法 | 提示数 | 单帧完整处理耗时 |
|---|---:|---:|
| 稀疏提示，首次加载模型 | 64 | 4.247 s |
| 稀疏提示，预热后第 1 次 | 64 | 0.819 s |
| 稀疏提示，预热后第 2 次 | 64 | 0.738 s |
| 密集搜索，模型已预热 | 1024 | 2.996 s |

运行库预初始化另耗时约 1.101 s。表格耗时包括识别函数内的端子模型、SAM2、几何处理和断口筛选，不含相机打开、取帧预热、窗口绘制和存盘。它不是连续视频帧率，也不能代表所有多束场景。

实际结果与报告：

- `output/wire_repair/controlled_benchmark/report.json`
- `output/wire_repair/controlled_benchmark/fast_warm_2/overlay.png`
- 同目录下的原图、二值线身/端子掩码、中心线和 `recognition.json`

该帧快速模式保留 9957 个前景像素、3 个几何端部候选，端子模型检出 2 处，另排除了 16 个疑似掩码断口。**仍有线身和端子漏检，未完成多束识别验收**。没有人工像素标注，不能计算真实 IoU、召回率或声称这些候选都是正确端子。用户确认没有配套标注数据，本轮先交付稳定性和速度修复，保留原图供后续标注。

之前的 `.35` 评分及局部裁剪实验产生了较多背景掩码；当前保留 0.5 评分阈值的稀疏方案。`output/wire_repair/20261026_190954/`、`color_live/` 以及早期名字含 `multi` 的部分探针实际没有拍到线材，不用于上表或多束识别效果结论；有效多束原图以 `prepared_scene/color.png` 为准。

## 稳定性及回归

- 修复前：普通真机启动在后台导入 YOLO 阶段复现 SIGFPE，记录为 `output/wire_repair/resume_plain_live.log`。调试器下当次未复现，未将其作为修复通过证据。
- 修复后：两次普通真机窗口运行，分别设定 25 秒和 60 秒后退出，均退出码 0、正常停止彩色流、释放设备。记录为 `fixed_plain_live.log` 和 `fixed_live_60s.log`。
- 当天真机画面是室内场景，并非之前准备的桌面线材，因此这两次只验证取流、运行和退出稳定性；多束效果来自上述已保存的真实线材帧。
- 117 项本地测试通过，记录为 `output/wire_repair/fixed_tests.log`。覆盖彩色流不要求深度、只读最新帧、暂时中断后恢复、窗口超时保留、旧结果隔离、主线程预初始化顺序、禁止后台首次导入、交叉线多端点和掩码断口筛选。

有限时长复测不能保证所有硬件断连或原生库故障都不会再发生。

## 使用

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
python scripts/run_live_gui.py --wire-auto --no-target
```

R 重新取帧识别，S 保存，Q 退出。若漏分，保持当前取景按 S；失败状态也会保存至 `output/wire_debug/`。原图可以这样复现：

```bash
python scripts/run_wire_segmentation.py --auto \
  --image output/wire_repair/prepared_scene/color.png \
  --out output/wire_reproduce_new
```

输出目录必须是新目录。自定义程序若在其他线程中使用 `AutomaticWireRecognizer`，需先在主线程调用 `prepare_terminal_runtime()`，再启动取流与推理线程。
