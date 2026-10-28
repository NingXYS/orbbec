# DC1 线材像素分割与端部附近识别

当前目标是识别可见线身和端子头附近的区域。`--wire` 在现有 OpenCV 入口中启用 SAM2 模式，不加载 GraspNet 抓取网络，不连接机械臂。原有 GraspNet 点名流程仍通过不带 `--wire` 的命令运行。

初始夹持策略现为**金属/绝缘交界后沿线身 10 mm**，闭合方向垂直于局部线材；使用 Shift+左键确认交界，详见 [WIRE_GRASP_POLICY.md](WIRE_GRASP_POLICY.md)。没有真实尺度时仅保存策略，不把像素位置当作毫米夹持中心。

## 启动与交互

### 自动取帧识别（无需手动框选）

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
python scripts/run_live_gui.py --wire-auto --no-target
```

启动后等待约 0.8 秒相机预热，自动截取当前帧并在本地处理。当前默认先从图像局部明暗线条提取最多 64 个稀疏提示，共用一次 SAM2 图像编码，再批量分割；不再默认做 32×32 的密集全图搜索。现有 `/home/ning/orbbec/best.pt` 独立分割端子，即使没有分出线身也保留端子结果。无需输入目标词、拖框、点选或 API 密钥。

二维识别使用独立彩色流和后台最新帧采集，不要求深度有效，不开启深度流，不修改激光或 D2C。彩色暂时超时时，窗口显示取流暂停并保留快照，等待新帧恢复；不会拿旧帧执行新一轮自动识别。原 GraspNet RGB-D 入口的深度有效性检查仍保留。

处理完成后固定显示本次原图和结果。按 **`R` 自动截取新一帧并重新识别**；空格/C 返回实时预览，之后按 R 才再次识别。不会每个视频帧重复搜索。鼠标仍可用于纠错，S 保存、Q 退出。`--wire-auto` 自动启用 `--wire`，不能同时指定初始 `--wire-box` / `--wire-point`。

青色 `T1` 等区域来自端子分割模型；绿色 A/B 区域来自线材几何端部推断，两者分开标记。只有掩码相交时，候选 JSON 才记录 `terminal_model_evidence`；`terminal_status=model_detected` 表示有模型支持，`geometry_only` 表示只有几何线索。模型支持不等于人工确认，因此全局 `terminal_semantics_verified` 和候选 `confirmed` 仍为 false。

端子模型在当前保存帧上检出上方端子（置信度约 0.863），**未检出下方端子**；两个线身端部都有几何候选。这一步没有解决端子模型的漏检。另试了两个端部的 96×96 局部裁剪，均未检出，因此没有把局部裁剪当成有效改进加入运行流程。

自动模式增加 `terminal_T1.png` 等原图尺寸的二值端子掩码，以及 JSON 的 `terminal_detections`、`automation` 字段。端子权重缺失/异常时保留线材结果并明确提示“端子模型不可用”。`wire.auto.terminal_enabled=false` 可关闭端子模型，不会下载替代权重。类别原名为 `1`，显示名称依照项目已有的端子模型说明配置；更换权重必须核对 `terminal_class_ids` 和类别语义。

自动模式的 `wire.auto` 参数仅限定当前线状候选：默认掩码中位宽度不超过图像短边的 2%，长度至少为短边的 6%，长度/宽度至少 12，SAM 评分至少 0.5。与旧版相比放宽了短线/宽度筛选；闭环或出画线身可以保留，但不会据此编造物理端点。它们不是类别识别条件，其他细长物体也可能入选。

`max_seeds=64` 限制单帧计算预算，`seed_spacing_px=32` 控制提示疏密，`prompt_batch=8` 控制批量解码大小。达到提示预算会在 JSON 和窗口警告中记录，不能承诺多束场景无漏检。`method=dense` 保留作对照诊断，常规使用 `line_prompts`。

多分支中心线逐端提取局部候选，不再仅保留主路径的两个端点；粘连处仍不保证线材实例对应。对于只有几何线索的候选，若原图线条沿端点外侧仍延伸，则排除疑似掩码断口，并记录 `automation.rejected_geometric_ends`。

复现此次无手工提示的离线检查：

```bash
python scripts/run_wire_segmentation.py --auto \
  --image output/live/20261026_181146/color.png --out output/wire_auto_reproduce

python scripts/run_live_gui.py --wire-auto --no-target \
  --from-dir output/live/20261026_181146 --wire-seconds 18
```

旧版密集搜索在该帧从 24 个自动掩码中保留两段可见细线（合计 5138 像素），端部附近候选约为 `(341,241)` 和 `(392,268)`，首次完整处理含模型加载约 7.25 秒。该记录保存在 `output/wire_auto_verified/`，属于旧实现，不能作为当前快速模式耗时。当前修复与复测见 [WIRE_REPAIR_REPORT.md](WIRE_REPAIR_REPORT.md)。

### 手动提示与纠错

先退出占用 DC1 的 GraspNet/Viewer 窗口，然后运行：

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
python scripts/run_live_gui.py --wire --no-target
```

- 在左侧原图**拖框圈住线材**，松开后分割；框只是提示，输出为真实像素掩码。
- 左键点击线身增加前景点；右键点击桌面/其他线材增加背景点。重新拖框会清空此前点提示。
- 第一次点选会冻结提示原图。右侧始终显示同一张原图的结果，防止移动相机后旧掩码错贴新画面。
- 空格或 `C` 清除提示、结果并恢复实时画面；开始下一轮需要重新拖框或点选。
- `S` 保存；识别失败或还没有结果时，会将真实原图和诊断信息保存至 `output/wire_debug/`，方便复现漏分。`Q` / ESC 退出。首次分割要加载模型，后续复用同一个模型。
- 不带 `--no-target` 时，可沿用终端输入目标词的交互。每次提交用现有 VLM 配置定位一次，之后由本地 SAM2 分割；该方式会向所配置的视觉服务发送当前原图。拖框/点选和离线入口全程本地，无需 API 密钥。

黄色为线材掩码边缘，紫色为中心线，绿色区域为几何端部区域。应用新夹持策略后，A/B 标识端点，粉色菱形标识人工确认的交界；绿色十字只用于有真实米制路径的夹持中心预览。中心线沿前景提取；接近原图或模型裁剪边缘的端点不作为真实端部候选。

## 结果及约束

保存至 `output/wire/<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `color.png` | 分割所用的原始 BGR 图像 |
| `mask.png` | 与原图同尺寸的 0/255 前景掩码 |
| `skeleton.png` | 线材中心线 |
| `end_region_A.png` 等 | 沿前景连通路径扩展的端部附近掩码，未验证金属端子类别 |
| `overlay.png` | 掩码、中心线、端部区域和候选位置 |
| `recognition.json` | 原图像素坐标、按连通段分开的中心线、候选、SAM 评分和警告 |

`wire.inset_px` 默认 12，表示沿中心线从端点内移 **12 像素**，端部区域沿前景扩展到 24 像素。它们不是毫米，不是夹爪尺寸，也不是已经校准的抓取偏置。JSON 明确保留 `terminal_semantics_verified=false`、`robot_pose_valid=false`、候选 `confirmed=false`。

存在多个连通段时，`centerline_xy` 为空，请读取 `centerlines_by_component`；程序不会把两个不连通线段直接拼成穿过背景的路径。

手动模式是**提示分割 + 几何端部候选**；自动模式增加全图掩码搜索和现有端子模型检测。SAM2 本身不会给出金属端子的类别。本次未训练/修改端子权重，尚未完成多根堆叠线材的实例对应、遮挡恢复、三维抓取验证。分支会提示歧义；闭合、过小、低分或无法确定端点时不编造端部位置。多个不连通前景不能仅凭图像确认属于同一根线材。

`wire.min_sam_score=0.5` 和 `max_mask_fraction=0.65` 用于排除低分/整块桌面掩码。SAM 评分是模型自身的质量估计，不是实际分割准确率；目前没有人工像素标注，不能报告真实 IoU。

## 已保存原图的复现

2026-10-26 对用户保存的原图（640×480）进行本地 CUDA 验证：

```bash
python scripts/run_wire_segmentation.py \
  --image output/live/20261026_181146/color.png \
  --box 0 85 415 355 \
  --out output/wire_reproduce
```

`--out` 必须是新目录。`--positive X Y` / `--negative X Y` 可以重复，坐标对应原图。

该帧线材左侧出画，两段可见前景共 4404 像素，SAM 评分 0.8047。两个可见端点为 `(345,250)`、`(404,264)`，内移候选为 `(340,240)`、`(393,268)`，排除了两个左侧截断端。实际掩码、叠加图、JSON 及测试记录在 `output/wire_verified/`。这是单帧检查，不代表堆叠场景或连续识别已经验收。

离线窗口复现（15 秒后关闭）：

```bash
python scripts/run_live_gui.py --wire --no-target \
  --from-dir output/live/20261026_181146 --wire-box 0 85 415 355 \
  --wire-seconds 15 --wire-preview output/wire_preview.png
```

`--wire-point X Y` 可重复指定初始前景点。`--wire-preview` 保存程序实际渲染的窗口画布，不是操作系统截图。提示框应按新场景重新指定，不能沿用上例坐标定位其他画面。

## 模型与依赖

使用 [Meta 官方 SAM2](https://github.com/facebookresearch/sam2) 的 SAM2.1 Hiera small。源码固定于 `2b90b9f5ceec907a1c18123530e92e794ad901a4`；检查点来自[官方分发地址](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)，SHA256 为 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`。

端子检测复用 [YOLO_TEST.md](YOLO_TEST.md) 中已有的本地 YOLOv8x-seg 权重与 `.yolo-deps` 依赖；不修改原 YOLO 测试入口或权重。

已在本机项目中安装。新环境运行 `bash scripts/install_sam2.sh`：源码放在 `third_party/sam2`，额外依赖放在 `third_party/sam2_runtime`，权重放在 `weights/sam2`。脚本不会升级已有 PyTorch/CUDA，不会编译附加 CUDA 扩展，也不会重置已有不同版本的 SAM2 源码。现有环境需具备 PyTorch、torchvision、OpenCV、NumPy、scikit-image 和 PyYAML；本机使用项目 `.conda`。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda/bin/python -m pytest tests/test_wire_segmentation.py -q
```
