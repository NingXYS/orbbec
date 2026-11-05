# SAM2自动正点：搜索区域、排除区域与离线A/B

## 范围与兼容性

`segmentation/automatic.py` 的 `automatic_line_seeds()` 默认保留原有形态学对比度、连通段和轮流采样算法。新配置默认关闭，未配置的旧YAML也走兼容模式。`line_contrast()`默认行为不变，只有传入搜索掩码时才在有效像素内计算85百分位阈值，避免搜索范围外的强对比区域抬高阈值。

本功能只约束 `line_prompts` 的**自动正点**。不修改YOLO检测、14mm策略、手眼转换或控制逻辑；不把区域掩码施加到SAM2输入或返回结果。线身穿过排除区时仍可能被预测出来；如果整根线都没有可用种子，也可能漏检，不能保证完整分割。

`dense`模式不支持这些种子开关：开启过滤或调试时会明确报错，避免无声忽略配置；两个开关关闭时保留原dense流程。

## 配置

从 `graspnet_orbbec` 工作目录开始，复制完整默认配置后编辑；配置加载器不会把稀疏YAML自动合并到默认配置：

```bash
cp configs/default.yaml configs/wire_seed_ab.yaml
```

在副本的 `wire.auto` 下修改：

```yaml
seed_filter:
  enabled: true
  coordinate_space: original_rgb
  search_regions: null
  exclude_regions: []
seed_debug:
  enabled: true
  directory: output/wire_seeds
```

- `search_regions: null`：搜索当前推理ROI内全部像素；`[]`：没有有效搜索像素，输出零个种子，不回退全图。
- 非空区域是整数半开矩形列表，每个矩形为 `[x1, y1, x2, y2]`，多个搜索矩形取并集，再扣除排除矩形。排除区域优先。
- `coordinate_space: original_rgb`：区域相对完整原图；`roi`：区域相对推理ROI左上角。该配置同时作用于搜索与排除矩形。
- 区域与ROI取交集，越界部分裁掉；零面积或完全在ROI之外的区域不贡献像素。反向矩形、非整数、布尔坐标或未知坐标系报错。
- 没有内置棋盘格、固定块、夹具或端子的截图坐标。操作者应按当前照片设置区域，场景变化后重核范围。
- `seed_filter.enabled: false`完全忽略区域设置，恢复旧采样；`seed_debug.enabled`独立控制调试文件，可以在旧模式也打开。
- 若既有配置打开了端子提示种子，新区域过滤会在交给SAM2前同样约束这些额外正点，不改变端子模型本身。日常默认的 `terminal_use_seeds: false`未改变。

例如，仅为说明坐标换算：推理ROI左上角为 `(100,200)`，局部正点 `(20,30)` 对应原图 `(120,230)`；这不是当前工位的配置值。ROI仍是已有推理裁剪范围，种子搜索区域不会成为新的推理裁剪。

## 离线比较：不运行模型或硬件

先把 `image_path` 设为待检查的真实RGB文件；`read`只读入路径，不采图。使用新输出目录，不覆盖旧比较：

```bash
read -r -p "待比较的RGB图片路径: " image_path
PYTHONDONTWRITEBYTECODE=1 .conda/bin/python scripts/compare_wire_seeds.py \
  --image "$image_path" \
  --config configs/wire_seed_ab.yaml \
  --out "output/wire_seed_ab_$(date +%Y%m%d_%H%M%S_%N)"
```

如需与现有ROI推理一致，额外传入 `--roi`及当前原图上的四个整数。工具不从YAML读取 `wire.auto.roi_xyxy`：未传 `--roi`时按完整RGB比较，避免隐含裁剪。A组强制关闭过滤，B组强制开启过滤，使用同一张照片、ROI和采样预算；B组区域取自指定YAML。工具不修改配置，不加载SAM2/YOLO，不连接相机或机械臂。

每组输出：

- `before.png`、`after.png`：完整RGB尺寸，ROI内标注种子；绿点为实际正点，暗色为搜索范围外，红色为排除区。叠色仅供调试，不修改输入。
- `before.json`、`after.json`：ROI局部及完整RGB种子坐标、实际裁剪区域、过滤统计。
- `summary.json`：前后种子数量、有效搜索像素数、过滤的对比像素数，以及未运行模型/未发送机器人命令标记。

数量不一定下降：过滤强对比背景后，预算可能重新分给其他连通段。应比较位置是否落在正确线身，而不能仅以种子变少判定更好。没有人工线材标注的数据不能宣称提示点精度或抓取成功率。

## 现有识别流程中的调试输出

用配置副本运行现有CLI：`run_wire_grasp.sh`要求把全局 `--config configs/wire_seed_ab.yaml` 放在 `preview`或`recognize`子命令之前；其余快照、平面、高度和工具参数沿用[日常手册](REAL_GRASP_STARTUP.md)。本功能不新增启动或控制脚本。

开启 `seed_debug` 后，每次推理在配置目录下建立独占 `seeds_<唯一ID>/`：

- `seeds_roi.png`：推理ROI大小的调试图。
- `seeds.json`：`seed_points_roi_xy`、`seed_points_rgb_xy`、`seed_count`、`seed_budget_reached`及区域统计。

GUI按S保存的 `recognition.json` 中，`automation.sampling.seed_points_xy`仍为原图坐标；新增的 `seed_filter`字段同时保留两套坐标和统计，`seed_debug_directory`指向调试目录。空搜索区域同样输出零点诊断，不加载SAM2。

统计中 `contrast_pixels_before_filter/after_filter/filtered`使用本组阈值计算，因此A、B阈值不同的情况下不能把两组该计数简单当作同一批候选的增减。额外正点被区域或边界过滤的数量单列在 `extra_points_filtered`；区域过滤先于预算截断。

## API与验证边界

- `automatic_line_seeds(image, cfg, extra_points=(), *, roi_xyxy=None, diagnostics=None)`：返回原兼容的 `(points, capped)`；`points`始终为ROI局部xy，`roi_xyxy`为该图像对应的完整RGB整数矩形，省略即原点为零。可选字典 `diagnostics`接收可序列化诊断信息。
- `line_contrast(image, cfg, search_mask=None)`：可选布尔搜索掩码只用于阈值统计；不改变原图或返回的完整对比图。
- `seed_search_mask()`负责区域坐标投影；`draw_seed_debug()`只生成可视化副本。

区域过滤不具备线材语义，允许区内的棋盘、反光和夹具仍可能生成错误正点。ROI边界附近的形态学对比仍会受相邻像素影响。调试目录持续增长，需按实验结束后归档。离线种子测试和假预测器集成测试不代替真实SAM2分割、人工标注准确率或真机抓取验证。

## 逐候选背景负点实验

自动SAM2的正负点组合、可信背景配置、批次格式、逐候选回退和调试字段见 [正负点实验说明](WIRE_NEGATIVE_PROMPTS.md)。正式配置默认关闭，实验配置独立开启；本页的纯种子A/B工具不运行SAM2，亦不评估负点效果。
