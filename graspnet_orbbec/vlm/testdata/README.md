# VLM 离线评测图（冻结副本）

`--eval` **只读这个目录**，不会扫描 `output/` 里的历史采集。

| 文件 | 内容 | 原始来源（已拷贝，评测不再读原路径） |
|---|---|---|
| `01_floor.png` | 地板：蓝色湿巾包 + 黄色小方块 | Astra 旧采集 `output/20260911_161932` |
| `02_mouse.png` | 近景：白色鼠标 + 白色杯子 | Astra 旧采集 |
| `03_laptop.png` | 近景：笔记本电脑 + 白色鼠标 | Astra 旧采集 |
| `04_desk.png` | 侧视桌面：机械键盘 + 黑色杯子 | Gemini 336L `output/gemini_grasp` |
| `05_desk2.png` | 侧视桌面：显示器 + 白色鼠标 | Gemini 336L `output/gemini_grasp` |

换图：覆盖这 5 个文件，并改 `configs/vlm_eval.yaml` 里的 `bbox_2d`。不要把新图丢进 `output/` 指望评测自动捡到。

结果默认只写 `output/vlm/latest/`，每次评测覆盖，不再追加时间戳目录。
