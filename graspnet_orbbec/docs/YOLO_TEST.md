# best.pt 分割效果检验

权重 `/home/ning/orbbec/best.pt` 已确认为 Ultralytics **YOLOv8x-seg**，保存版本 **8.3.232**，原始类别为 `{0: "1"}`。用户确认目标是**线材金属端子**，程序默认将单类别的显示名称设为“线材金属端子”，仅修改运行时标签，不改权重文件。`--label ""` 保留原名，`--label 名称` 可覆盖显示名称。程序直接加载现有权重，不重新训练。

## 直接启动

```bash
cd /home/ning/orbbec
bash graspnet_orbbec/scripts/run_yolo.sh
```

默认读取项目配置中的 Orbbec 相机（当前为 DaBai DC1，序列号 `CC1N162037K`），仅启用彩色流，不依赖深度、D2C 或激光设置。请先关闭其它占用相机的程序。自动选择 CUDA，可用时使用第一块 GPU，否则使用 CPU。画面几乎全黑时先检查镜头遮挡和光照。

窗口显示分割掩膜、检测框、类别、置信度及推理速度；顶部 `Confidence %` 滑块调节阈值。`s` 保存当前帧，`q` / Esc 退出。图片模式按任意键查看下一张；`s` 保存后继续下一张。速度包括单帧预处理、推理、后处理和结果转 CPU，不包括取流、绘图与存盘，也不能当作全流程帧率。第一帧包含初始化开销。

```bash
# 普通 UVC 摄像头
bash graspnet_orbbec/scripts/run_yolo.sh --source 0

# 图片、图片目录（不递归）或本地视频：替换成自己的路径
bash graspnet_orbbec/scripts/run_yolo.sh --source /path/to/image.jpg
bash graspnet_orbbec/scripts/run_yolo.sh --source /path/to/images --no-window
bash graspnet_orbbec/scripts/run_yolo.sh --source /path/to/video.mp4

# 不弹窗，拍摄并检测 30 帧；CPU 模式可显式指定 --device cpu
bash graspnet_orbbec/scripts/run_yolo.sh --no-window --max-frames 30

# 调节阈值和输入尺寸，或查看模型信息
bash graspnet_orbbec/scripts/run_yolo.sh --conf 0.4 --imgsz 640
bash graspnet_orbbec/scripts/run_yolo.sh --info
```

默认权重与配置使用脚本所在项目的绝对路径定位。`--source`、`--weights`、`--config`、`--output` 中用户传入的相对路径按当前终端目录解释。更换相机时可用 `--config /path/to/camera.yaml`；已有相机配置需与实际型号和序列号匹配。`--output` 必须是尚不存在的新目录，避免覆盖测试证据。

## 保存内容

默认输出 `graspnet_orbbec/output/yolo/<时间戳>/`：

- `000000_raw.jpg` / `000000_result.jpg`：原图和叠加图。首帧、末帧及按 `s` 的帧自动保存；图片目录保存每张图片。视频不自动另存完整视频。
- `predictions.jsonl`：每帧来源、原图尺寸、阈值、推理时间、类别、置信度、像素坐标检测框和分割轮廓。空检测也记录。
- `summary.json`：帧数、检出统计、设备、平均推理耗时及运行是否正常结束。取流失败会返回非零退出码并记录错误，不当作成功。

## 已完成的实测（2026-10-26）

- `output/yolo_cpu_smoke`：现有 DC1 照片完成 CPU 分割推理，检出 1 个类别“1”区域，置信度约 0.453。首帧约 684 ms。该照片没有标注，不代表准确率。
- `output/yolo_gpu_smoke`：同一照片完成 RTX 4060 GPU 推理，检出同一区域，首帧含初始化约 899 ms。
- `output/yolo_live_smoke`：最初复用 RGB-D 接口时，有效深度全零导致取流失败。YOLO 入口随后改为独立彩色取流，未修改原 RGB-D 采集接口。
- `output/yolo_color_live_smoke`：实际 DC1 彩色流连续 12 帧，GPU 预热后平均约 34.8 ms/帧。当前画面几乎全黑，12 帧均未检出目标，因此尚不能据此判断目标识别效果。
- `output/yolo_video_smoke`：使用两张既有相机照片组成的短视频验证读取与帧数上限，`--max-frames 3` 输出恰好 3 帧；`--conf 1` 的空结果正常记录。它用于验证输入路径，不是实拍视频评测。
- `output/yolo_directory_smoke`、`output/yolo_gui_smoke`：图片目录与带滑块窗口入口的运行记录。

模型效果仍需在明确目标、正常光照、不同距离/角度及有无目标的场景下检查。正式计算 precision / recall / mAP 需要独立、带标注的验证集；现有权重和未标注照片不足以生成可信准确率。

## 依赖复现

启动器复用 `.conda` 的 Python 3.10、PyTorch 2.7.1+cu126、NumPy 1.26.4、OpenCV 4.10.0。YOLO 相关依赖单独安装在 `.yolo-deps/`，不会升级抓取环境中的包：

```bash
cd /home/ning/orbbec/graspnet_orbbec
PYTHONNOUSERSITE=1 .conda/bin/python -m pip install --no-deps \
  --target .yolo-deps -r requirements-yolo.txt
```

API 参考：[Ultralytics 推理接口](https://docs.ultralytics.com/modes/predict/)、[分割结果](https://docs.ultralytics.com/tasks/segment/)、[验证模式](https://docs.ultralytics.com/modes/val/)。
