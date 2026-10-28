"""线材识别窗口：在原图快照上点选，后台分割，不产生机械臂指令。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import queue
import threading
import time

import cv2
import numpy as np

from capture.color_camera import FileColorCamera as FileCamera, open_color_camera as open_camera
from common.paths import OUTPUT_DIR
from segmentation.wire import Sam2WireSegmenter, draw_result, save_result
from segmentation.grasp_policy import apply_grasp_policy, confirm_boundary
from vlm.target_locator import put_cjk


@dataclass
class PromptState:
    source: np.ndarray | None = None
    box: list[int] | None = None
    points: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    revision: int = 0
    result: object = None
    message: str = "拖框选择线材；左键补前景点，右键补背景点"
    camera_error: str = ""

    def invalidate(self):
        self.revision += 1
        self.result = None

    def clear(self):
        self.source = None
        self.box = None
        self.points.clear()
        self.labels.clear()
        self.invalidate()
        self.message = "已清除，返回实时画面"

    def accept(self, revision, result, error):
        # 提示修改或恢复实时后，旧任务即使完成也不得覆盖新画面。
        if revision != self.revision:
            return False
        self.result = result
        self.message = error or f"掩码 {int(result.mask.sum())} 像素；端部候选 {len(result.candidates)} 个（几何推断）"
        return True


class SegmentationWorker(threading.Thread):
    def __init__(self, cfg, segmenter=None, auto_recognizer=None):
        super().__init__(name="sam2-wire", daemon=True)
        self.cfg = cfg
        self.segmenter = segmenter or Sam2WireSegmenter(cfg)
        self.auto_recognizer = auto_recognizer
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.stopping = threading.Event()

    def prepare_runtime(self):
        """在主线程完成会修改OpenCV全局状态的首次导入。"""
        if self.auto_recognizer is None:
            from segmentation.automatic import AutomaticWireRecognizer
            self.auto_recognizer = AutomaticWireRecognizer(self.cfg, self.segmenter)
        prepare = getattr(self.auto_recognizer, 'prepare_terminal_runtime', None)
        if prepare is not None:
            prepare()

    def submit(self, revision, image, box, points, labels, target=None, automatic=False):
        job = (revision, image.copy(), list(box) if box is not None else None,
               np.asarray(points).copy(), np.asarray(labels).copy(), target, automatic)
        # 只保留最后一组提示，避免逐次鼠标点击排成长队。
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait(job)

    def stop(self):
        self.stopping.set()
        self.join()

    def run(self):
        while not self.stopping.is_set():
            try:
                revision, image, box, points, labels, target, automatic = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if automatic:
                    if self.auto_recognizer is None:
                        from segmentation.automatic import AutomaticWireRecognizer
                        self.auto_recognizer = AutomaticWireRecognizer(self.cfg, self.segmenter)
                    result = self.auto_recognizer.recognize(image)
                    apply_grasp_policy(result, self.cfg)
                    self.results.put((revision, result, None))
                    continue
                if target:
                    from vlm.vlm_client import VLMClient
                    detection = VLMClient(self.cfg).detect(image, target)
                    if not detection.found:
                        raise ValueError("目标未找到，请直接拖框或在线身上点选")
                    box = detection.bbox_2d
                result = self.segmenter.segment(image, box=box, points=points, labels=labels,
                                                target=target or "线材")
                apply_grasp_policy(result, self.cfg)
                self.results.put((revision, result, None))
            except Exception as exc:
                self.results.put((revision, None, str(exc)))


def panel_point(x, y, width, height, header=64):
    """OpenCV图像坐标映射到左侧原始RGB，拒绝右面板和文字区域。"""
    return (x, y - header) if 0 <= x < width and header <= y < header + height else None


def render_canvas(live, state):
    h, w = live.shape[:2]
    canvas = np.full((h + 174, w * 2, 3), 30, dtype=np.uint8)
    left = (state.source if state.source is not None else live).copy()
    if state.box is not None:
        cv2.rectangle(left, tuple(state.box[:2]), tuple(state.box[2:]), (255, 220, 0), 1)
    for p, label in zip(state.points, state.labels):
        cv2.circle(left, tuple(map(int, p)), 4, (0, 255, 0) if label else (0, 0, 255), -1)
    canvas[64:64 + h, :w] = left
    if state.result is not None:
        canvas[64:64 + h, w:] = draw_result(state.result)
    title = "识别原图（已冻结）" if state.source is not None else ("最近画面（取流中断）" if state.camera_error else "实时原图")
    put_cjk(canvas, title, (10, 30), size=20)
    put_cjk(canvas, "SAM2 掩码 / 端点 / 夹持中心预览（原图快照）", (w + 10, 30), size=18)
    put_cjk(canvas, "R：自动识别 | Shift+左键：确认交界 | 拖框/点选纠正 | 空格/C：实时 | S：保存 | Q：退出", (10, h + 92), size=18)
    put_cjk(canvas, (state.camera_error or state.message)[:100], (10, h + 120), color=(0, 220, 255), size=18)
    warning = state.result.warnings[0] if state.result is not None and state.result.warnings else "A/B为几何候选；青色T为端子模型检测，仍需确认；仅输出二维像素位置"
    if state.result is not None and state.result.grasp_policy:
        offset = state.result.grasp_policy['offset_mm']
        warning = f'策略：交界后沿线身 {offset:g} mm；A/B为端点，粉色为交界，绿色为有尺度的夹持中心预览'
    put_cjk(canvas, warning[:100], (10, h + 151), color=(0, 190, 255), size=18)
    return canvas


def run(cfg, args, input_channel):
    cam = FileCamera(args.from_dir, loop=True) if args.from_dir else open_camera(cfg)
    state = PromptState()
    worker = SegmentationWorker(cfg)
    window = "SAM2 wire recognition"
    worker_started = False
    start = time.monotonic()
    latest = None
    drag_start = None
    preview_saved = False
    capture_pending = args.wire_auto
    capture_at = time.monotonic() + float(cfg.get_path('wire.auto.warmup_s', 0.8))

    def submit(target=None, automatic=False):
        state.invalidate()
        state.message = "自动搜索线材与端子中…" if automatic else "正在分割（首次加载模型需要稍等）…"
        worker.submit(state.revision, state.source, state.box, state.points, state.labels, target, automatic)

    def mouse(event, x, y, flags, param):
        nonlocal drag_start, capture_pending
        if latest is None:
            return
        if state.camera_error and state.source is None:
            return
        h, w = latest.shape[:2]
        point = panel_point(x, y, w, h)
        if point is not None and event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:
            capture_pending, drag_start = False, None
            if state.result is None:
                state.message = '请先分割线材，再Shift+左键确认金属/绝缘交界'
                return
            try:
                chosen = confirm_boundary(state.result, point, cfg)
                plan = next(p for p in state.result.grasp_plans if p['candidate_id'] == chosen)
                if plan['status'] == 'wire_instance_ambiguous':
                    state.message = f'{chosen}交界已记录；请先分开粘连线材，路径存在歧义'
                else:
                    state.message = f'{chosen}交界已确认；沿线身后退{plan["offset_mm"]:g}mm，待真实三维尺度'
            except ValueError as exc:
                state.message = str(exc)
            print(state.message, flush=True)
            return
        if event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            origin, drag_start = drag_start, None
            if point is None:
                return
            if np.linalg.norm(np.asarray(point) - origin) > 5:
                state.box = [min(origin[0], point[0]), min(origin[1], point[1]),
                             max(origin[0], point[0]), max(origin[1], point[1])]
                state.points.clear()
                state.labels.clear()
            else:
                state.points.append(point)
                state.labels.append(1)
            submit()
        elif point is not None and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            capture_pending = False
            if state.source is None:
                state.source = latest.copy()
            if event == cv2.EVENT_LBUTTONDOWN:
                drag_start = point
            else:
                state.points.append(point)
                state.labels.append(0)
                submit()

    try:
        print('正在初始化本地视觉运行库…', flush=True)
        worker.prepare_runtime()
        cam.open()
        worker.start()
        worker_started = True
        if not args.no_target:
            input_channel.start_stdin()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse)
        print("线材识别窗口已打开：R自动识别；Shift+左键确认金属/绝缘交界。空格/C恢复实时，S保存，Q退出。", flush=True)
        if args.wire_auto:
            state.message = "相机预热后自动取一帧识别，无需框选…"
            print(state.message, flush=True)
        if not args.no_target:
            print("输入目标词会调用已配置的VLM定位一次，再由本地SAM2分割；自动识别和鼠标点选全程本地。", flush=True)
        else:
            print("当前为本地识别模式，不使用外部API。", flush=True)
        first = True
        while True:
            fresh = False
            try:
                frame = cam.read()
                latest = frame.color_bgr
                fresh = True
                if state.camera_error:
                    print('彩色取流已恢复。', flush=True)
                state.camera_error = ''
            except TimeoutError as exc:
                if not state.camera_error:
                    print(f'取流暂停，窗口保留并等待恢复：{exc}', flush=True)
                state.camera_error = f'取流暂停：{exc}；等待新帧，Q可退出'
                if latest is None:
                    latest = np.zeros((int(cfg.get_path('camera.color_height', 480)),
                                       int(cfg.get_path('camera.color_width', 640)), 3), np.uint8)
            if fresh and first and (args.wire_box is not None or args.wire_point):
                state.source = latest.copy()
                state.box = args.wire_box
                state.points = args.wire_point
                state.labels = [1] * len(state.points)
                submit()
            if fresh:
                first = False
            if fresh and capture_pending and time.monotonic() >= capture_at:
                state.clear()
                state.source = latest.copy()
                capture_pending = False
                submit(automatic=True)
            while True:
                try:
                    revision, result, error = worker.results.get_nowait()
                except queue.Empty:
                    break
                if state.accept(revision, result, error):
                    print(state.message, flush=True)
                    if result is not None:
                        if result.automation:
                            state.message += f"；端子模型检出 {len(result.terminal_detections)} 个"
                            print(f'本次自动识别 {result.elapsed_s:.3f}s；{result.automation.get("sampling", {}).get("seed_count", "?")} 个自动提示', flush=True)
                        for item in result.candidates:
                            print(f"端点{item['id']}: {item['tip_xy']} 像素；夹持中心策略为交界后退{result.grasp_policy['offset_mm']:g}mm（待确认交界/三维尺度）", flush=True)
                        for item in result.terminal_detections:
                            print(f"端子模型{item['id']}: confidence={item['confidence']:.3f}（模型检测，未人工确认）", flush=True)
                        for warning in result.warnings:
                            print(f"提示：{warning}", flush=True)
            command = input_channel.poll() if not args.no_target else None
            if command:
                capture_pending = False
                drag_start = None
                if command.lower() in ("q", "quit", "exit"):
                    break
                if command.lower() in ("c", "clear"):
                    state.clear()
                else:
                    state.clear()
                    if fresh:
                        state.source = latest.copy()
                        submit(command)
                    else:
                        state.message = '取流暂停，目标词未提交，请恢复后重试'
            canvas = render_canvas(latest, state)
            cv2.imshow(window, canvas)
            if args.wire_preview and state.result is not None and not preview_saved:
                out = Path(args.wire_preview)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out), canvas):
                    raise OSError(f"预览保存失败：{out}")
                preview_saved = True
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("c"), ord(" ")):
                capture_pending = False
                drag_start = None
                state.clear()
            if key == ord('r'):
                drag_start = None
                state.clear()
                state.message = '正在重新取景…'
                capture_pending = True
                capture_at = time.monotonic() + 0.15
            if key == ord("s"):
                if state.result is None:
                    image = state.source if state.source is not None else (latest if fresh else None)
                    if image is None:
                        print('尚无可保存的真实帧', flush=True)
                    else:
                        out = OUTPUT_DIR / 'wire_debug' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        out.mkdir(parents=True)
                        cv2.imwrite(str(out / 'color.png'), image)
                        (out / 'diagnostic.json').write_text(json.dumps({'message': state.message, 'camera_error': state.camera_error}, ensure_ascii=False, indent=2))
                        print(f'已保存未识别的原图及诊断信息：{out}', flush=True)
                else:
                    out = OUTPUT_DIR / "wire" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    print(f"已保存：{save_result(state.result, out)}", flush=True)
            if args.wire_seconds > 0 and time.monotonic() - start >= args.wire_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        input_channel.stop_flag = True
        if worker_started:
            worker.stop()
        cam.close()
        cv2.destroyAllWindows()
    return 0
