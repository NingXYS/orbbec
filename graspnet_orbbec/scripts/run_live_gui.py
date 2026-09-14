#!/usr/bin/env python3
"""连续规划 + 点名闭环：终端输入目标词 → VLM → 3D 定位 → 最近抓取高亮。

VLM 只在提交目标词时调用一次并缓存，显示主循环不打 API。

窗口：
  q / ESC  退出
  s        保存当前帧与抓取到 output/
  空格      暂停 / 继续推理
  3        弹出可旋转 3D 窗口
终端：
  输入目标词后回车（例如 白色鼠标）
  c        清除高亮，回到等待输入
  q        退出

示例：
  python scripts/run_live_gui.py
  python scripts/run_live_gui.py --from-dir third_party/graspnet-baseline/doc/example_data
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from capture.camera import FileCamera, open_camera, save_rgbd_dir
from capture.run_capture import overlay_rgbd
from common.config import load_config
from common.paths import GRASPNET_ROOT, OUTPUT_DIR, timestamp_dir
from infer.filters import grasp_to_dicts
from infer.graspnet_infer import GraspNetInfer
from infer.match_grasp import NoGraspMatchError, match_nearest_grasp
from infer.target_query import QueryError, QueryResult, QueryWorker
from vlm.target_locator import draw_locate_overlay, put_cjk
from viz.visualize import overlay_grasps_on_rgb, save_triplet

LOGGER = logging.getLogger("graspnet_orbbec.live")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraspNet 连续规划 GUI")
    p.add_argument("--config", default=None)
    p.add_argument("--from-dir", default=None, help="离线目录循环播放")
    p.add_argument("--official-demo", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--top-k", type=int, default=30, help="3D 窗口夹爪数量（官方 demo 常用 50）")
    p.add_argument("--rgb-k", type=int, default=8, help="RGB 上只画前 k 个，避免框叠成一团")
    p.add_argument("--no-o3d", action="store_true", help="只要 OpenCV 窗口，不要旁路 3D")
    p.add_argument("--no-target", action="store_true", help="关掉点名通道，行为与旧 GUI 相同")
    return p.parse_args()


class InferWorker(threading.Thread):
    """后台推理，避免卡住取流刷新。"""

    def __init__(self, engine: GraspNetInfer):
        super().__init__(daemon=True)
        self.engine = engine
        self._lock = threading.Lock()
        self._frame = None
        self._pending = False
        self.result = None
        self.error = None
        self.infer_ms = 0.0
        self.busy = False
        self.stop_flag = False
        self.paused = False

    def submit(self, frame) -> None:
        if self.paused or self.busy:
            return
        with self._lock:
            self._frame = frame
            self._pending = True

    def run(self) -> None:
        while not self.stop_flag:
            frame = None
            with self._lock:
                if self._pending and self._frame is not None and not self.paused:
                    frame = self._frame
                    self._pending = False
                    self.busy = True
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            try:
                self.result = self.engine.infer_frame(frame)
                self.error = None
            except Exception as e:
                LOGGER.exception("推理失败")
                self.error = str(e)
            self.infer_ms = (time.time() - t0) * 1000.0
            self.busy = False


class InputChannel:
    """终端一行一个目标词。主循环只 poll，不会阻塞取流。"""

    def __init__(self):
        self._q: queue.Queue[str] = queue.Queue()
        self.stop_flag = False
        self._th: threading.Thread | None = None

    def start_stdin(self) -> None:
        if not sys.stdin.isatty():
            return
        self._th = threading.Thread(target=self._reader, name="stdin-target", daemon=True)
        self._th.start()

    def _reader(self) -> None:
        while not self.stop_flag:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            self.push(line)

    def push(self, text: str) -> None:
        s = str(text).strip()
        if s:
            self._q.put(s)

    def poll(self) -> str | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


class LiveCloudView:
    """常驻 Open3D 窗口：点云 + 夹爪，可用鼠标旋转。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.vis = None
        self.pcd = None
        self.grippers = []
        self._inited_bbox = False
        if not enabled:
            return
        try:
            import open3d as o3d

            self.o3d = o3d
            vis = o3d.visualization.Visualizer()
            if not vis.create_window("GraspNet 6-DoF 点云", width=960, height=720, visible=True):
                LOGGER.warning("无法创建 Open3D 窗口，仅保留 RGB 界面")
                self.enabled = False
                return
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.background_color = np.array([1, 1, 1])
            opt.point_size = 3.5
            opt.mesh_show_back_face = True
            self.vis = vis
            self.pcd = pcd
        except Exception as e:
            LOGGER.warning("Open3D 窗口不可用: %s", e)
            self.enabled = False

    def update(self, pcd, gg, top_k: int = 20) -> bool:
        if not self.enabled or self.vis is None:
            return True
        try:
            if pcd is not None and np.asarray(pcd.points).size:
                self.pcd.points = pcd.points
                self.pcd.colors = pcd.colors
                self.vis.update_geometry(self.pcd)
                if not self._inited_bbox:
                    self.vis.reset_view_point(True)
                    self._inited_bbox = True
            for g in self.grippers:
                self.vis.remove_geometry(g, reset_bounding_box=False)
            self.grippers = []
            if gg is not None and len(gg) > 0:
                show = gg[: min(int(top_k), len(gg))]
                for mesh in show.to_open3d_geometry_list():
                    self.vis.add_geometry(mesh, reset_bounding_box=False)
                    self.grippers.append(mesh)
            if not self.vis.poll_events():
                return False
            self.vis.update_renderer()
            return True
        except Exception as e:
            LOGGER.warning("Open3D 更新失败: %s", e)
            return True

    def close(self) -> None:
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass
            self.vis = None


def _hud(img: np.ndarray, lines: list[str]) -> None:
    y = 24
    for text in lines:
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22


def _open_blocking_3d(pcd, gg) -> None:
    import open3d as o3d

    geoms = [pcd] if pcd is not None else []
    if gg is not None and len(gg) > 0:
        geoms.extend(gg.to_open3d_geometry_list())
    if not geoms:
        print("当前没有点云/抓取可显示")
        return
    print("3D 窗口：鼠标旋转，滚轮缩放，关掉窗口继续实时规划")
    o3d.visualization.draw_geometries(geoms, window_name="GraspNet 当前规划（关掉继续）")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = args.from_dir
    if args.official_demo:
        data_dir = str(GRASPNET_ROOT / "doc" / "example_data")
    if data_dir:
        cam = FileCamera(data_dir, loop=True)
    else:
        cam = open_camera(cfg)
    print("正在打开相机并加载 GraspNet，请稍候…")
    cam.open()
    engine = GraspNetInfer(cfg, checkpoint=args.checkpoint)
    worker = InferWorker(engine)
    worker.start()
    view3d = LiveCloudView(enabled=not args.no_o3d)

    vlm_client = None
    qworker = None
    inch = InputChannel()
    if not args.no_target:
        try:
            from vlm.vlm_client import VLMClient

            vlm_client = VLMClient(cfg)
        except Exception as e:
            LOGGER.warning("VLM 未就绪（点名将提示错误）: %s", e)
            vlm_client = None
        qworker = QueryWorker(cfg, vlm_client)
        qworker.start()
        inch.start_stdin()

    win = "GraspNet 点名闭环  [终端输入目标  q退出  空格暂停  s保存]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    n_cam = 0
    t_fps = time.time()
    cam_fps = 0.0
    last_gg = None
    last_filtered = None
    last_all = None
    last_pcd = None
    last_n = 0
    last_score = 0.0
    cached: QueryResult | None = None
    status_msg = "等待输入目标词（终端回车）"
    print("界面已打开。把物体放到桌面中央，336L 建议相距约 0.5 m。")
    if not args.no_target:
        print("在此终端输入目标词后回车，例如：白色鼠标   （c 清除，q 退出）")
    try:
        while True:
            try:
                frame = cam.read()
            except StopIteration:
                break
            except Exception as e:
                LOGGER.warning("读帧失败: %s", e)
                time.sleep(0.05)
                continue

            n_cam += 1
            now = time.time()
            if now - t_fps >= 1.0:
                cam_fps = n_cam / (now - t_fps)
                n_cam = 0
                t_fps = now

            worker.submit(frame)
            result = worker.result
            if result is not None:
                last_gg = result["grasps"]
                last_filtered = result.get("grasps_filtered", last_gg)
                last_all = result.get("grasps_all", last_filtered)
                last_pcd = result["cloud"]
                last_n = int(len(last_gg)) if last_gg is not None else 0
                if last_n > 0:
                    last_score = float(np.asarray(last_gg.scores)[0])
                else:
                    last_score = 0.0

            if qworker is not None:
                cmd = inch.poll()
                if cmd is not None:
                    low = cmd.lower()
                    if low in ("q", "quit", "exit"):
                        break
                    if low in ("c", "clear"):
                        cached = None
                        qworker.result = None
                        qworker.error = None
                        status_msg = "已清除，等待输入目标词"
                        print(status_msg)
                    elif qworker.busy:
                        print(f"上一轮点名还在进行，已忽略：{cmd}")
                    else:
                        cached = None
                        # 用碰撞后的全部候选匹配，避免几何过滤把目标上的抓取裁光
                        match_gg = last_all if last_all is not None else (last_filtered if last_filtered is not None else last_gg)
                        qworker.submit(frame, cmd, match_gg)
                        status_msg = f"查询中：{cmd}"
                        print(status_msg)
                if not qworker.busy:
                    if qworker.error is not None:
                        cached = None
                        status_msg = qworker.error.message
                        if getattr(qworker, "_announced", None) is not qworker.error:
                            print(status_msg)
                            qworker._announced = qworker.error
                    elif qworker.result is not None:
                        cached = qworker.result
                        m = cached.match
                        status_msg = (
                            f"高亮「{cached.target}」 dist={m.distance_m*1000:.0f}mm "
                            f"score={m.score:.2f}  {cached.elapsed_s:.2f}s"
                        )
                        if getattr(qworker, "_announced", None) is not qworker.result:
                            print(status_msg)
                            qworker._announced = qworker.result

            hi = None
            draw_gg = last_gg
            if cached is not None:
                draw_gg = last_all if last_all is not None else (last_filtered if last_filtered is not None else last_gg)
                min_sc = float(cfg.get_path("live.min_match_score", 0.08))
                max_d = float(cfg.get_path("live.max_match_dist_m", 0.20))
                try:
                    live_m = match_nearest_grasp(draw_gg, cached.locate.xyz_m, min_score=min_sc, max_dist_m=max_d)
                    hi = live_m.index
                except NoGraspMatchError:
                    hi = cached.match.index if draw_gg is not None and cached.match.index < len(draw_gg) else None
            rgb_g = overlay_grasps_on_rgb(
                frame.color_bgr, draw_gg, frame.color_intrinsics, top_k=args.rgb_k, highlight_index=hi
            )
            if cached is not None:
                rgb_g = draw_locate_overlay(rgb_g, cached.locate, depth_m=frame.depth_m, label=cached.target)
                gt = None
                if hi is not None and draw_gg is not None and 0 <= hi < len(draw_gg):
                    gt = np.asarray(draw_gg.translations)[hi]
                else:
                    gt = np.asarray(cached.match.translation_m, dtype=np.float64)
                K = frame.color_intrinsics
                if gt is not None and float(gt[2]) > 1e-6:
                    gu = int(K.fx * float(gt[0]) / float(gt[2]) + K.cx)
                    gv = int(K.fy * float(gt[1]) / float(gt[2]) + K.cy)
                    cv2.line(rgb_g, tuple(cached.locate.uv), (gu, gv), (0, 255, 255), 2, cv2.LINE_AA)
            depth_v = overlay_rgbd(frame.color_bgr, frame.depth_m)
            valid = float((frame.depth_m > 0).mean())
            status = "INF" if worker.busy else ("PAUSE" if worker.paused else "RUN")
            if qworker is not None and qworker.busy:
                status = "VLM"
            z_mm = float(frame.depth_m[frame.depth_m > 0].mean() * 1000) if np.any(frame.depth_m > 0) else 0.0
            _hud(
                rgb_g,
                [
                    f"cam {cam_fps:.1f} FPS  infer {worker.infer_ms:.0f} ms  [{status}]",
                    f"grasps {last_n}  top {last_score:.2f}  depth {valid*100:.0f}%  z~{z_mm:.0f}mm",
                    "q quit | space pause | s save | 3 Open3D | type target in terminal",
                ],
            )
            put_cjk(rgb_g, status_msg, (12, rgb_g.shape[0] - 18), (0, 255, 255) if cached else (0, 0, 255) if (qworker and qworker.error) else (0, 255, 0), 20)
            if worker.error:
                cv2.putText(rgb_g, worker.error[:80], (12, rgb_g.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            h = 480
            left = cv2.resize(rgb_g, (int(rgb_g.shape[1] * h / rgb_g.shape[0]), h))
            right = cv2.resize(depth_v, (left.shape[1], h))
            canvas = np.hstack([left, right])
            cv2.imshow(win, canvas)

            if not view3d.update(last_pcd, last_gg, top_k=args.top_k):
                print("3D 窗口已关闭，RGB 界面继续运行")
                view3d.close()
                view3d.enabled = False

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                worker.paused = not worker.paused
                print("暂停" if worker.paused else "继续")
            if key == ord("s"):
                out = timestamp_dir(OUTPUT_DIR / "live")
                save_rgbd_dir(frame, out)
                if last_pcd is not None:
                    save_triplet(frame, last_pcd, last_gg, out, top_k=args.top_k)
                print(f"已保存 {out}  grasps={last_n}")
            if key == ord("3"):
                if last_pcd is not None:
                    _open_blocking_3d(last_pcd, last_gg)
    finally:
        worker.stop_flag = True
        if qworker is not None:
            qworker.stop_flag = True
        inch.stop_flag = True
        view3d.close()
        cv2.destroyAllWindows()
        cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
