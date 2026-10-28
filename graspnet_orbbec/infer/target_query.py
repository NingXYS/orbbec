"""点名一次：VLM → 3D 定位 → 最近抓取。不在显示主循环里调用。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from capture.types import RGBDFrame
from common.config import Config, load_config
from infer.match_grasp import GraspMatch, NoGraspMatchError, match_nearest_grasp
from vlm.target_locator import InsufficientDepthError, LocateResult, locate_target
from vlm.vlm_client import VLMConfigurationError, VLMDetectResult, redact_secrets

LOGGER = logging.getLogger("graspnet_orbbec.target_query")


class QueryError(RuntimeError):
    """点名失败。kind: not_found | timeout | network | no_depth | no_grasp | config"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class QueryResult:
    target: str
    detect: VLMDetectResult
    locate: LocateResult
    match: GraspMatch
    elapsed_s: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "bbox_2d": self.detect.bbox_2d,
            "found": self.detect.found,
            "xyz_m": self.locate.xyz_m,
            "uv": self.locate.uv,
            "match": self.match.to_dict(),
            "elapsed_s": self.elapsed_s,
            "vlm_s": self.detect.latency_s,
        }


def snapshot_frame(frame: RGBDFrame) -> RGBDFrame:
    """拷贝当前帧，避免取流线程覆盖正在查询的图。"""
    return RGBDFrame(
        color_bgr=np.ascontiguousarray(frame.color_bgr.copy()),
        depth_u16=np.ascontiguousarray(frame.depth_u16.copy()),
        depth_m=np.ascontiguousarray(frame.depth_m.copy()),
        color_intrinsics=frame.color_intrinsics,
        depth_intrinsics=frame.depth_intrinsics,
        aligned=frame.aligned,
        timestamp_ns=frame.timestamp_ns,
        depth_scale=frame.depth_scale,
        depth_to_color=frame.depth_to_color,
        backend=frame.backend,
        extra=dict(frame.extra or {}),
    )


def _classify_vlm_exception(exc: BaseException, timeout_s: float) -> QueryError:
    name = type(exc).__name__.lower()
    msg = redact_secrets(str(exc))
    blob = f"{name} {msg}".lower()
    if isinstance(exc, VLMConfigurationError):
        return QueryError("config", msg)
    if isinstance(exc, TimeoutError) or "timeout" in blob or "timed out" in blob:
        return QueryError("timeout", f"VLM 超时（{timeout_s:.0f} 秒），网络或服务无响应，请检查网络后重新输入")
    if any(k in blob for k in ("connection", "connect", "network", "dns", "unreachable", "name or service")):
        return QueryError("network", f"VLM 网络失败：{msg}。请检查网络后重新输入")
    return QueryError("network", f"VLM 调用失败：{type(exc).__name__}: {msg}。请重新输入")


def run_target_query(
    frame: RGBDFrame,
    target: str,
    grasps,
    client,
    cfg: Optional[Config] = None,
) -> QueryResult:
    """对冻结的一帧跑完点名链路。禁止在 GUI 刷新循环里直接调用。"""
    cfg = cfg or load_config()
    target = str(target).strip()
    if not target:
        raise QueryError("config", "目标词为空，请重新输入")
    timeout_s = float(cfg.get_path("live.vlm_timeout_s", 10.0))
    max_retries = int(cfg.get_path("live.vlm_max_retries", 0))
    min_score = float(cfg.get_path("live.min_match_score", cfg.get_path("filter.min_score", 0.08)))
    max_dist = float(cfg.get_path("live.max_match_dist_m", 0.20))
    t0 = time.perf_counter()
    try:
        detect = client.detect(frame.color_bgr, target, timeout_s=timeout_s, max_retries=max_retries)
    except QueryError:
        raise
    except Exception as e:
        raise _classify_vlm_exception(e, timeout_s) from e
    if not detect.found:
        raise QueryError("not_found", f"未找到目标「{target}」，请重新输入")
    try:
        locate = locate_target(frame, detect.bbox_2d, cfg=cfg, target=target)
    except InsufficientDepthError as e:
        raise QueryError("no_depth", f"无有效点云：{e}") from e
    try:
        match = match_nearest_grasp(grasps, locate.xyz_m, min_score=min_score, max_dist_m=max_dist)
    except NoGraspMatchError as e:
        raise QueryError("no_grasp", str(e)) from e
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "query target=%r xyz=%s match_i=%d dist=%.3fm score=%.3f elapsed=%.3fs",
        target,
        locate.xyz_m,
        match.index,
        match.distance_m,
        match.score,
        elapsed,
    )
    return QueryResult(target=target, detect=detect, locate=locate, match=match, elapsed_s=elapsed)


class QueryWorker:
    """后台点名。显示线程只读 result/error，绝不在主循环调 API。"""

    def __init__(self, cfg, client):
        import threading

        self.cfg = cfg
        self.client = client
        self._lock = threading.Lock()
        self._job = None
        self.result: QueryResult | None = None
        self.error: QueryError | None = None
        self.busy = False
        self.stop_flag = False
        self.elapsed_s = 0.0
        self._thread = threading.Thread(target=self.run, name="vlm-query", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame, target: str, grasps) -> None:
        if self.client is None:
            self.error = QueryError("config", "未设置环境变量 DASHSCOPE_API_KEY，无法点名")
            self.result = None
            return
        frozen = snapshot_frame(frame)
        with self._lock:
            self._job = (frozen, str(target).strip(), grasps)
            self.busy = True
            self.error = None
            self.result = None

    def run(self) -> None:
        while not self.stop_flag:
            job = None
            with self._lock:
                if self._job is not None:
                    job = self._job
                    self._job = None
            if job is None:
                time.sleep(0.01)
                continue
            frame, target, grasps = job
            t0 = time.perf_counter()
            timeout_s = float(self.cfg.get_path("live.vlm_timeout_s", 10.0))
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                fut = pool.submit(run_target_query, frame, target, grasps, self.client, self.cfg)
                try:
                    self.result = fut.result(timeout=timeout_s + 0.5)
                    self.error = None
                except FutTimeout:
                    self.result = None
                    self.error = QueryError(
                        "timeout",
                        f"VLM 超时（{timeout_s:.0f} 秒），网络或服务无响应，请检查网络后重新输入",
                    )
            except QueryError as e:
                self.result = None
                self.error = e
                LOGGER.warning("点名失败 kind=%s %s", e.kind, e.message)
            except Exception as e:
                self.result = None
                self.error = QueryError("network", f"点名异常：{type(e).__name__}: {e}。请重新输入")
                LOGGER.exception("点名未捕获异常")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
                self.elapsed_s = time.perf_counter() - t0
                self.busy = False

