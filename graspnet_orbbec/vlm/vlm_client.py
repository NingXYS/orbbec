"""百炼 OpenAI 兼容接口上的 Qwen3-VL 检测客户端。

只从环境变量 DASHSCOPE_API_KEY 读密钥；日志与异常里做脱敏。
不接实时采集循环。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from common.config import Config, load_config

LOGGER = logging.getLogger("graspnet_orbbec.vlm")
ImageLike = Union[str, Path, np.ndarray]
_KEY_RE = re.compile(r"(sk-[A-Za-z0-9]{8,})|(Bearer\s+\S+)", re.IGNORECASE)


def redact_secrets(text: str) -> str:
    """去掉日志/异常里可能出现的 key，避免泄露。"""
    if not text:
        return text
    return _KEY_RE.sub("***", text)


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            record.args = tuple(redact_secrets(a) if isinstance(a, str) else a for a in record.args)
        return True


if not any(isinstance(f, _RedactFilter) for f in LOGGER.filters):
    LOGGER.addFilter(_RedactFilter())


def _coerce_detect_dict(data: dict) -> dict:
    found = data.get("found", None)
    if isinstance(found, str):
        found = found.strip().lower() in ("1", "true", "yes", "y")
    elif found is None:
        raise ValueError("缺少 found")
    else:
        found = bool(found)
    bbox = data.get("bbox_2d", [0, 0, 0, 0] if not found else None)
    if bbox is None:
        bbox = [0, 0, 0, 0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox_2d 必须是长度为 4 的数组")
    nums = [float(x) for x in bbox]
    return {"found": found, "bbox_2d": nums}


def parse_vlm_json(text: str) -> dict:
    """从模型输出里抠 JSON，容忍 markdown、截断括号、字段散落。"""
    if text is None:
        raise ValueError("空响应")
    raw = str(text).strip()
    if not raw:
        raise ValueError("空响应")
    raw = raw.replace("```json", "```")
    if "```" in raw:
        parts = raw.split("```")
        raw = max(parts, key=lambda s: s.count("{"))
    start = raw.find("{")
    blob = None
    if start >= 0:
        depth = 0
        end = None
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            # 被 max_tokens 截断时补上缺失的 }
            blob = raw[start:] + ("}" * max(1, raw.count("{") - raw.count("}")))
        else:
            blob = raw[start:end]
    err: Optional[Exception] = None
    if blob:
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                return _coerce_detect_dict(data)
        except (json.JSONDecodeError, ValueError) as e:
            err = e
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
                if isinstance(data, dict):
                    return _coerce_detect_dict(data)
            except (json.JSONDecodeError, ValueError) as e2:
                err = e2
    found_m = re.search(r'"found"\s*:\s*(true|false)', raw, re.IGNORECASE)
    bbox_m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', raw)
    if found_m:
        found = found_m.group(1).lower() == "true"
        nums = [0.0, 0.0, 0.0, 0.0]
        if bbox_m:
            parts = [p.strip() for p in bbox_m.group(1).split(",") if p.strip()]
            if len(parts) == 4:
                nums = [float(p) for p in parts]
        return {"found": found, "bbox_2d": nums}
    raise ValueError(f"无法解析检测 JSON: {err or '无对象'}")


def bbox_to_pixels(
    bbox: list[float],
    width: int,
    height: int,
    coord_space: str = "auto",
) -> list[int]:
    """把模型坐标映射到原图像素，并保证 x1<x2、y1<y2、落在图内。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    space = (coord_space or "auto").lower()
    mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if space == "pixel":
        pass
    elif space == "norm1" or (space == "auto" and mx <= 1.5):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif space == "norm1000" or (space == "auto" and mx <= 1000):
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(round(max(0, min(width - 1, x1))))
    x2 = int(round(max(0, min(width - 1, x2))))
    y1 = int(round(max(0, min(height - 1, y1))))
    y2 = int(round(max(0, min(height - 1, y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def iou_xyxy(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _load_bgr(image: ImageLike) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    path = Path(image)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"读图失败: {path}")
    return bgr


def encode_jpeg_b64(bgr: np.ndarray, quality: int = 85, max_side: int = 1280) -> tuple[str, int, int]:
    """压成 JPEG data-url 用的 base64；返回 (b64, orig_w, orig_h)。坐标仍按原图尺寸。"""
    h, w = bgr.shape[:2]
    vis = bgr
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        vis = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii"), w, h


@dataclass
class VLMDetectResult:
    found: bool
    bbox_2d: list[int]  # 原图像素 [x1,y1,x2,y2]
    target: str
    latency_s: float
    retries: int
    model: str
    image_size: tuple[int, int]  # (w, h)
    raw_text: str = ""
    extra: dict = field(default_factory=dict)


class VLMClient:
    """qwen3-vl-flash 检测客户端。prompt / 模型名全部来自配置。"""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("未设置环境变量 DASHSCOPE_API_KEY")
        self._api_key = key
        self.base_url = str(self.cfg.get_path("vlm.base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        self.model = str(self.cfg.get_path("vlm.model", "qwen3-vl-flash-2026-01-22"))
        self.timeout_s = float(self.cfg.get_path("vlm.timeout_s", 5.0))
        self.max_retries = int(self.cfg.get_path("vlm.max_retries", 2))
        self._client = None

    def _openai(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError("未安装 openai。请在项目 conda 环境执行: pip install openai") from e
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        return self._client

    def detect(
        self,
        image: ImageLike,
        target: str,
        *,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> VLMDetectResult:
        """在静态图上定位 target，返回 found + 像素框。"""
        target = str(target).strip()
        if not target:
            raise ValueError("target 为空")
        bgr = _load_bgr(image)
        quality = int(self.cfg.get_path("vlm.jpeg_quality", 85))
        max_side = int(self.cfg.get_path("vlm.max_side", 1280))
        b64, width, height = encode_jpeg_b64(bgr, quality=quality, max_side=max_side)
        system = str(self.cfg.get_path("vlm.system_prompt", "")).strip()
        user_tmpl = str(self.cfg.get_path("vlm.user_prompt", "目标：{target}"))
        user_text = user_tmpl.format(target=target, width=width, height=height)
        enable_thinking = bool(self.cfg.get_path("vlm.enable_thinking", False))
        temperature = float(self.cfg.get_path("vlm.temperature", 0.0))
        max_tokens = int(self.cfg.get_path("vlm.max_tokens", 128))
        coord_space = str(self.cfg.get_path("vlm.coord_space", "norm1000"))
        resp_fmt = self.cfg.get_path("vlm.response_format", "json_object")
        call_timeout = float(self.timeout_s if timeout_s is None else timeout_s)
        retry_n = int(self.max_retries if max_retries is None else max_retries)

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        )

        last_err: Optional[Exception] = None
        raw_text = ""
        t0 = time.perf_counter()
        attempts = max(1, retry_n + 1)
        used = 0
        for used in range(attempts):
            try:
                remain = call_timeout - (time.perf_counter() - t0)
                if remain <= 0.05:
                    raise TimeoutError(f"VLM 超过 {call_timeout:.0f}s 时限")
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": enable_thinking},
                    timeout=remain,
                )
                if resp_fmt:
                    kwargs["response_format"] = {"type": str(resp_fmt)}
                completion = self._openai().chat.completions.create(**kwargs)
                raw_text = (completion.choices[0].message.content or "") if completion.choices else ""
                parsed = parse_vlm_json(raw_text)
                found = bool(parsed["found"])
                bbox_px = [0, 0, 0, 0]
                if found:
                    bbox_px = bbox_to_pixels(parsed["bbox_2d"], width, height, coord_space)
                    if bbox_px[2] - bbox_px[0] < 2 or bbox_px[3] - bbox_px[1] < 2:
                        raise ValueError("bbox 退化")
                latency = time.perf_counter() - t0
                LOGGER.info(
                    "vlm detect target=%r found=%s bbox=%s latency=%.3fs retries=%d model=%s",
                    target,
                    found,
                    bbox_px if found else None,
                    latency,
                    used,
                    self.model,
                )
                return VLMDetectResult(
                    found=found,
                    bbox_2d=bbox_px,
                    target=target,
                    latency_s=latency,
                    retries=used,
                    model=self.model,
                    image_size=(width, height),
                    raw_text=raw_text[:500],
                )
            except Exception as e:
                last_err = e
                LOGGER.warning(
                    "vlm 调用失败 attempt=%d/%d: %s",
                    used + 1,
                    attempts,
                    redact_secrets(f"{type(e).__name__}: {e}"),
                )
                if used + 1 >= attempts:
                    break
                time.sleep(0.15 * (used + 1))
        latency = time.perf_counter() - t0
        raise RuntimeError(
            redact_secrets(f"VLM 检测失败（{used + 1} 次）: {type(last_err).__name__}: {last_err}")
        ) from last_err
