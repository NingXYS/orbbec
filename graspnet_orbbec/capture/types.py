"""取流模块的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Intrinsics:
    """针孔相机内参。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[np.ndarray] = None  # k1,k2,p1,p2,k3,...

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        d = {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": int(self.width),
            "height": int(self.height),
        }
        if self.distortion is not None:
            d["distortion"] = self.distortion.astype(float).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        dist = d.get("distortion")
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["width"]),
            height=int(d["height"]),
            distortion=np.array(dist, dtype=np.float64) if dist is not None else None,
        )


@dataclass
class Extrinsics:
    """depth 相机到 color 相机的刚体变换：P_color = R @ P_depth + t。"""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.astype(float).tolist(),
            "translation": self.translation.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Extrinsics":
        return cls(
            rotation=np.array(d["rotation"], dtype=np.float64).reshape(3, 3),
            translation=np.array(d["translation"], dtype=np.float64).reshape(3),
        )


@dataclass
class RGBDFrame:
    """对齐后的一帧 RGB-D。depth 与 color 同分辨率、同坐标系。"""

    color_bgr: np.ndarray  # HxWx3 uint8, BGR
    depth_u16: np.ndarray  # HxW uint16，毫米（或 raw * scale 后的毫米）
    depth_m: np.ndarray  # HxW float32，米
    color_intrinsics: Intrinsics
    depth_intrinsics: Intrinsics
    aligned: bool
    timestamp_ns: int = 0
    depth_scale: float = 1.0  # raw * scale = mm
    depth_to_color: Optional[Extrinsics] = None
    backend: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def color_rgb(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.color_bgr, cv2.COLOR_BGR2RGB)
