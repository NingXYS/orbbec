"""SAM2线材掩码、中心线与端部附近二维候选。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from common.config import Config, load_config
from common.paths import ROOT


class SegmentationError(ValueError):
    """没有可验证的线材掩码或几何信息。"""


@dataclass
class WireResult:
    image_bgr: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    centerline_xy: np.ndarray
    candidates: list[dict]
    score: float
    elapsed_s: float
    crop_xyxy: list[int]
    warnings: list[str] = field(default_factory=list)
    target: str = "线材"
    paths_xy: list[np.ndarray] = field(default_factory=list)
    end_regions: list[np.ndarray] = field(default_factory=list)
    automation: dict = field(default_factory=dict)
    terminal_detections: list[dict] = field(default_factory=list)
    terminal_masks: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target, "mask_pixels": int(self.mask.sum()), "sam_score": self.score,
            "elapsed_s": self.elapsed_s, "crop_xyxy": self.crop_xyxy,
            "centerline_xy": self.centerline_xy.astype(int).tolist(),
            "centerlines_by_component": [p.astype(int).tolist() for p in self.paths_xy],
            "candidates": self.candidates, "warnings": self.warnings,
            "terminal_semantics_verified": False, "robot_pose_valid": False,
            "coordinate_space": "original_rgb_pixels",
            "automation": self.automation,
            "terminal_detections": self.terminal_detections,
        }


def _end_region(mask, tip, radius):
    """沿前景连通路径扩展端部区域，避免圆形邻域吞入旁边的背景/另一根线。"""
    region = np.zeros_like(mask)
    h, w = mask.shape
    tx, ty = map(int, tip)
    distances = {(ty, tx): 0.0}
    queue = [(0.0, ty, tx)]
    while queue:
        distance, y, x = heapq.heappop(queue)
        if distance > distances[(y, x)]:
            continue
        region[y, x] = True
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ny, nx = y + dy, x + dx
            d = distance + (1.41421356 if dx and dy else 1)
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and d <= radius and d < distances.get((ny, nx), np.inf):
                distances[(ny, nx)] = d
                heapq.heappush(queue, (d, ny, nx))
    return region


def _prompt_components(mask: np.ndarray, positives: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        raise SegmentationError("分割未得到前景像素")
    if len(positives):
        ids = labels[positives[:, 1], positives[:, 0]]
        if np.any(ids == 0):
            raise SegmentationError("分割未覆盖全部前景点")
        chosen = np.unique(ids)
    else:
        # 线材可能出画后再入画，不能静默丢掉第二段；后续逐段提示歧义。
        chosen = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 12)
        chosen = chosen[chosen > 0]
    return np.isin(labels, chosen)


def _skeleton_path(mask: np.ndarray, local_extent=34):
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    yx = np.argwhere(skeleton)
    if len(yx) < 8:
        return skeleton, np.empty((0, 2), dtype=int), ["掩码过小，无法估计线材端部"], []
    ids = {tuple(p): i for i, p in enumerate(yx)}
    graph = [[] for _ in yx]
    for i, (y, x) in enumerate(yx):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            j = ids.get((y + dy, x + dx))
            if j is not None:
                graph[i].append((j, 1.41421356 if dx and dy else 1.0))
    ends = [i for i, neighbors in enumerate(graph) if len(neighbors) == 1]
    if len(ends) < 2:
        return skeleton, np.empty((0, 2), dtype=int), ["中心线没有两个清晰端点，可能闭合、粘连或分割错误"], []
    if len(ends) > 40:
        return skeleton, np.empty((0, 2), dtype=int), ["中心线分支过多，端部不可靠，请缩小提示区域或补充点选"], []

    best = None
    start = ends[0]
    # 两次最远点搜索限制分支场景的耗时；有环/分支时仅是主路径，不保证全局最长。
    for _ in range(2):
        dist = np.full(len(yx), np.inf)
        parent = np.full(len(yx), -1, int)
        dist[start] = 0
        queue = [(0.0, start)]
        while queue:
            distance, i = heapq.heappop(queue)
            if distance > dist[i]:
                continue
            for j, weight in graph[i]:
                candidate = distance + weight
                if candidate < dist[j]:
                    dist[j], parent[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        end = max(ends, key=lambda j: dist[j] if np.isfinite(dist[j]) else -1)
        if best is None or dist[end] > best[0]:
            best = (dist[end], start, end, parent)
        start = end
    _, start, end, parent = best
    path = [end]
    while path[-1] != start:
        path.append(int(parent[path[-1]]))
    xy = yx[path, ::-1].copy()
    warnings = []
    local_paths = []
    if len(ends) != 2:
        warnings.append(f"中心线存在{len(ends)}个端点，逐端给出几何候选；粘连处不保证线材实例对应")
        # 多束粘连时逐端短距离搜索，不能只保留主路径的两个端点。
        for endpoint in ends:
            distances, parents = {endpoint: 0.0}, {}
            queue = [(0.0, endpoint)]
            while queue:
                distance, i = heapq.heappop(queue)
                if distance > distances[i]:
                    continue
                for j, weight in graph[i]:
                    d = distance + weight
                    if d <= local_extent and d < distances.get(j, np.inf):
                        distances[j], parents[j] = d, i
                        heapq.heappush(queue, (d, j))
            end = max(distances, key=distances.get)
            local = [end]
            while local[-1] != endpoint:
                local.append(parents[local[-1]])
            local_paths.append(yx[local[::-1], ::-1].copy())
    return skeleton, xy, warnings, local_paths


def analyze_mask(image_bgr: np.ndarray, mask: np.ndarray, *, score: float = 0,
                 elapsed_s: float = 0, crop_xyxy=None, inset_px: float = 12,
                 target: str = "线材") -> WireResult:
    """端部由掩码几何推断，不宣称识别了金属端子或可执行抓取。"""
    if mask.shape != image_bgr.shape[:2] or mask.ndim != 2:
        raise SegmentationError("掩码必须与原始RGB同尺寸")
    mask = mask.astype(bool)
    if int(mask.sum()) < 12:
        raise SegmentationError("前景太少，无法分割线材")
    if not np.isfinite(inset_px) or inset_px <= 0:
        raise SegmentationError("端部内移像素必须为正数")
    h, w = mask.shape
    crop_xyxy = list(crop_xyxy or [0, 0, w, h])
    x1, y1, x2, y2 = crop_xyxy
    count, components = cv2.connectedComponents(mask.astype(np.uint8), 8)
    skeleton = np.zeros_like(mask)
    paths, warnings = [], []
    candidates, end_regions = [], []
    if count > 2:
        warnings.append("掩码包含多段不连通前景，不能据此确认它们属于同一根线材")
    for component_id in range(1, count):
        part_skeleton, path, part_warnings, local_paths = _skeleton_path(components == component_id, 2 * inset_px + 10)
        skeleton |= part_skeleton
        warnings.extend(part_warnings)
        if not len(path):
            continue
        paths.append(path)
        total_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if total_length < 2 * inset_px + 5:
            warnings.append("中心线过短，无法在两端附近给出独立的候选区域")
        else:
            for directed in (local_paths or [path, path[::-1]]):
                tx, ty = directed[0]
                # 画面/搜索裁剪边缘的截断不是物理端点，不能给出端部抓取候选。
                if min(tx - x1, ty - y1, x2 - 1 - tx, y2 - 1 - ty) <= 4:
                    warnings.append(f"第{component_id}段有端点接近画面或裁剪边界，已排除截断端候选")
                    continue
                distances = np.r_[0.0, np.linalg.norm(np.diff(directed, axis=0), axis=1).cumsum()]
                if distances[-1] < inset_px + 5:
                    warnings.append('某端点附近可用中心线过短，未生成内移候选')
                    continue
                i = int(np.searchsorted(distances, inset_px))
                a, b = max(0, i - 5), min(len(directed) - 1, i + 5)
                vector = directed[b] - directed[a]
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                axis = vector.astype(float) / norm
                region = _end_region(components == component_id, directed[0], 2 * inset_px)
                end_regions.append(region)
                candidates.append({"id": chr(65 + len(candidates)), "component_id": component_id,
                                   "tip_xy": directed[0].astype(int).tolist(),
                                   "near_tip_xy": directed[i].astype(int).tolist(),
                                   "tangent_xy": axis.tolist(),
                                   "inset_px": float(distances[i]),
                                   "end_region_pixels": int(region.sum()),
                                   "end_region_extent_px": float(2 * inset_px),
                                   "kind": "geometric_end_candidate", "confirmed": False})
    # 多段中心线不拼接，避免下游误把出画间隙当成一段真实线身。
    path = paths[0] if len(paths) == 1 else np.empty((0, 2), dtype=int)
    return WireResult(image_bgr.copy(), mask, skeleton, path, candidates, float(score),
                      float(elapsed_s), crop_xyxy, warnings, target, paths, end_regions)


class Sam2WireSegmenter:
    """按框或正负点提示分割；模型常驻，提示框仅限制搜索，不作为输出掩码。"""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.predictor = None
        self.device = None

    def _load(self):
        if self.predictor is not None:
            return
        for path in (ROOT / "third_party/sam2_runtime", ROOT / "third_party/sam2"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("SAM2依赖未就绪，请运行 bash scripts/install_sam2.sh") from exc
        checkpoint = Path(self.cfg.get_path("wire.checkpoint", "weights/sam2/sam2.1_hiera_small.pt"))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise RuntimeError(f"缺少SAM2权重：{checkpoint}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = str(self.cfg.get_path("wire.model_config", "configs/sam2.1/sam2.1_hiera_s.yaml"))
        model = build_sam2(model_cfg, str(checkpoint), device=self.device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)

    def segment(self, image_bgr: np.ndarray, *, box=None, points=None, labels=None, target="线材") -> WireResult:
        import torch

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise SegmentationError("需要原始BGR彩色图像")
        h, w = image_bgr.shape[:2]
        points = np.asarray(points if points is not None else [], dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(labels if labels is not None else np.ones(len(points)), dtype=np.int32)
        if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all():
            raise SegmentationError("点提示标签必须为0/1，且与点数相同")
        if not np.isfinite(points).all() or (len(points) and
                (np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h))):
            raise SegmentationError("点提示必须位于原始图像内")
        crop_box = [0, 0, w, h]
        prompt_box = None
        if box is not None:
            b = np.asarray(box, dtype=float)
            if b.shape != (4,) or not np.isfinite(b).all():
                raise SegmentationError("提示框须为有限的[x1,y1,x2,y2]")
            x1, y1, x2, y2 = map(int, np.rint(b))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                raise SegmentationError("提示框太小或坐标顺序错误")
            margin = int(self.cfg.get_path("wire.crop_margin_px", 24))
            crop_box = [max(0, x1 - margin), max(0, y1 - margin), min(w, x2 + margin + 1), min(h, y2 + margin + 1)]
            prompt_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        if prompt_box is None and not np.any(labels == 1):
            raise SegmentationError("请先框选线材、输入目标词，或在线身上点一个前景点")
        cx1, cy1, cx2, cy2 = crop_box
        if len(points) and (np.any(points[:, 0] < cx1) or np.any(points[:, 0] >= cx2)
                            or np.any(points[:, 1] < cy1) or np.any(points[:, 1] >= cy2)):
            raise SegmentationError("点提示在裁剪区域外，请扩大框或清除提示框")
        self._load()
        t0 = time.perf_counter()
        crop = cv2.cvtColor(image_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        offset = np.array([cx1, cy1], dtype=np.float32)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
        with torch.inference_mode(), amp:
            self.predictor.set_image(crop)
            masks, scores, _ = self.predictor.predict(
                point_coords=points - offset if len(points) else None,
                point_labels=labels if len(points) else None,
                box=prompt_box - np.tile(offset, 2) if prompt_box is not None else None,
                multimask_output=True,
            )
        positives = np.rint(points[labels == 1] - offset).astype(int)
        negatives = np.rint(points[labels == 0] - offset).astype(int)
        for seeds in (positives, negatives):
            seeds[:, 0] = np.clip(seeds[:, 0], 0, crop.shape[1] - 1)
            seeds[:, 1] = np.clip(seeds[:, 1], 0, crop.shape[0] - 1)
        choices = []
        for mask, score in zip(masks, scores):
            if not np.isfinite(score) or float(score) < float(self.cfg.get_path("wire.min_sam_score", 0.5)):
                continue
            mask = mask.astype(bool)
            if len(positives) and not mask[positives[:, 1], positives[:, 0]].all():
                continue
            if len(negatives) and mask[negatives[:, 1], negatives[:, 0]].any():
                continue
            try:
                component = _prompt_components(mask, positives)
            except SegmentationError:
                continue
            ys, xs = np.where(component)
            if len(xs) < 12:
                continue
            fill = len(xs) / ((int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
            if component.mean() > float(self.cfg.get_path("wire.max_mask_fraction", 0.65)):
                continue
            # 细线优先，防止框提示把整块桌面作为高分前景；保留真实SAM评分供核查。
            rank = float(score) - 0.25 * fill
            choices.append((rank, float(score), component))
        if not choices:
            raise SegmentationError("SAM2未得到符合提示的线材掩码，请在线身补前景点、背景补负点")
        _, score, selected = max(choices, key=lambda row: row[0])
        mask = np.zeros((h, w), dtype=bool)
        mask[cy1:cy2, cx1:cx2] = selected
        return analyze_mask(image_bgr, mask, score=score, elapsed_s=time.perf_counter() - t0,
                            crop_xyxy=crop_box, inset_px=float(self.cfg.get_path("wire.inset_px", 12)), target=target)


def draw_result(result: WireResult) -> np.ndarray:
    vis = result.image_bgr.copy()
    tint = np.zeros_like(vis)
    tint[:] = (0, 180, 255)
    vis[result.mask] = (vis[result.mask] * 0.40 + tint[result.mask] * 0.60).astype(np.uint8)
    contours, _ = cv2.findContours(result.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
    for region in result.end_regions:
        vis[region] = (vis[region] * 0.2 + np.array([0, 255, 0]) * 0.8).astype(np.uint8)
    vis[result.skeleton] = (255, 0, 255)
    for item in result.candidates:
        tip = np.asarray(item["tip_xy"], dtype=int)
        near = np.asarray(item["near_tip_xy"], dtype=int)
        axis = np.asarray(item["tangent_xy"])
        normal = np.array([-axis[1], axis[0]])
        cv2.circle(vis, tuple(tip), 5, (255, 255, 0), 1)
        cv2.circle(vis, tuple(near), 7, (0, 0, 255), 1)
        cv2.line(vis, tuple(np.rint(near - normal * 9).astype(int)),
                 tuple(np.rint(near + normal * 9).astype(int)), (0, 0, 255), 2)
        cv2.putText(vis, item["id"], tuple(near + [9, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for detection, mask in zip(result.terminal_detections, result.terminal_masks):
        vis[mask] = (vis[mask] * 0.3 + np.array([255, 220, 0]) * 0.7).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)
        x, y = np.rint(detection['bbox_xyxy'][:2]).astype(int)
        cv2.putText(vis, f"{detection['id']} {detection['confidence']:.2f}", (x + 12, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return vis


def save_result(result: WireResult, out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, img in (("color.png", result.image_bgr), ("mask.png", result.mask.astype(np.uint8) * 255),
                      ("skeleton.png", result.skeleton.astype(np.uint8) * 255), ("overlay.png", draw_result(result)),
                      *[(f"end_region_{c['id']}.png", region.astype(np.uint8) * 255)
                        for c, region in zip(result.candidates, result.end_regions)],
                      *[(d['mask_file'], mask.astype(np.uint8) * 255)
                        for d, mask in zip(result.terminal_detections, result.terminal_masks)]):
        if not cv2.imwrite(str(out / name), img):
            raise OSError(f"图像保存失败：{name}")
    (out / "recognition.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return out
