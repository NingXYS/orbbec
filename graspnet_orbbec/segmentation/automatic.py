"""本地自动线材候选与端子模型辅助识别；不依赖手工框或云端服务。"""

from contextlib import nullcontext
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from common.config import load_config
from common.paths import ROOT
from segmentation.wire import Sam2WireSegmenter, SegmentationError, WireResult, analyze_mask


def line_contrast(image, cfg):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = max(9, int(min(gray.shape) * 0.035) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    contrast = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
                          cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
    threshold = max(float(cfg.get_path('wire.auto.min_contrast', 12)), float(np.percentile(contrast, 85)))
    return contrast, threshold


def automatic_line_seeds(image, cfg, extra_points=()):
    """从局部明暗细线提取稀疏提示，按连通段轮流采样，避免大线束独占预算。"""
    from skimage.morphology import skeletonize
    contrast, threshold = line_contrast(image, cfg)
    binary = (contrast >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    valid = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 16]
    skeleton = skeletonize(np.isin(labels, valid))
    parts = [np.argwhere(skeleton & (labels == i)) for i in valid]
    parts = sorted([p for p in parts if len(p) >= 8], key=len, reverse=True)
    maximum = int(cfg.get_path('wire.auto.max_seeds', 64))
    spacing = float(cfg.get_path('wire.auto.seed_spacing_px', 32))
    if maximum < 1 or spacing <= 0:
        raise ValueError('max_seeds 和 seed_spacing_px 必须为正数')
    seeds = [list(map(int, p)) for p in extra_points][:maximum]
    exhausted = set()
    rounds = 0
    while len(seeds) < maximum and len(exhausted) < len(parts):
        for i, part in enumerate(parts):
            if i in exhausted or len(seeds) >= maximum:
                continue
            if rounds == 0:
                idx = len(part) // 2
            else:
                distances = np.min(((part[:, None, :] - np.asarray(seeds)[None, :, ::-1]) ** 2).sum(2), axis=1)
                idx = int(distances.argmax())
                if distances[idx] < spacing ** 2:
                    exhausted.add(i)
                    continue
            point = part[idx, ::-1].tolist()
            if point not in seeds:
                seeds.append(point)
        rounds += 1
    return np.asarray(seeds, np.float32).reshape(-1, 2), len(seeds) >= maximum


def prune_continuing_endpoints(result, cfg):
    """原图线条仍向外延伸时，掩码断口不能当作物理线头；保留拒绝原因供复核。"""
    contrast, threshold = line_contrast(result.image_bgr, cfg)
    h, w = contrast.shape
    candidates, regions, rejected = [], [], []
    for candidate, region in zip(result.candidates, result.end_regions):
        tip = np.asarray(candidate['tip_xy'])
        axis = np.asarray(candidate['tangent_xy'])
        hits = []
        for distance in range(4, 17, 2):
            x, y = np.rint(tip - axis * distance).astype(int)
            if 2 <= x < w - 2 and 2 <= y < h - 2:
                hits.append(bool(contrast[y-2:y+3,x-2:x+3].max() >= threshold))
        if not candidate.get('terminal_model_evidence') and len(hits) >= 3 and np.mean(hits) >= .6:
            rejected.append(dict(candidate, rejection_reason='image_line_continues_past_mask_tip'))
            continue
        candidate['id'] = chr(65+len(candidates)) if len(candidates)<26 else f'E{len(candidates)+1}'
        candidates.append(candidate)
        regions.append(region)
    result.candidates, result.end_regions = candidates, regions
    result.automation['rejected_geometric_ends'] = rejected
    if rejected:
        result.warnings.append(f'原图线条仍在延伸，已排除{len(rejected)}个疑似掩码断口候选')


def select_wire_masks(image, records, cfg, reference_shape=None):
    """筛选线状前景，保留没有可靠端点的线身；几何筛选不赋予类别语义。"""
    h, w = image.shape[:2]
    scale = min(reference_shape or (h, w))
    choices = []
    seen = []
    for record in sorted(records, key=lambda r: float(r['predicted_iou']), reverse=True):
        mask = np.asarray(record['segmentation'], dtype=bool)
        if mask.shape != (h, w):
            continue
        score = float(record['predicted_iou'])
        if not np.isfinite(score) or score < float(cfg.get_path('wire.auto.min_score', 0.5)):
            continue
        if mask.sum() < 12 or mask.mean() > float(cfg.get_path('wire.auto.max_area_fraction', 0.20)):
            continue
        if any(np.logical_and(mask, previous).sum() / min(mask.sum(), previous.sum()) > 0.85 for previous in seen):
            continue
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask)
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        width = 2 * float(np.median(dt[skeleton])) if skeleton.any() else 0
        if not width or width > scale * float(cfg.get_path('wire.auto.max_width_fraction', 0.02)):
            continue
        geometry = analyze_mask(image, mask, inset_px=float(cfg.get_path('wire.inset_px', 12)))
        length = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in geometry.paths_xy)
        if not length:
            length = float(skeleton.sum())
        slenderness = length / max(width, 1)
        if (length < scale * float(cfg.get_path('wire.auto.min_length_fraction', 0.06))
                or slenderness < float(cfg.get_path('wire.auto.min_slenderness', 12))):
            continue
        seen.append(mask)
        choices.append((score * slenderness, mask, {
            'sam_score': score, 'median_width_px': width, 'centerline_length_px': length,
            'slenderness': slenderness,
        }))
    selected = []
    for _, mask, info in sorted(choices, key=lambda row: row[0], reverse=True):
        # 同一根线的包含/重复掩码只保留一份，避免重复输出端部。
        if any(np.logical_and(mask, previous).sum() / min(mask.sum(), previous.sum()) > 0.75
               for previous, _ in selected):
            continue
        selected.append((mask, info))
        if len(selected) >= int(cfg.get_path('wire.auto.max_masks', 32)):
            break
    return selected


def attach_terminals(result, detections, masks):
    """仅关联与端部区域重叠的检测，未匹配的几何候选仍保留未确认状态。"""
    result.terminal_detections = detections
    result.terminal_masks = masks
    for candidate, region in zip(result.candidates, result.end_regions):
        evidence = []
        for detection, mask in zip(detections, masks):
            if np.any(region & mask):
                evidence.append({'id': detection['id'], 'confidence': detection['confidence']})
        candidate['terminal_model_evidence'] = evidence
        candidate['terminal_status'] = 'model_detected' if evidence else 'geometry_only'
    if not detections:
        result.warnings.append('端子模型未检出端子；当前只有线材几何端部候选')
    elif any(not item['terminal_model_evidence'] for item in result.candidates):
        result.warnings.append('部分几何端部没有端子模型支持，不能将它们全部视为已识别端子')


class AutomaticWireRecognizer:
    def __init__(self, cfg=None, segmenter=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.segmenter = segmenter or Sam2WireSegmenter(self.cfg)
        self.generator = None
        self.terminal_model = None
        self.terminal_model_class = None
        self.terminal_runtime_error = None
        self.generation_stats = {}
        self.terminal_seed_points = []

    def prepare_terminal_runtime(self):
        """主线程在取流/窗口/推理线程启动前调用，防止库导入修改OpenCV全局线程池。"""
        if (not self.cfg.get_path('wire.auto.terminal_enabled', True)
                or self.terminal_model_class is not None or self.terminal_runtime_error is not None):
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('请先在主线程调用prepare_terminal_runtime，再启动彩色采集和推理线程')
        try:
            runtime = str(ROOT / '.yolo-deps')
            if runtime not in sys.path:
                sys.path.insert(0, runtime)
            os.environ.setdefault('YOLO_OFFLINE', 'true')
            os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
            os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'output/yolo_config'))
            from ultralytics import YOLO
            self.terminal_model_class = YOLO
        except Exception as exc:
            # 缓存失败，不得在窗口已运行后重试首次导入。
            self.terminal_runtime_error = str(exc)

    def _generate(self, image):
        import torch
        self.segmenter._load()
        amp = (torch.autocast('cuda', dtype=torch.bfloat16)
               if self.segmenter.device == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext())
        if self.cfg.get_path('wire.auto.method', 'line_prompts') != 'dense':
            points, capped = automatic_line_seeds(image, self.cfg, self.terminal_seed_points)
            self.generation_stats = {'method': 'line_prompts', 'seed_count': len(points),
                                     'seed_budget_reached': capped, 'seed_points_xy': points.tolist()}
            if not len(points):
                return []
            records = []
            batch = int(self.cfg.get_path('wire.auto.prompt_batch', 8))
            if batch < 1:
                raise ValueError('prompt_batch 必须为正数')
            with torch.inference_mode(), amp:
                self.segmenter.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                for i in range(0, len(points), batch):
                    pts = points[i:i + batch, None, :]
                    masks, scores, _ = self.segmenter.predictor.predict(
                        point_coords=pts, point_labels=np.ones(pts.shape[:2], np.int32), multimask_output=True)
                    if masks.ndim == 3:
                        masks, scores = masks[None], scores[None]
                    for point, alternatives, qualities in zip(points[i:i + batch], masks, scores):
                        x, y = map(int, point)
                        for mask, score in zip(alternatives, qualities):
                            if np.isfinite(score) and float(score) >= float(self.cfg.get_path('wire.auto.min_score', 0.5)) and mask[y, x]:
                                records.append({'segmentation': mask, 'predicted_iou': float(score)})
            return records
        if self.generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self.generator = SAM2AutomaticMaskGenerator(
                self.segmenter.predictor.model,
                points_per_side=int(self.cfg.get_path('wire.auto.points_per_side', 32)),
                points_per_batch=int(self.cfg.get_path('wire.auto.points_per_batch', 32)),
                pred_iou_thresh=float(self.cfg.get_path('wire.auto.min_score', 0.5)),
                stability_score_thresh=float(self.cfg.get_path('wire.auto.stability_score', 0.8)),
                min_mask_region_area=0,
            )
        self.generation_stats = {'method': 'dense', 'seed_count': int(self.cfg.get_path('wire.auto.points_per_side', 32)) ** 2}
        with torch.inference_mode(), amp:
            return self.generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def _detect_terminals(self, image):
        import torch
        weights = Path(self.cfg.get_path('wire.auto.terminal_weights', '../best.pt'))
        if not weights.is_absolute():
            weights = ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(f'端子权重不存在：{weights}；没有下载或替换模型')
        if self.terminal_model is None:
            self.prepare_terminal_runtime()
            if self.terminal_runtime_error is not None:
                raise RuntimeError(self.terminal_runtime_error)
            self.terminal_model = self.terminal_model_class(str(weights.resolve()))
            if self.terminal_model.task != 'segment':
                raise ValueError('端子模型必须是分割模型，不能把检测框当成端子掩码')
        result = self.terminal_model.predict(
            image, conf=float(self.cfg.get_path('wire.auto.terminal_confidence', 0.4)),
            imgsz=640, device='0' if torch.cuda.is_available() else 'cpu',
            retina_masks=True, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return [], []
        if result.masks is None or tuple(result.masks.data.shape[1:]) != image.shape[:2]:
            raise ValueError('端子模型未返回与原图同尺寸的像素掩码')
        detections, masks = [], []
        allowed = self.cfg.get_path('wire.auto.terminal_class_ids', [0])
        for box, tensor in zip(result.boxes.cpu(), result.masks.data.cpu()):
            cls = int(box.cls.item())
            if cls not in allowed:
                continue
            name = f'T{len(detections) + 1}'
            detections.append({
                'id': name, 'class_id': cls, 'raw_class_name': result.names[cls],
                'label': self.cfg.get_path('wire.auto.terminal_label', '线材金属端子'),
                'confidence': float(box.conf.item()), 'bbox_xyxy': box.xyxy[0].tolist(),
                'kind': 'terminal_model_detection', 'confirmed': False,
                'mask_file': f'terminal_{name}.png',
            })
            masks.append(tensor.numpy() > 0.5)
        return detections, masks

    def recognize(self, image):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise SegmentationError('自动识别需要原始BGR彩色图像')
        full_image = image
        h, w = image.shape[:2]
        roi = self.cfg.get_path('wire.auto.roi_xyxy')
        if roi is None:
            roi = [0, 0, w, h]
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(type(v) is not int for v in roi)):
            raise ValueError('自动识别工作区必须是原图整数像素xyxy')
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and min(x2-x1, y2-y1) >= 32):
            raise ValueError('自动识别工作区越界或小于32像素')
        image = image[y1:y2, x1:x2]
        start = time.perf_counter()
        detections, terminal_masks, terminal_error = [], [], None
        terminal_enabled = self.cfg.get_path('wire.auto.terminal_enabled', True)
        if terminal_enabled:
            try:
                detections, terminal_masks = self._detect_terminals(image)
            except Exception as exc:
                terminal_error = str(exc)
        self.terminal_seed_points = []
        for mask in terminal_masks:
            dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            if dt.any():
                y, x = np.unravel_index(dt.argmax(), dt.shape)
                self.terminal_seed_points.append([int(x), int(y)])
        terminal_s = time.perf_counter() - start
        records = self._generate(image)
        sam_s = time.perf_counter() - start - terminal_s
        selected = select_wire_masks(image, records, self.cfg, reference_shape=full_image.shape[:2])
        if 'seed_points_xy' in self.generation_stats:
            self.generation_stats['seed_points_xy'] = (
                np.asarray(self.generation_stats['seed_points_xy']).reshape(-1, 2)+[x1, y1]).tolist()
        self.generation_stats['coordinate_space'] = 'original_rgb_pixels'
        # 推理可限于工作区，但所有保存像素、交界和手眼射线必须恢复到原始RGB坐标。
        def restore(mask):
            output = np.zeros((h, w), bool)
            output[y1:y2, x1:x2] = mask
            return output
        terminal_masks = [restore(mask) for mask in terminal_masks]
        for detection in detections:
            box = np.asarray(detection['bbox_xyxy']) + [x1, y1, x1, y1]
            detection['bbox_xyxy'] = box.tolist()
        if selected:
            mask = restore(np.logical_or.reduce([m for m, _ in selected]))
            result = analyze_mask(full_image, mask, crop_xyxy=roi,
                                  score=float(np.mean([i['sam_score'] for _, i in selected])),
                                  inset_px=float(self.cfg.get_path('wire.inset_px', 12)))
        else:
            blank = np.zeros(full_image.shape[:2], bool)
            result = WireResult(full_image.copy(), blank, blank.copy(), np.empty((0, 2), int), [], 0, 0,
                                list(roi), ['未分出线身；端子检测单独保留，可按R重新取景'])
        result.automation = {
            'method': self.generation_stats.get('method', 'line_prompts'), 'manual_prompts': False,
            'generated_masks': len(records), 'selected_masks': [info for _, info in selected],
            'sam_score_aggregation': 'mean_predicted_iou_of_selected_masks',
            'external_api_calls': 0,
            'roi_xyxy_original_rgb': list(roi),
            'sampling': self.generation_stats,
            'timing_s': {'terminal': terminal_s, 'sam': sam_s},
        }
        result.warnings.append('自动线状候选不代表已确认线材类别；遮挡/粘连处不保证分实例')
        if self.generation_stats.get('seed_budget_reached'):
            result.warnings.append('本帧达到自动提示预算，可能漏检；可补点或增加max_seeds')
        for candidate in result.candidates:
            candidate['terminal_model_evidence'] = []
            candidate['terminal_status'] = 'geometry_only'
        if terminal_enabled:
            if terminal_error is None:
                attach_terminals(result, detections, terminal_masks)
                result.automation['terminal_status'] = 'ok'
            else:
                result.automation['terminal_status'] = 'unavailable'
                result.warnings.append(f'端子模型不可用：{terminal_error}；仅显示几何候选')
        else:
            result.automation['terminal_status'] = 'disabled'
            result.warnings.append('端子模型已关闭；仅显示几何端部候选')
        prune_continuing_endpoints(result, self.cfg)
        result.elapsed_s = time.perf_counter() - start
        return result
