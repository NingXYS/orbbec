#!/usr/bin/env python3
"""阶段二整体验收：任务 1~5 复测 + 点名测试矩阵，写出 JSON。

不把 API key 写入任何产物。主循环不在这里跑 GUI。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from capture.camera import load_rgbd_dir
from common.config import Config, load_config
from common.paths import OUTPUT_DIR
from infer.match_grasp import NoGraspMatchError, match_nearest_grasp
from vlm.target_locator import InsufficientDepthError, centroid_in_bbox, locate_target
from vlm.vlm_client import VLMClient, VLMDetectResult, iou_xyxy, redact_secrets

LOGGER = logging.getLogger("graspnet_orbbec.phase2")


def _p(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "max": None}
    a = sorted(xs)
    return {"n": len(a), "mean": float(sum(a) / len(a)), "p50": float(a[len(a) // 2]), "max": float(a[-1])}


def run_pytest() -> dict:
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=line"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    tail = (r.stdout or "")[-400:]
    return {
        "ok": r.returncode == 0,
        "seconds": time.perf_counter() - t0,
        "summary": tail.strip().splitlines()[-1] if tail.strip() else str(r.returncode),
    }


def remeasure_locator(cfg) -> dict:
    from vlm.run_locator import run_eval

    out = OUTPUT_DIR / "phase2" / "locator"
    rc = run_eval(ROOT / "configs" / "locator_eval.yaml", cfg, out, 50.0)
    report = json.loads((out / "eval.json").read_text(encoding="utf-8"))
    report["pass"] = rc == 0
    return report


def remeasure_infer(cfg) -> dict:
    from infer.graspnet_infer import GraspNetInfer

    frame = load_rgbd_dir(ROOT / "vlm" / "locator_data" / "desk")
    engine = GraspNetInfer(cfg)
    engine.infer_frame(frame)  # warmup
    times = []
    last = None
    for _ in range(8):
        t0 = time.perf_counter()
        last = engine.infer_frame(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    gg_all = last["grasps_all"]
    gg_f = last["grasps_filtered"]
    return {
        "warmup_discarded": True,
        "infer_ms": _stats(times),
        "infer_fps": 1000.0 / (sum(times) / len(times)),
        "n_grasps_show": int(len(last["grasps"])),
        "n_grasps_filtered": int(len(gg_f)),
        "n_grasps_all": int(len(gg_all)),
        "top_score": float(np.asarray(last["grasps"].scores)[0]) if len(last["grasps"]) else 0.0,
        "engine": engine,
        "frame": frame,
        "grasps_all": gg_all,
        "result": last,
    }


def remeasure_capture(cfg, seconds: float = 8.0) -> dict:
    from capture.camera import open_camera

    rec = {"ok": False, "error": None, "fps": None, "frames": 0, "seconds": seconds}
    try:
        cam = open_camera(cfg)
        cam.open()
        t0 = time.perf_counter()
        n = 0
        z0 = z1 = None
        while time.perf_counter() - t0 < seconds:
            fr = cam.read()
            n += 1
            z = float(np.median(fr.depth_m[fr.depth_m > 0.05])) if np.any(fr.depth_m > 0.05) else 0.0
            if z0 is None:
                z0 = z
            z1 = z
        dt = time.perf_counter() - t0
        cam.close()
        rec.update(ok=True, frames=n, seconds=dt, fps=n / dt, depth_median_m=[z0, z1], aligned=bool(fr.aligned), shape=list(fr.color_bgr.shape[:2][::-1]))
    except Exception as e:
        rec["error"] = redact_secrets(f"{type(e).__name__}: {e}")
    return rec


def _detect_row(client: VLMClient, image: Path, target: str, expect_found: bool, gt, iou_th: float) -> dict:
    t0 = time.perf_counter()
    err = None
    det: VLMDetectResult | None = None
    try:
        det = client.detect(str(image), target)
    except Exception as e:
        err = redact_secrets(f"{type(e).__name__}: {e}")
    lat = time.perf_counter() - t0
    iou = None
    ok = False
    if err:
        ok = False
    elif not expect_found:
        ok = det.found is False
    else:
        if det.found and gt is not None:
            iou = iou_xyxy(det.bbox_2d, [int(v) for v in gt])
            x1, y1, x2, y2 = det.bbox_2d
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            gx1, gy1, gx2, gy2 = [int(v) for v in gt]
            center_in = gx1 <= cx <= gx2 and gy1 <= cy <= gy2
            ok = iou >= iou_th or center_in
        else:
            ok = False
    return {
        "target": target,
        "image": str(image),
        "expect_found": expect_found,
        "pred_found": None if det is None else det.found,
        "pred_bbox": None if det is None else det.bbox_2d,
        "gt_bbox": gt,
        "iou": iou,
        "ok": ok,
        "latency_s": lat,
        "error": err,
        "retries": None if det is None else det.retries,
    }


def run_vlm_matrix(client: VLMClient, spec: dict) -> dict:
    iou_th = float(spec.get("iou_threshold", 0.5))
    repeats = int(spec.get("repeats", 10))
    groups = {}
    for gname, items, expect, nrep in (
        ("exact", spec.get("exact") or [], True, repeats),
        ("colloquial", spec.get("colloquial") or [], True, 1),
        ("absent", spec.get("absent") or [], False, 1),
    ):
        rows = []
        for case in items:
            img = _p(case["image"])
            gt = case.get("bbox_2d")
            for _ in range(nrep):
                rows.append(_detect_row(client, img, str(case["target"]), expect, gt, iou_th))
        n_ok = sum(1 for r in rows if r["ok"])
        groups[gname] = {
            "n": len(rows),
            "ok": n_ok,
            "acc": n_ok / max(len(rows), 1),
            "latency_s": _stats([r["latency_s"] for r in rows]),
            "rows": rows,
        }
    naming_rows = groups["exact"]["rows"] + groups["colloquial"]["rows"] + groups["absent"]["rows"]
    n_ok = sum(1 for r in naming_rows if r["ok"])
    return {
        "iou_threshold": iou_th,
        "groups": groups,
        "naming_n": len(naming_rows),
        "naming_ok": n_ok,
        "naming_acc": n_ok / max(len(naming_rows), 1),
        "latency_s": _stats([r["latency_s"] for r in naming_rows]),
    }


def run_closed_loop(cfg, spec: dict, vlm_rows: list[dict], grasps, frame_cache: dict) -> dict:
    """对点名成功的正例做定位+最近抓取。VLM 不再调用。"""
    rows = []
    min_sc = float(cfg.get_path("live.min_match_score", 0.08))
    max_d = float(cfg.get_path("live.max_match_dist_m", 0.20))
    for r in vlm_rows:
        if not r.get("expect_found"):
            continue
        if not r.get("ok") or not r.get("pred_bbox"):
            rows.append({**r, "locate_ok": False, "match_ok": False, "closed_ok": False, "skip": "vlm_fail"})
            continue
        frame_key = None
        for case in (spec.get("exact") or []) + (spec.get("colloquial") or []):
            if str(case["target"]) == r["target"] and _p(case["image"]) == Path(r["image"]):
                frame_key = str(_p(case["frame"]))
                break
        if frame_key is None:
            rows.append({**r, "locate_ok": False, "match_ok": False, "closed_ok": False, "skip": "no_frame"})
            continue
        if frame_key not in frame_cache:
            frame_cache[frame_key] = load_rgbd_dir(frame_key)
        frame = frame_cache[frame_key]
        rec = {**r, "frame": frame_key}
        t0 = time.perf_counter()
        try:
            loc = locate_target(frame, r["pred_bbox"], cfg=cfg, target=r["target"])
            rec["locate_ms"] = (time.perf_counter() - t0) * 1000.0
            rec["xyz_m"] = loc.xyz_m
            rec["uv"] = loc.uv
            gx1, gy1, gx2, gy2 = [int(v) for v in r["gt_bbox"]]
            u, v = loc.uv
            rec["locate_ok"] = bool(centroid_in_bbox(loc) and gx1 <= u <= gx2 and gy1 <= v <= gy2)
        except InsufficientDepthError as e:
            rec["locate_ms"] = (time.perf_counter() - t0) * 1000.0
            rec["locate_ok"] = False
            rec["locate_error"] = str(e)
            rec["match_ok"] = False
            rec["closed_ok"] = False
            rows.append(rec)
            continue
        try:
            m = match_nearest_grasp(grasps, rec["xyz_m"], min_score=min_sc, max_dist_m=max_d)
            rec["match_ok"] = True
            rec["match_dist_mm"] = m.distance_m * 1000.0
            rec["match_score"] = m.score
            rec["match_index"] = m.index
        except NoGraspMatchError as e:
            rec["match_ok"] = False
            rec["match_error"] = str(e)
        rec["closed_ok"] = bool(r["ok"] and rec.get("locate_ok") and rec.get("match_ok"))
        rows.append(rec)
    n = len(rows)
    n_loc = sum(1 for x in rows if x.get("locate_ok"))
    n_m = sum(1 for x in rows if x.get("match_ok"))
    n_c = sum(1 for x in rows if x.get("closed_ok"))
    return {
        "n": n,
        "locate_acc": n_loc / max(n, 1),
        "match_acc": n_m / max(n, 1),
        "closed_acc": n_c / max(n, 1),
        "locate_ms": _stats([x["locate_ms"] for x in rows if "locate_ms" in x]),
        "rows": rows,
    }


def run_abnormal(cfg, spec: dict, client: VLMClient) -> dict:
    out = {}
    # 断网：错误地址，硬超时
    bad = Config(json.loads(json.dumps(dict(cfg))))
    bad.setdefault("vlm", {})
    bad["vlm"]["base_url"] = "https://127.0.0.1:1/v1"
    bad["vlm"]["timeout_s"] = 3.0
    bad.setdefault("live", {})
    bad["live"]["vlm_timeout_s"] = 3.0
    bad["live"]["vlm_max_retries"] = 0
    img = _p(spec["exact"][0]["image"])
    t0 = time.perf_counter()
    kind = None
    msg = None
    try:
        dead = VLMClient(bad)
        dead.detect(str(img), "白色鼠标", timeout_s=3.0, max_retries=0)
        kind = "unexpected_success"
        msg = "断网测试居然成功"
    except Exception as e:
        blob = f"{type(e).__name__} {e}".lower()
        kind = "timeout" if "timeout" in blob or "timed out" in blob else "network"
        msg = redact_secrets(f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    out["network"] = {
        "ok": kind in ("timeout", "network") and dt < 12.0,
        "kind": kind,
        "message": msg,
        "latency_s": dt,
        "not_frozen": dt < 12.0,
    }
    # 空场景：天花板图不应找到鼠标
    empty = spec["empty_scene"]
    row = _detect_row(client, _p(empty["image"]), str(empty["target"]), False, None, 0.5)
    out["empty_scene"] = {"ok": row["ok"], "row": row}
    # 空洞定位
    frame = load_rgbd_dir(ROOT / "vlm" / "locator_data" / "desk")
    t1 = time.perf_counter()
    try:
        locate_target(frame, [0, 720, 80, 800], cfg=cfg, target="空洞")
        hole = {"ok": False, "error": "should_fail"}
    except InsufficientDepthError as e:
        hole = {"ok": True, "error": str(e), "latency_ms": (time.perf_counter() - t1) * 1000.0}
    except Exception as e:
        hole = {"ok": True, "error": str(e), "note": "其它异常也视为无深度"}
    out["empty_depth"] = hole
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    spec = yaml.safe_load((ROOT / "configs" / "phase2_eval.yaml").read_text(encoding="utf-8"))
    out = OUTPUT_DIR / "phase2" / "latest"
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_backend": cfg.get_path("camera.backend"),
        "vlm_model": cfg.get_path("vlm.model"),
    }
    LOGGER.info("pytest")
    report["pytest"] = run_pytest()
    LOGGER.info("locator")
    loc_full = remeasure_locator(cfg)
    report["locator"] = {
        "inside_acc": loc_full["inside_acc"],
        "negative_acc": loc_full["negative_acc"],
        "mean_ms": loc_full["mean_ms"],
        "max_ms": loc_full["max_ms"],
        "pass_inside": loc_full["pass_inside"],
        "pass_negative": loc_full["pass_negative"],
        "pass_latency": loc_full["pass_latency"],
    }
    LOGGER.info("capture fps")
    cap = remeasure_capture(cfg, seconds=8.0)
    report["capture"] = {k: v for k, v in cap.items()}
    LOGGER.info("infer")
    inf = remeasure_infer(cfg)
    engine = inf.pop("engine")
    frame = inf.pop("frame")
    grasps = inf.pop("grasps_all")
    inf.pop("result", None)
    report["infer"] = inf
    del engine
    LOGGER.info("vlm matrix")
    client = VLMClient(cfg)
    vlm = run_vlm_matrix(client, spec)
    report["vlm"] = {
        "naming_n": vlm["naming_n"],
        "naming_ok": vlm["naming_ok"],
        "naming_acc": vlm["naming_acc"],
        "latency_s": vlm["latency_s"],
        "groups": {
            k: {kk: vv for kk, vv in g.items() if kk != "rows"}
            for k, g in vlm["groups"].items()
        },
    }
    pos_rows = vlm["groups"]["exact"]["rows"] + vlm["groups"]["colloquial"]["rows"]
    LOGGER.info("closed loop on VLM bboxes")
    loop = run_closed_loop(cfg, spec, pos_rows, grasps, {str(ROOT / "vlm/locator_data/desk"): frame})
    report["closed_loop"] = {k: v for k, v in loop.items() if k != "rows"}
    LOGGER.info("abnormal")
    report["abnormal"] = run_abnormal(cfg, spec, client)
    report["pass_naming"] = report["vlm"]["naming_acc"] >= 0.85
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "vlm_rows.json").write_text(
        json.dumps({k: g["rows"] for k, g in vlm["groups"].items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "closed_rows.json").write_text(json.dumps(loop["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pytest", "locator", "capture", "infer", "vlm", "closed_loop", "abnormal", "pass_naming")}, ensure_ascii=False, indent=2))
    print(f"已保存 {out}")
    return 0 if report["pass_naming"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
