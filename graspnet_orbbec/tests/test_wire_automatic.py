"""自动识别的几何筛选、端子证据和无手工提示调度。"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import Config
from segmentation.automatic import AutomaticWireRecognizer, attach_terminals, select_wire_masks, automatic_line_seeds, prune_continuing_endpoints
from segmentation.wire import SegmentationError, analyze_mask, save_result
from segmentation.wire_gui import PromptState, SegmentationWorker


def scene():
    image = np.full((480, 640, 3), 128, np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    cv2.polylines(mask, [np.array([[20, 180], [150, 100], [300, 180], [350, 240]])], False, 1, 3)
    return image, mask.astype(bool)


def record(mask, score=.8):
    return {'segmentation': mask, 'predicted_iou': score}


def test_auto_filters_desk_thick_cables_and_duplicate_masks():
    image, thin = scene()
    thick = np.zeros_like(thin, np.uint8)
    cv2.line(thick, (400, 60), (550, 350), 1, 14)
    selected = select_wire_masks(image, [record(np.ones_like(thin)), record(thick),
                                        record(thin, .7), record(thin, .9)],
                                 Config({'wire': {'auto': {'max_width_fraction': .014}}}))
    assert len(selected) == 1
    assert np.array_equal(selected[0][0], thin)
    assert selected[0][1]['sam_score'] == .9


def test_auto_does_not_claim_ends_for_cut_wire_or_closed_loop():
    image, _ = scene()
    cut = np.zeros(image.shape[:2], np.uint8)
    cv2.line(cut, (0, 160), (639, 160), 1, 3)
    loop = np.zeros_like(cut)
    cv2.circle(loop, (300, 200), 80, 1, 3)
    selected = select_wire_masks(image, [record(cut), record(loop)], Config())
    assert len(selected) == 2  # 可见线身应保留，但不能凭截断或闭环编造端子。
    assert all(not analyze_mask(image, mask).candidates for mask, _ in selected)


def test_empty_scene_low_score_and_wrong_size_do_not_fabricate_wire():
    image, thin = scene()
    assert not select_wire_masks(image, [record(np.zeros_like(thin)), record(thin, .1),
                                        record(np.zeros((10, 10))), record(thin, np.nan)], Config())
    recognizer = AutomaticWireRecognizer(Config({'wire': {'auto': {'terminal_enabled': False}}}))
    recognizer._generate = lambda image: []
    result = recognizer.recognize(image)
    assert not result.mask.any() and not result.candidates
    assert any('未分出线身' in w for w in result.warnings)


def test_terminal_detection_survives_when_no_wire_mask_is_found():
    image, wire = scene()
    recognizer = AutomaticWireRecognizer(Config())
    recognizer._generate = lambda image: []
    terminal = np.zeros_like(wire)
    terminal[235:245, 345:355] = True
    detection = {'id': 'T1', 'confidence': .8, 'bbox_xyxy': [345,235,355,245], 'mask_file': 'terminal_T1.png'}
    recognizer._detect_terminals = lambda image: ([detection], [terminal])
    result = recognizer.recognize(image)
    assert not result.mask.any()
    assert len(result.terminal_detections) == 1
    assert np.array_equal(result.terminal_masks[0], terminal)


def test_sparse_seeds_cover_multiple_dark_and_bright_wires_without_grid():
    image = np.full((480, 640, 3), 128, np.uint8)
    for y, color in ((80,20), (180,240), (280,30), (380,230)):
        cv2.line(image, (50,y), (570,y), (color,)*3, 3)
    points, capped = automatic_line_seeds(image, Config())
    assert len(points) <= 64
    for y in (80,180,280,380):
        assert np.any(abs(points[:,1]-y) <= 3)
    assert not len(automatic_line_seeds(np.full_like(image,128), Config())[0])


def test_automatic_seed_budget_is_reported_and_respected():
    image, mask = scene()
    image[mask] = 10
    points, capped = automatic_line_seeds(image, Config({'wire': {'auto': {'max_seeds': 2}}}))
    assert len(points) == 2 and capped


def test_mask_break_is_not_a_true_wire_tip_if_image_line_continues():
    image=np.full((100,200,3),160,np.uint8)
    cv2.line(image,(20,50),(180,50),(20,20,20),3)
    mask=np.zeros((100,200),np.uint8)
    cv2.line(mask,(60,50),(160,50),1,3)
    result=analyze_mask(image,mask)
    assert len(result.candidates)==2
    prune_continuing_endpoints(result,Config())
    assert not result.candidates
    assert len(result.automation['rejected_geometric_ends'])==2


def test_actual_visible_end_is_not_pruned():
    image=np.full((100,200,3),160,np.uint8)
    mask=np.zeros((100,200),np.uint8)
    cv2.line(mask,(20,50),(180,50),1,3)
    image[mask>0]=20
    result=analyze_mask(image,mask)
    prune_continuing_endpoints(result,Config())
    assert len(result.candidates)==2


def test_terminal_evidence_only_attaches_to_overlapping_end():
    image, mask = scene()
    result = analyze_mask(image, mask)
    near = result.end_regions[0].copy()
    detection = {'id': 'T1', 'confidence': .88, 'bbox_xyxy': [10, 100, 100, 200],
                 'mask_file': 'terminal_T1.png'}
    attach_terminals(result, [detection], [near])
    assert result.candidates[0]['terminal_status'] == 'model_detected'
    assert result.candidates[1]['terminal_status'] == 'geometry_only'
    assert not result.candidates[1]['terminal_model_evidence']
    assert all(not c['confirmed'] for c in result.candidates)
    assert not result.to_dict()['terminal_semantics_verified']


def test_auto_pipeline_needs_no_box_points_or_network(tmp_path):
    image, mask = scene()
    recognizer = AutomaticWireRecognizer(Config())
    recognizer._generate = lambda actual: [record(mask)]
    terminal_mask = analyze_mask(image, mask).end_regions[0]
    detection = {'id': 'T1', 'confidence': .9, 'bbox_xyxy': [10, 100, 100, 200],
                 'mask_file': 'terminal_T1.png'}
    recognizer._detect_terminals = lambda actual: ([detection], [terminal_mask])
    result = recognizer.recognize(image)
    assert result.automation['manual_prompts'] is False
    assert result.automation['external_api_calls'] == 0
    assert result.automation['terminal_status'] == 'ok'
    assert len(result.terminal_masks) == 1
    save_result(result, tmp_path/'auto')
    assert np.array_equal(cv2.imread(str(tmp_path/'auto/terminal_T1.png'), 0)>0, terminal_mask)


def test_missing_terminal_weights_is_explicit_without_losing_wire():
    image, mask = scene()
    cfg = Config({'wire': {'auto': {'terminal_weights': '/no_such_wire_test_model.pt'}}})
    recognizer = AutomaticWireRecognizer(cfg)
    recognizer._generate = lambda actual: [record(mask)]
    result = recognizer.recognize(image)
    assert result.automation['terminal_status'] == 'unavailable'
    assert result.mask.any() and not result.terminal_detections
    assert all(c['terminal_status'] == 'geometry_only' for c in result.candidates)
    assert any('端子模型不可用' in w for w in result.warnings)


def test_empty_terminal_detection_stays_geometry_only():
    image, mask = scene()
    recognizer = AutomaticWireRecognizer(Config())
    recognizer._generate = lambda actual: [record(mask)]
    recognizer._detect_terminals = lambda actual: ([], [])
    result = recognizer.recognize(image)
    assert all(c['terminal_status'] == 'geometry_only' for c in result.candidates)
    assert any('未检出' in w for w in result.warnings)


def test_disabled_terminal_model_is_not_loaded():
    image, mask = scene()
    recognizer = AutomaticWireRecognizer(Config({'wire': {'auto': {'terminal_enabled': False}}}))
    recognizer._generate = lambda actual: [record(mask)]
    recognizer._detect_terminals = lambda actual: pytest.fail('不应调用端子模型')
    result = recognizer.recognize(image)
    assert result.automation['terminal_status'] == 'disabled'
    assert all(c['terminal_status'] == 'geometry_only' for c in result.candidates)


def test_first_terminal_runtime_import_is_rejected_in_worker_thread():
    import threading
    errors=[]
    recognizer=AutomaticWireRecognizer(Config())
    def attempt():
        try:
            recognizer.prepare_terminal_runtime()
        except RuntimeError as exc:
            errors.append(str(exc))
    thread=threading.Thread(target=attempt)
    thread.start();thread.join(2)
    assert errors and '主线程' in errors[0]
    assert recognizer.terminal_model_class is None


def test_prepared_runtime_does_not_reimport_from_worker():
    import threading
    recognizer=AutomaticWireRecognizer(Config())
    recognizer.terminal_model_class=object()
    finished=[]
    thread=threading.Thread(target=lambda:(recognizer.prepare_terminal_runtime(),finished.append(True)))
    thread.start();thread.join(2)
    assert finished==[True]


def test_worker_automatic_mode_uses_snapshot_without_vlm_or_manual_segment():
    image, mask = scene()

    class Auto:
        def recognize(self, actual):
            assert np.all(actual == 128)
            return analyze_mask(actual, mask)

    class Manual:
        def segment(self, *a, **kw):
            pytest.fail('自动路径不应调用手工提示分割')

    worker = SegmentationWorker(Config(), segmenter=Manual(), auto_recognizer=Auto())
    worker.submit(4, image, None, [], [], automatic=True)
    image[:] = 0
    worker.start()
    try:
        revision, result, error = worker.results.get(timeout=3)
        assert revision == 4 and result is not None and error is None
        state = PromptState(revision=5)
        assert not state.accept(revision, result, error)
    finally:
        worker.stop()


def test_auto_cli_implies_wire_and_rejects_manual_prompts(monkeypatch):
    from scripts.run_live_gui import parse_args
    monkeypatch.setattr(sys, 'argv', ['run_live_gui.py', '--wire-auto', '--no-target'])
    args = parse_args()
    assert args.wire and args.wire_auto and args.no_target
    monkeypatch.setattr(sys, 'argv', ['run_live_gui.py', '--wire-auto', '--wire-box', '1', '2', '30', '40'])
    with pytest.raises(SystemExit):
        parse_args()


def test_gui_auto_start_and_r_take_two_fresh_snapshots(monkeypatch):
    from types import SimpleNamespace as NS
    import time
    import segmentation.wire_gui as gui

    image, mask = scene()
    samples, displayed = [], []
    lifecycle=[]
    clock = [0.0]

    class Camera:
        index = 0
        closed = False

        def open(self):
            assert lifecycle==['prepare']
            lifecycle.append('camera')

        def read(self):
            self.index += 1
            clock[0] += 1
            return NS(color_bgr=np.full_like(image, self.index))

        def close(self): self.closed = True

    class Auto:
        def prepare_terminal_runtime(self):
            import threading
            assert threading.current_thread() is threading.main_thread()
            assert not worker.is_alive()
            lifecycle.append('prepare')

        def recognize(self, frame):
            samples.append(int(frame[0, 0, 0]))
            return analyze_mask(frame, mask)

    camera = Camera()
    worker = SegmentationWorker(Config(), auto_recognizer=Auto())
    monkeypatch.setattr(gui, 'open_camera', lambda cfg: camera)
    monkeypatch.setattr(gui, 'SegmentationWorker', lambda cfg: worker)
    monkeypatch.setattr(gui, 'time', NS(monotonic=lambda: clock[0]))
    for name in ('namedWindow', 'setMouseCallback', 'imshow', 'destroyAllWindows'):
        monkeypatch.setattr(gui.cv2, name, lambda *a, **kw: None)
    monkeypatch.setattr(gui.cv2, 'getWindowProperty', lambda *a: 1)

    def render(live, state):
        if state.result is not None:
            displayed.append((int(state.source[0, 0, 0]), int(state.result.image_bgr[0, 0, 0])))
        return live

    monkeypatch.setattr(gui, 'render_canvas', render)
    calls = [0]

    def key(delay):
        calls[0] += 1
        if calls[0] in (1, 3):
            deadline = time.monotonic() + 3
            while worker.results.empty() and time.monotonic() < deadline:
                time.sleep(.005)
            assert not worker.results.empty()
            return -1
        return ord('r') if calls[0] == 2 else ord('q')

    monkeypatch.setattr(gui.cv2, 'waitKey', key)
    args = NS(from_dir=None, no_target=True, wire_auto=True, wire_box=None, wire_point=[],
              wire_seconds=0, wire_preview=None)
    channel = NS(stop_flag=False)
    assert gui.run(Config({'wire': {'auto': {'warmup_s': 0}}}), args, channel) == 0
    assert samples == [1, 3]
    assert displayed == [(1, 1), (3, 3)]
    assert camera.closed and channel.stop_flag and not worker.is_alive()


def test_gui_survives_camera_timeouts_and_does_not_auto_process_stale_frame(monkeypatch):
    from types import SimpleNamespace as NS
    import time
    import segmentation.wire_gui as gui
    image, mask = scene()
    clock = [0.0]
    samples, views = [], []

    class Camera:
        index = 0
        closed = False
        def open(self): pass
        def read(self):
            self.index += 1
            clock[0] += 1
            if self.index in (1,3,4):
                raise TimeoutError('test color timeout')
            return NS(color_bgr=np.full_like(image,self.index))
        def close(self): self.closed = True

    class Auto:
        def recognize(self, actual):
            samples.append(int(actual[0,0,0]))
            return analyze_mask(actual, mask)

    camera = Camera()
    worker = SegmentationWorker(Config(), auto_recognizer=Auto())
    monkeypatch.setattr(gui,'open_camera',lambda cfg:camera)
    monkeypatch.setattr(gui,'SegmentationWorker',lambda cfg:worker)
    monkeypatch.setattr(gui,'time',NS(monotonic=lambda:clock[0]))
    for name in ('namedWindow','setMouseCallback','imshow','destroyAllWindows'):
        monkeypatch.setattr(gui.cv2,name,lambda *a,**kw:None)
    monkeypatch.setattr(gui.cv2,'getWindowProperty',lambda *a:1)
    def render(live,state):
        views.append(bool(state.camera_error))
        return live
    monkeypatch.setattr(gui,'render_canvas',render)
    def key(delay):
        if camera.index in (2,5):
            deadline=time.monotonic()+3
            while worker.results.empty() and time.monotonic()<deadline:
                time.sleep(.005)
            assert not worker.results.empty()
        return ord('r') if camera.index==3 else (ord('q') if camera.index>=6 else -1)
    monkeypatch.setattr(gui.cv2,'waitKey',key)
    args=NS(from_dir=None,no_target=True,wire_auto=True,wire_box=None,wire_point=[],wire_seconds=0,wire_preview=None)
    assert gui.run(Config({'wire':{'auto':{'warmup_s':0}}}),args,NS(stop_flag=False))==0
    assert samples==[2,5]
    assert views==[True,False,True,True,False,False]
    assert camera.closed and not worker.is_alive()
