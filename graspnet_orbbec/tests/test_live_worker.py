"""退出时须等待进行中的原生推理，避免解释器销毁与工作线程竞态。"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_live_gui import InferWorker, LiveCloudView


def test_stop_waits_for_inflight_inference():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class Engine:
        def infer_frame(self, frame):
            entered.set()
            assert release.wait(2)
            return {"frame": frame}

    worker = InferWorker(Engine())
    worker.start()
    worker.submit("test-frame")
    try:
        assert entered.wait(2)
        stopper = threading.Thread(target=lambda: (worker.stop(), finished.set()))
        stopper.start()
        assert not finished.wait(0.03)
    finally:
        release.set()
        worker.stop()
    stopper.join(2)
    assert finished.is_set()
    assert not worker.is_alive()
    assert worker.result == {"frame": "test-frame"}


def test_3d_geometry_is_not_rebuilt_for_repeated_results():
    from types import SimpleNamespace as NS
    import numpy as np

    view = LiveCloudView(enabled=False)
    calls = {"add": 0, "update": 0, "meshes": 0}
    def count(name):
        calls[name] += 1
    view.enabled = True
    view.vis = NS(add_geometry=lambda *a, **kw: count("add"),
                  update_geometry=lambda *a: count("update"), reset_view_point=lambda *a: None,
                  poll_events=lambda: True, update_renderer=lambda: None)
    view.pcd = NS()
    cloud = NS(points=np.ones((10, 3)), colors=np.ones((10, 3)))

    class Grasps:
        def __len__(self): return 1
        def __getitem__(self, item): return self
        def to_open3d_geometry_list(self):
            count("meshes")
            return [object()]

    grasps = Grasps()
    assert view.update(None, None)
    assert calls["add"] == 0
    assert view.update(cloud, grasps)
    assert view.update(cloud, grasps)
    assert calls == {"add": 2, "update": 0, "meshes": 1}
