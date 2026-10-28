"""真实长度与像素分开，交界按快照确认，弯线按弧长计算。"""

from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import Config
from segmentation.grasp_policy import plan_candidate, apply_grasp_policy, confirm_boundary
from segmentation.wire import analyze_mask, draw_result, save_result


def straight():
    pixels = np.c_[20+np.arange(40)*5, np.full(40,50)]
    xyz = np.c_[np.arange(40)*.001, np.zeros(40), np.full(40,.5)]
    return {'id':'A','inward_path_xy':pixels.tolist(),'path_ambiguous':False}, xyz


def test_default_requires_boundary_and_has_no_commanded_pose():
    candidate, xyz = straight()
    plan = plan_candidate(candidate,None,path_xyz_m=xyz,plane_normal=[0,0,1])
    assert plan['status']=='boundary_required'
    assert plan['grasp_camera_m'] is None
    assert not plan['robot_pose_valid'] and not plan['tcp_compensation_applied']


def test_confirmed_boundary_does_not_turn_pixels_into_millimeters():
    candidate, _ = straight()
    plan = plan_candidate(candidate,[45,50])
    assert plan['status']=='metric_path_required'
    assert plan['boundary_confirmed']
    assert plan['grasp_xy_preview'] is None and plan['grasp_camera_m'] is None


def test_10mm_is_measured_from_boundary_and_not_a_fixed_pixel_inset():
    candidate, xyz = straight()
    normal = np.array([0.,0.,2.])
    plan=plan_candidate(candidate,[45,50],path_xyz_m=xyz,plane_normal=normal)
    assert plan['status']=='camera_frame_preview_ready'
    assert plan['grasp_camera_m']==pytest.approx([.015,0,.5])
    assert plan['grasp_xy_preview']==pytest.approx([95,50])
    assert plan['closing_direction_camera']==pytest.approx([0,1,0])
    assert np.dot(plan['closing_direction_camera'],plan['wire_tangent_camera'])==pytest.approx(0)
    assert np.array_equal(normal,[0,0,2])
    assert not plan['robot_pose_valid']


def test_curved_wire_uses_3d_arc_length_instead_of_straight_displacement():
    candidate={'id':'A','inward_path_xy':[[10,10],[70,10],[70,70]]}
    xyz=[[0,0,.5],[.006,0,.5],[.006,.006,.5]]
    plan=plan_candidate(candidate,[10,10],path_xyz_m=xyz,plane_normal=[0,0,1])
    assert plan['grasp_camera_m']==pytest.approx([.006,.004,.5])
    assert plan['grasp_xy_preview']==pytest.approx([70,50])
    assert plan['closing_direction_camera']==pytest.approx([-1,0,0])


def test_no_support_plane_means_orientation_pending():
    candidate, xyz=straight()
    plan=plan_candidate(candidate,[45,50],path_xyz_m=xyz)
    assert plan['status']=='support_plane_required'
    assert plan['grasp_camera_m'] is not None and plan['closing_direction_camera'] is None


def test_short_wire_and_ambiguous_instance_do_not_extrapolate():
    candidate, xyz=straight()
    plan=plan_candidate(candidate,[45,50],path_xyz_m=xyz,offset_mm=100)
    assert plan['status']=='insufficient_wire_length' and plan['grasp_camera_m'] is None
    candidate['path_ambiguous']=True
    plan=plan_candidate(candidate,[45,50],path_xyz_m=xyz)
    assert plan['status']=='wire_instance_ambiguous' and plan['grasp_camera_m'] is None


@pytest.mark.parametrize('offset',[-1,0,np.nan,np.inf])
def test_invalid_offset_is_rejected(offset):
    candidate,_=straight()
    with pytest.raises(ValueError):
        plan_candidate(candidate,None,offset_mm=offset)


def test_metric_path_is_strictly_aligned_valid_camera_data():
    candidate,xyz=straight()
    for invalid in (xyz[:-1],np.full_like(xyz,np.nan),np.zeros_like(xyz)):
        with pytest.raises(ValueError):
            plan_candidate(candidate,[45,50],path_xyz_m=invalid)
    with pytest.raises(ValueError):
        plan_candidate(candidate,[45,50],path_xyz_m=xyz,plane_normal=[0,0,0])


def test_wire_parallel_to_plane_normal_does_not_invent_closing_axis():
    candidate,xyz=straight()
    plan=plan_candidate(candidate,[45,50],path_xyz_m=xyz,plane_normal=[1,0,0])
    assert plan['status']=='wire_direction_inconsistent_with_plane'
    assert plan['closing_direction_camera'] is None


def result_fixture():
    image=np.full((120,240,3),150,np.uint8)
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.line(mask,(20,60),(220,60),1,3)
    return analyze_mask(image,mask)


def test_confirm_boundary_and_save_are_per_result_without_metric_guess(tmp_path):
    result=result_fixture()
    cfg=Config({'wire':{'grasp_policy':{'offset_mm':10}}})
    apply_grasp_policy(result,cfg)
    assert all(p['status']=='boundary_required' for p in result.grasp_plans)
    candidate=result.candidates[0]
    point=candidate['inward_path_xy'][8]
    assert confirm_boundary(result,point,cfg)==candidate['id']
    plan=next(p for p in result.grasp_plans if p['candidate_id']==candidate['id'])
    assert plan['status']=='metric_path_required' and plan['offset_mm']==10
    save_result(result,tmp_path/'saved')
    saved=json.loads((tmp_path/'saved/recognition.json').read_text())
    assert saved['grasp_policy']['offset_mm']==10
    assert saved['grasp_plans'][0]['boundary_confirmed']
    assert not saved['robot_pose_valid']
    with pytest.raises(ValueError):
        confirm_boundary(result,[1000,1000],cfg)
    assert result.grasp_plans[0]['boundary_confirmed']
    fresh=result_fixture()
    apply_grasp_policy(fresh,cfg)
    assert not any(p['boundary_confirmed'] for p in fresh.grasp_plans)


def test_unknown_boundary_id_rejected_and_configurable_offset_applied():
    result=result_fixture()
    cfg=Config({'wire':{'grasp_policy':{'offset_mm':15}}})
    with pytest.raises(ValueError):
        apply_grasp_policy(result,cfg,{'absent':[10,10]})
    apply_grasp_policy(result,cfg)
    assert all(p['offset_mm']==15 for p in result.grasp_plans)


def test_policy_marks_boundary_without_drawing_unscaled_grip_center():
    result=result_fixture()
    cfg=Config()
    apply_grasp_policy(result,cfg)
    confirm_boundary(result,result.candidates[0]['inward_path_xy'][8],cfg)
    vis=draw_result(result)
    assert vis.shape==result.image_bgr.shape
    assert not np.any(np.all(vis==[0,255,0],axis=2))


def test_shift_click_confirms_boundary_without_restarting_segmentation(monkeypatch):
    from types import SimpleNamespace as NS
    import time
    import segmentation.wire_gui as gui
    cfg=Config({'wire':{'auto':{'warmup_s':0}}})
    original=result_fixture()
    clock=[0.]
    calls=[]
    callbacks=[]
    shown=[]
    class Camera:
        def open(self): pass
        def close(self): pass
        def read(self):
            clock[0]+=1
            return NS(color_bgr=original.image_bgr)
    class Recognizer:
        def recognize(self,image):
            calls.append(True)
            return original
    worker=gui.SegmentationWorker(cfg,auto_recognizer=Recognizer())
    monkeypatch.setattr(gui,'open_camera',lambda cfg:Camera())
    monkeypatch.setattr(gui,'SegmentationWorker',lambda cfg:worker)
    monkeypatch.setattr(gui,'time',NS(monotonic=lambda:clock[0]))
    for name in ('namedWindow','imshow','destroyAllWindows'):
        monkeypatch.setattr(gui.cv2,name,lambda *a,**kw:None)
    monkeypatch.setattr(gui.cv2,'setMouseCallback',lambda name,callback:callbacks.append(callback))
    monkeypatch.setattr(gui.cv2,'getWindowProperty',lambda *a:1)
    def render(live,state):
        if state.result is not None:
            shown.append(any(p['boundary_confirmed'] for p in state.result.grasp_plans))
        return live
    monkeypatch.setattr(gui,'render_canvas',render)
    keys=[0]
    def key(delay):
        keys[0]+=1
        if keys[0]==1:
            deadline=time.monotonic()+3
            while worker.results.empty() and time.monotonic()<deadline:
                time.sleep(.005)
            assert not worker.results.empty()
        if keys[0]==2:
            callbacks[0](cv2.EVENT_LBUTTONDOWN,35,124,cv2.EVENT_FLAG_SHIFTKEY,None)
            callbacks[0](cv2.EVENT_LBUTTONUP,35,124,cv2.EVENT_FLAG_SHIFTKEY,None)
        return ord('q') if keys[0]>=3 else -1
    monkeypatch.setattr(gui.cv2,'waitKey',key)
    args=NS(from_dir=None,no_target=True,wire_auto=True,wire_box=None,wire_point=[],wire_seconds=0,wire_preview=None)
    assert gui.run(cfg,args,NS(stop_flag=False))==0
    assert calls==[True] and shown==[False,True]
    plan=next(p for p in original.grasp_plans if p['boundary_confirmed'])
    assert plan['offset_mm']==10 and plan['status']=='metric_path_required'
