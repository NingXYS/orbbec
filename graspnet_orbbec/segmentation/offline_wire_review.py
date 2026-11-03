"""原线材窗口的离线复核会话：仅文件、原算法和人工多边形，不构造相机。"""
import copy,json,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from segmentation.wire import WireResult,save_result
from segmentation.grasp_policy import confirm_body_boundary
from segmentation.grasp_v2 import FrozenScene,apply_preview
from segmentation.terminal_annotation import create_annotation,save_annotation,load_annotation


def load_frozen_result(scene_path):
    metadata=json.loads(Path(scene_path).read_text());scene=FrozenScene.load(scene_path)
    recognition=json.loads(Path(metadata['inputs']['recognition']['path']).read_text())
    image=cv2.imread(str(scene.image_path));mask=cv2.imread(metadata['inputs']['body_mask']['path'],0)
    if image is None or mask is None or mask.shape!=image.shape[:2]:raise ValueError('FROZEN_MASK_OR_IMAGE_MISSING')
    from skimage.morphology import skeletonize
    mask=mask>0
    result=WireResult(image,mask,skeletonize(mask),np.asarray(recognition['centerline_xy']),
        copy.deepcopy(recognition['candidates']),recognition['sam_score'],recognition['elapsed_s'],recognition['crop_xyxy'],
        warnings=list(recognition['warnings']),target=recognition['target'],
        paths_xy=[np.asarray(p) for p in recognition['centerlines_by_component']],
        automation={**recognition['automation'],'offline_source':'FROZEN_RECOGNITION_NOT_NEW_INFERENCE'},
        terminal_detections=copy.deepcopy(recognition['terminal_detections']),
        grasp_policy=copy.deepcopy(recognition['grasp_policy']),grasp_plans=copy.deepcopy(recognition['grasp_plans']))
    return result,scene


class OfflineWireSession:
    def __init__(self,result,cfg,*,scene=None,output_dir,reviewer='local_operator',seconds=0,annotation_file=None,image_path=None):
        if cfg.get_path('wire.grasp_policy.offset_mm',14)!=14:raise ValueError('NEW_OFFLINE_SESSION_REQUIRES_TOTAL_14MM')
        self.result=result;self.cfg=cfg;self.scene=scene;self.output_dir=Path(output_dir)
        self.image_path=scene.image_path if scene is not None else (Path(image_path) if image_path is not None else None)
        self.reviewer=reviewer;self.seconds=seconds;self.annotation=None;self.polygons=[];self.pending=[];self.editing=False
        self.annotation_source='HUMAN_POLYGON_UI';self.message='离线冻结图：Shift+左键交界；T绘端子，Enter闭合，Y人工确认，S保存；Z放大'
        self.state=None;self.zoom=False;self.zoom_origin=(0,0);self.last_canvas=None
        apply_preview(self.result,self.cfg,self.scene)
        if annotation_file is not None:
            p=self.selected_plan()
            if self.image_path is None:raise ValueError('ANNOTATION_REQUIRES_BOUND_FROZEN_IMAGE')
            self.annotation=load_annotation(annotation_file,self.image_path,result.image_bgr,result.mask,
                candidate_id=p['candidate_id'],boundary_xy=p['boundary_input_xy'])
            self.polygons=copy.deepcopy(self.annotation['payload']['polygons_xy'])
            self.annotation_source=self.annotation['payload']['annotation_source'];self.recompute()

    def selected_plan(self):
        plans=[p for p in self.result.grasp_plans if p.get('boundary_confirmed')]
        if len(plans)!=1:raise ValueError('CONFIRM_EXACTLY_ONE_BOUNDARY_FIRST')
        return plans[0]

    def recompute(self):
        apply_preview(self.result,self.cfg,self.scene,self.annotation)
        if self.state is not None:self.state.result=self.result;self.state.message=self.message

    def set_boundary(self,point):
        confirm_body_boundary(self.result,point,self.cfg)
        # 新分界/实例会使旧确认绑定失效；保留可见轮廓供重新审阅，不继承确认。
        self.annotation=None
        self.message='原始交界已更新，14mm中心已重算；旧端子确认已失效'
        self.recompute()

    def make_annotation(self,human_confirmed=False,source=None):
        if self.image_path is None:raise ValueError('ANNOTATION_REQUIRES_BOUND_FROZEN_IMAGE')
        p=self.selected_plan();source=source or self.annotation_source
        payload=create_annotation(self.image_path,self.result.image_bgr,self.result.mask,self.polygons,
            candidate_id=p['candidate_id'],boundary_xy=p['boundary_input_xy'],source=source,
            reviewer=self.reviewer,human_confirmed=human_confirmed,relationship_confirmed=human_confirmed)
        self.annotation={'payload':payload,'annotation_file':None,'annotation_file_sha256':None}
        self.recompute();return payload

    def close_polygon(self):
        from segmentation.terminal_annotation import validate_polygon
        polygon=validate_polygon(self.pending,*self.result.image_bgr.shape[1::-1]);self.polygons.append(polygon)
        self.pending=[];self.editing=False;self.annotation_source='HUMAN_POLYGON_UI'
        self.make_annotation(False);self.message='端子轮廓已闭合，尚未人工确认；核对图像与绝缘关系后按Y'

    def confirm_polygon(self):
        self.make_annotation(True,'HUMAN_POLYGON_UI')
        self.message='已记录本地人工轮廓确认；仍只是离线模型几何，不是物理安全验收'

    def recognize(self):
        from segmentation.automatic import AutomaticWireRecognizer
        from common.config import Config
        previous=[p['boundary_input_xy'] for p in self.result.grasp_plans if p.get('boundary_confirmed')]
        cfg=Config(copy.deepcopy(self.cfg))
        roi=self.result.automation.get('roi_xyxy_original_rgb')
        if self.scene is not None and roi is not None:
            # 回放复用历史识别的搜索范围，不把全图棋盘误选结果当成原配置复现。
            cfg.setdefault('wire',{}).setdefault('auto',{})['roi_xyxy']=roi
        recognizer=AutomaticWireRecognizer(cfg);recognizer.prepare_terminal_runtime()
        result=recognizer.recognize(self.result.image_bgr)
        result.automation['replay_roi_source']='historical_original_roi' if self.scene is not None and roi is not None else 'current_config'
        from segmentation.grasp_policy import apply_grasp_policy
        apply_grasp_policy(result,self.cfg)
        self.result=result;self.annotation=None;self.polygons=[];self.pending=[]
        for point in previous:
            try:confirm_body_boundary(result,point,self.cfg)
            except ValueError:pass
        self.message='原自动分割算法已重跑；原图未变，端子轮廓需重新标注/确认'
        self.recompute()

    def export(self):
        label=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        if self.annotation is not None and self.annotation.get('annotation_file') is None:
            self.annotation=save_annotation(self.annotation['payload'],self.output_dir/'annotations'/f'{label}.json')
            self.recompute()
        out=save_result(self.result,self.output_dir/'reviews'/label)
        if self.last_canvas is not None:cv2.imwrite(str(out/'preview_canvas.png'),self.last_canvas)
        return out

    def _click(self,event,point,flags):
        try:
            if self.editing:
                if event==cv2.EVENT_LBUTTONDOWN:
                    self.pending.append(list(map(float,point)));self.annotation=None;self.recompute()
                elif event==cv2.EVENT_RBUTTONDOWN and self.pending:self.pending.pop()
            elif event==cv2.EVENT_LBUTTONDOWN and flags&cv2.EVENT_FLAG_SHIFTKEY:self.set_boundary(point)
        except (ValueError,OSError) as e:self.message=str(e)

    def run_gui(self,render_canvas):
        from segmentation.wire_gui import PromptState,panel_point
        self.state=PromptState(source=self.result.image_bgr.copy(),result=self.result,message=self.message,offline_review=True)
        identity=self.scene.image_path.parents[1].name if self.scene else 'image'
        name='SAM2 wire recognition - OFFLINE - '+identity;zoom_name='Terminal polygon zoom - '+identity
        image=self.result.image_bgr;h,w=image.shape[:2];start=time.monotonic()
        def mouse(event,x,y,flags,param):
            p=panel_point(x,y,w,h)
            if p is not None:self._click(event,p,flags)
        def zoom_mouse(event,x,y,flags,param):
            p=(self.zoom_origin[0]+x/4,self.zoom_origin[1]+y/4)
            if 0<=p[0]<w and 0<=p[1]<h:self._click(event,p,flags)
        cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(name,mouse)
        try:
            while True:
                self.state.result=self.result;self.state.message=self.message
                canvas=render_canvas(image,self.state)
                for poly in self.polygons:
                    pts=np.rint(poly).astype(np.int32)+[0,64];cv2.polylines(canvas,[pts],True,(0,140,255),1)
                if self.pending:
                    pts=np.rint(self.pending).astype(np.int32)+[0,64];cv2.polylines(canvas,[pts],False,(0,140,255),1)
                    for p in pts:cv2.circle(canvas,tuple(p),2,(0,140,255),-1)
                from vlm.target_locator import put_cjk
                put_cjk(canvas,'离线：Shift交界 | T端子多边形 / 右键撤点 / Enter闭合 | Y人工确认 | Z放大 | S保存 | R重识别 | Q退出',
                        (10,h+92),size=15)
                self.last_canvas=canvas;cv2.imshow(name,canvas)
                if self.zoom:
                    try:p=np.array(self.selected_plan()['boundary_input_xy'])
                    except ValueError:p=np.array([w/2,h/2])
                    x0=max(0,min(w-160,int(p[0])-80));y0=max(0,min(h-120,int(p[1])-60));self.zoom_origin=(x0,y0)
                    crop=canvas[64+y0:64+y0+120,x0:x0+160]
                    cv2.imshow(zoom_name,cv2.resize(crop,(640,480),interpolation=cv2.INTER_NEAREST))
                key=cv2.waitKey(15)&0xff
                if key in (27,ord('q')) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
                try:
                    if key==ord('t'):self.editing=not self.editing;self.message='在原图或Z放大窗点选端子轮廓；Enter闭合，右键撤点'
                    elif key in (10,13):self.close_polygon()
                    elif key==ord('y'):self.confirm_polygon()
                    elif key==ord('s'):self.message='已保存：'+str(self.export())
                    elif key==ord('r'):self.recognize()
                    elif key==ord('z'):
                        self.zoom=not self.zoom
                        if self.zoom:cv2.namedWindow(zoom_name,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(zoom_name,zoom_mouse)
                        else:cv2.destroyWindow(zoom_name)
                    elif key==ord('x'):
                        self.polygons=[];self.pending=[];self.annotation=None;self.recompute()
                except (ValueError,OSError) as e:self.message=str(e)
                if self.seconds>0 and time.monotonic()-start>=self.seconds:break
        finally:
            cv2.destroyWindow(name)
            if self.zoom:cv2.destroyWindow(zoom_name)
        return 0
