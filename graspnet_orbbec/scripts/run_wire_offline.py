#!/usr/bin/env python3
"""Phase1.8离线视觉主入口；文件回放/原分割/标注/预览，没有相机和执行分支。"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common.config import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--snapshot',choices=['enabled','raised'])
    source.add_argument('--scene',type=Path,help='已绑定文件哈希的离线scene JSON')
    source.add_argument('--image',type=Path,help='普通图片必须加--recognize；无标定时不输出毫米包络')
    p.add_argument('--config');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recognize',action='store_true',help='实际重跑原AutomaticWireRecognizer，保持其分割算法')
    p.add_argument('--boundary-point',nargs=2,type=float,help='当前冻结图中人工指定的原始交界，不是旧抓取中心')
    p.add_argument('--annotation',type=Path,help='加载已保存并有.sha256侧文件的端子标注')
    p.add_argument('--draft-polygons',type=Path,help='导入未人工确认的多边形草稿，不能算避让通过')
    p.add_argument('--gui',action='store_true');p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--reviewer',default='local_operator');p.add_argument('--compare-10mm',action='store_true',help='只额外导出同锚点旧策略比较，不改变新策略')
    args=p.parse_args();cfg=load_config(args.config)
    if cfg.get_path('wire.grasp_policy.offset_mm',14)!=14:p.error('新离线预览只接受14mm配置；旧10mm仅用--compare-10mm比较')
    if args.out.exists():raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    if args.annotation and args.draft_polygons:p.error('正式标注与未确认草稿不能同时导入')
    from segmentation.offline_wire_review import load_frozen_result,OfflineWireSession
    from segmentation.grasp_policy import apply_grasp_policy,confirm_body_boundary
    if args.snapshot:
        index=json.loads((ROOT/'configs/wire_offline_snapshots.json').read_text());args.scene=Path(index[args.snapshot])
    if args.scene:
        result,scene=load_frozen_result(args.scene)
    else:
        if not args.recognize:p.error('--image需要--recognize，不会自动启动相机')
        import cv2
        from segmentation.automatic import AutomaticWireRecognizer
        image=cv2.imread(str(args.image))
        if image is None:raise ValueError('IMAGE_NOT_READABLE')
        recognizer=AutomaticWireRecognizer(cfg);recognizer.prepare_terminal_runtime();result=recognizer.recognize(image);scene=None
        apply_grasp_policy(result,cfg)
    session=OfflineWireSession(result,cfg,scene=scene,output_dir=args.out,reviewer=args.reviewer,seconds=args.seconds,image_path=args.image)
    if args.recognize and args.scene:session.recognize()
    if args.boundary_point:session.set_boundary(args.boundary_point)
    if args.annotation:
        from segmentation.terminal_annotation import load_annotation
        plan=session.selected_plan()
        if session.image_path is None:raise ValueError('ANNOTATION_REQUIRES_BOUND_FROZEN_IMAGE')
        session.annotation=load_annotation(args.annotation,session.image_path,session.result.image_bgr,session.result.mask,
            candidate_id=plan['candidate_id'],boundary_xy=plan['boundary_input_xy'])
        session.polygons=session.annotation['payload']['polygons_xy'];session.recompute()
    if args.draft_polygons:
        session.polygons=json.loads(args.draft_polygons.read_text())['polygons_xy']
        session.annotation_source='ASSISTANT_VISUAL_DRAFT';session.make_annotation(False)
    args.out.mkdir(parents=True,exist_ok=False)
    if args.gui:
        from segmentation.wire_gui import run
        run(cfg,SimpleNamespace(),SimpleNamespace(stop_flag=False),offline_session=session)
    else:
        from segmentation.wire_gui import PromptState,render_canvas
        session.last_canvas=render_canvas(session.result.image_bgr,PromptState(source=session.result.image_bgr,result=session.result,offline_review=True,message=session.message))
    exported=session.export()
    if args.compare_10mm:
        import copy
        from common.config import Config
        from segmentation.grasp_v2 import apply_preview
        old=copy.deepcopy(session.result);legacy=Config({'wire':{'grasp_policy':{'offset_mm':10}}})
        apply_preview(old,legacy,scene)
        (args.out/'comparison_10mm.json').write_text(json.dumps(old.to_dict(),ensure_ascii=False,indent=2))
    summary={'status':'OFFLINE_PREVIEW_COMPLETE','output':str(exported.resolve()),
        'results':[{'candidate_id':q['candidate_id'],'validity':q.get('validity'),'reasons':q.get('rejection_reasons'),'offset_mm':q['offset_mm']} for q in session.result.grasp_plans],
        'hardware_connections':0,'motion_actions':0,'phase2':'BLOCKED'}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError) as e:print('离线复核拒绝：'+str(e),file=sys.stderr);sys.exit(2)
