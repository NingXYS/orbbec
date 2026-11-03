"""原视觉窗口中的离线包络/标注显示；坐标均为原图像素，不按固定像素比例造毫米。"""
import cv2
import numpy as np


def draw_review(image,result):
    out=image.copy()
    for record in result.manual_terminal_annotations:
        p=record['payload']
        for polygon in p['polygons_xy']:
            points=np.rint(polygon).astype(np.int32)
            cv2.polylines(out,[points],True,(0,140,255),1)
            label='HUMAN' if p['human_confirmed'] else 'UNCONFIRMED'
            cv2.putText(out,label,tuple(points[0]+[3,13]),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,140,255),1)
        from segmentation.terminal_annotation import avoidance_geometry
        for polygon in avoidance_geometry(p)['polygons_xy']:
            cv2.polylines(out,[np.rint(polygon).astype(np.int32)],True,(0,100,255),2)
    for p in result.grasp_plans:
        if p.get('grasp_xy_preview') is None:continue
        center=tuple(np.rint(p['grasp_xy_preview']).astype(int))
        if p.get('boundary_input_xy'):
            origin=tuple(np.rint(p['boundary_input_xy']).astype(int))
            cv2.arrowedLine(out,origin,center,(0,255,255),1,tipLength=.25)
        for key,color in [('full_gripper_footprint',(30,30,255)),('contact_footprint',(255,0,255))]:
            vertices=p.get(key,{}).get('projected_polygon_xy')
            if vertices:cv2.polylines(out,[np.rint(vertices).astype(np.int32)],True,color,1)
        cv2.drawMarker(out,center,(0,255,0),cv2.MARKER_CROSS,14,2)
        cv2.putText(out,f"{p['offset_mm']:g}mm TOTAL",tuple(np.array(center)+[8,17]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,230,255),1)
    return out


def review_lines(result):
    lines=[]
    for p in result.grasp_plans:
        if p.get('status')!='offline_grasp_preview':continue
        full=p['full_gripper_footprint'];contact=p['contact_footprint']
        def number(value):return '未知' if value is None else f'{value*1000:+.3f}mm（模型估计，非实测）'
        lines.append(f"{p['candidate_id']} {p['validity']} | 原交界总后退14mm | TCP名义141.5mm | 接触带22mm，理想余量3mm")
        lines.append('红：完整夹爪投影；紫：接触带；橙：端子轮廓（'+p['terminal_evidence']+'）；人工确认：'+str(p['terminal_human_confirmed']))
        if p.get('terminal_visible_component_count',0)>1:
            lines.append(f"同一端子可见{p['terminal_visible_component_count']}片；反光/未知缺口保守填充，不当作夹爪净空")
        lines.append('完整模型到端子投影间距：'+number(full['minimum_nominal_clearance_m'])+'；接触带间距：'+number(contact['minimum_nominal_clearance_m']))
        lines.append('保留Phase1.7近缘：'+number(p['phase17_original_near_projection_m'])+'；当前近缘：'+number(p['current_contact_near_projection_m']))
        reasons=p['rejection_reasons']
        for i in range(0,len(reasons),2):lines.append('拒绝/阻塞：'+' / '.join(reasons[i:i+2]))
    return lines
