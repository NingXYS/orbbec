"""从原始交界按三维弧长选择中心；二维包络不能认证实体净空。"""
from pathlib import Path
import json
import numpy as np
import cv2
from .chain import require, rigid, inverse, vector

ROOT = Path(__file__).resolve().parent


def config():
    return json.loads((ROOT / 'config/geometry_policy.json').read_text())


def select_center(path_m, *, anchor_kind, direction, scale_evidence,
                  offset_m=None, existing_offsets=0):
    """path[0] 必须是原始交界且朝绝缘方向；不消费历史派生抓取中心。"""
    cfg = config()['wire_policy']
    require(anchor_kind == 'original_boundary' and existing_offsets == 0,
            'OFFSET_ALREADY_APPLIED_OR_WRONG_ANCHOR')
    require(direction == 'toward_insulation', 'WIRE_DIRECTION_UNVERIFIED')
    require(scale_evidence in ('SYNTHETIC', 'MODEL_DERIVED', 'PHYSICALLY_VERIFIED'),
            'METRIC_SCALE_UNRELIABLE')
    distance = cfg['total_offset_m'] if offset_m is None else offset_m
    require(np.isfinite(distance) and distance > 0, 'INVALID_OFFSET')
    points = np.asarray(path_m, float)
    require(points.ndim == 2 and points.shape[1:] == (3,) and len(points) >= 2
            and np.isfinite(points).all(), 'MISSING_METRIC_CENTERLINE')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    require(np.all(lengths > 1e-10), 'DEGENERATE_CENTERLINE')
    arc = np.r_[0., lengths.cumsum()]
    require(arc[-1] >= distance, 'INSUFFICIENT_CENTERLINE_FOR_CENTER')
    j = int(np.searchsorted(arc, distance, side='left'))
    fraction = (distance - arc[j-1]) / lengths[j-1]
    center = points[j-1] + fraction * (points[j] - points[j-1])
    tangent = (points[j] - points[j-1]) / lengths[j-1]
    return dict(center_m=center.tolist(), tangent=tangent.tolist(), offset_m=float(distance),
                total_length_m=float(arc[-1]), boundary_m=points[0].tolist(),
                segment_index=j-1, segment_fraction=float(fraction),
                near_nominal_m=distance-cfg['nominal_occupancy_along_wire_m']/2,
                far_nominal_m=distance+cfg['nominal_occupancy_along_wire_m']/2,
                anchor_kind=anchor_kind, offsets_applied=1, scale_evidence=scale_evidence)


def contact_box(length_m=.022, transverse_m=None, depth_m=None):
    """工具 X 为闭合轴、Y 为沿线占用轴；尺寸缺失时不补默认厚度。"""
    require(transverse_m is not None and depth_m is not None, 'GRIPPER_GEOMETRY_MISSING')
    dims = np.array([transverse_m, length_m, depth_m], float)
    require(np.isfinite(dims).all() and np.all(dims > 0), 'INVALID_GRIPPER_GEOMETRY')
    return np.array([[x,y,z] for x in (-dims[0]/2,dims[0]/2)
                     for y in (-dims[1]/2,dims[1]/2) for z in (-dims[2]/2,dims[2]/2)])


def _hull(points):
    points = np.asarray(points, float)
    require(points.ndim == 2 and points.shape[1:] == (2,) and len(points) >= 3
            and np.isfinite(points).all(), 'INVALID_POLYGON')
    result = cv2.convexHull(points.astype(np.float32)).reshape(-1,2).astype(float)
    require(abs(cv2.contourArea(result.astype(np.float32))) > 1e-12, 'DEGENERATE_FOOTPRINT')
    return result


def _point_segment(p,a,b):
    ab=b-a
    f=np.clip(np.dot(p-a,ab)/max(np.dot(ab,ab),1e-30),0,1)
    return float(np.linalg.norm(p-(a+f*ab)))


def _segment_hit(a,b,c,d):
    cross=lambda u,v:float(u[0]*v[1]-u[1]*v[0])
    u,v=b-a,d-c;den=cross(u,v)
    if abs(den) < 1e-14:
        return min(_point_segment(a,c,d),_point_segment(b,c,d),
                   _point_segment(c,a,b),_point_segment(d,a,b)) <= 1e-10
    t,s=cross(c-a,v)/den,cross(c-a,u)/den
    return -1e-10<=t<=1+1e-10 and -1e-10<=s<=1+1e-10


def polygon_clearance(a,b):
    """支持凹端子轮廓；相交、包含或接触均算碰撞，距离是平面模型距离。"""
    a=np.asarray(a,float);b=np.asarray(b,float)
    for p in (a,b):
        require(p.ndim==2 and p.shape[1:]==(2,) and len(p)>=3
                and np.isfinite(p).all(), 'INVALID_POLYGON')
    edges_a=list(zip(a,np.roll(a,-1,axis=0)));edges_b=list(zip(b,np.roll(b,-1,axis=0)))
    hit=any(_segment_hit(x,y,u,v) for x,y in edges_a for u,v in edges_b)
    hit=hit or any(cv2.pointPolygonTest(b.astype(np.float32),tuple(map(float,p)),False)>=0 for p in a)
    hit=hit or any(cv2.pointPolygonTest(a.astype(np.float32),tuple(map(float,p)),False)>=0 for p in b)
    distance=0. if hit else min([_point_segment(p,x,y) for p in a for x,y in edges_b]
                               +[_point_segment(p,x,y) for p in b for x,y in edges_a])
    return bool(hit),distance


def validate_footprint(selection, T_camera_tool, vertices_tool_m, terminal_polygons_m,
                       insulation_polygon_m, *, outward, scale_evidence,
                       geometry_evidence, terminal_evidence, scene_geometry_complete=False):
    """正交投影全部占用顶点到线材平面并检查；不以弧长间距替代包络。"""
    out=dict(validity='PREVIEW_ONLY', geometric_check='NOT_TESTED',
             intersects_terminal=None, minimum_nominal_clearance_m=None,
             insulation_sufficient=False, physical_clearance_verified=False,
             projected_polygon_camera_m=None, rejection_reasons=[], executable=False,
             evidence={'scale':scale_evidence,'gripper':geometry_evidence,'terminal':terminal_evidence})
    try:
        require(scale_evidence in ('SYNTHETIC','MODEL_DERIVED','PHYSICALLY_VERIFIED'), 'METRIC_SCALE_UNRELIABLE')
        require(vertices_tool_m is not None and geometry_evidence!='MISSING', 'GRIPPER_GEOMETRY_MISSING')
        t=rigid(T_camera_tool);center=vector(selection['center_m']);tangent=vector(selection['tangent'])
        require(np.linalg.norm(center-t[:3,3])<1e-9,'FOOTPRINT_CENTER_MISMATCH')
        n=vector(outward);require(abs(np.linalg.norm(n)-1)<1e-7,'INVALID_PLANE_NORMAL')
        tangent=tangent-n*(n@tangent);require(np.linalg.norm(tangent)>1e-10,'INVALID_WIRE_TANGENT');tangent/=np.linalg.norm(tangent)
        across=np.cross(n,tangent);basis=np.column_stack([tangent,across])
        vertices=np.asarray(vertices_tool_m,float)
        require(vertices.ndim==2 and vertices.shape[1:]==(3,) and len(vertices)>=4
                and np.isfinite(vertices).all(),'INVALID_GRIPPER_GEOMETRY')
        world=vertices@t[:3,:3].T+center;poly=_hull((world-center)@basis)
        out['projected_polygon_camera_m']=(poly@basis.T+center).tolist()
        out['projected_width_along_wire_m']=float(np.ptp(poly[:,0]))
        out['near_projected_boundary_m']=float((center-vector(selection['boundary_m']))@tangent+poly[:,0].min())
        require(selection['total_length_m']>=max(.025,selection['offset_m']+float(poly[:,0].max()))-1e-12,
                'INSULATION_SEGMENT_TOO_SHORT')
        require(terminal_polygons_m is not None and len(terminal_polygons_m)>0
                and terminal_evidence!='MISSING', 'TERMINAL_REGION_MISSING')
        clearances=[];hit=False
        for terminal in terminal_polygons_m:
            points=np.asarray(terminal,float)
            require(points.ndim==2 and points.shape[1:]==(3,) and len(points)>=3
                    and np.isfinite(points).all(),'INVALID_TERMINAL_REGION')
            intersects,distance=polygon_clearance(poly,(points-center)@basis)
            hit|=intersects;clearances.append(distance)
        out['intersects_terminal']=bool(hit);out['minimum_nominal_clearance_m']=min(clearances)
        require(not hit,'TERMINAL_ENVELOPE_INTERSECTION')
        require(insulation_polygon_m is not None,'INSULATION_REGION_MISSING')
        body=np.asarray(insulation_polygon_m,float)
        require(body.ndim==2 and body.shape[1:]==(3,) and len(body)>=3
                and np.isfinite(body).all(),'INVALID_INSULATION_REGION')
        body=(body-center)@basis
        # 只要求被占用的实际线身处于绝缘区域；夹爪横向空间另由三维场景验证。
        wire_points=np.array([selection['boundary_m'],selection['center_m']],float)
        samples=np.linspace(-.5,.5,23)*out['projected_width_along_wire_m']
        require(all(cv2.pointPolygonTest(body.astype(np.float32),(float(s),0.),False)>=0
                    for s in samples),'INSULATION_ENVELOPE_NOT_COVERED')
        out['insulation_sufficient']=True;out['geometric_check']='PASS_2D_MODEL'
        out['validity']='VALID_OFFLINE' if all(e=='SYNTHETIC' for e in (scale_evidence,geometry_evidence,terminal_evidence)) else 'PREVIEW_ONLY'
        if not scene_geometry_complete:out['rejection_reasons'].append('FULL_3D_SCENE_CLEARANCE_NOT_VERIFIED')
    except (ValueError,TypeError,KeyError) as e:
        code=getattr(e,'code','MALFORMED_GEOMETRY_INPUT');out['rejection_reasons'].append(code)
        if code in ('TERMINAL_ENVELOPE_INTERSECTION','INSULATION_SEGMENT_TOO_SHORT','INSULATION_ENVELOPE_NOT_COVERED'):
            out['validity']='INVALID';out['geometric_check']='FAIL'
    return out


def tcp_mapping(T_base_grasp, *, already_compensated=False, plate_correction_m=0.):
    """输出名义工具 TCP 与现有 MoveIt tcp_link(法兰参考)；补偿只在此发生一次。"""
    require(not already_compensated,'DOUBLE_TCP_COMPENSATION')
    require(plate_correction_m==0.,'BASE_PLATE_DOUBLE_COMPENSATION')
    cfg=config()['tool'];tool=rigid(cfg['T_flange_grasp_tcp']);grasp=rigid(T_base_grasp)
    # 用户定义第一版工具TCP恰为目标接触原点；identity仅是名义设计，不是测量结果。
    return dict(T_base_grasp_tcp=grasp.tolist(), T_base_model_tcp=(grasp@inverse(tool)).tolist(),
                T_flange_grasp_tcp=tool.tolist(), model_endpoint='tcp_link',
                nominal_endpoint='grasp_tcp_link',quality='NOMINAL_USER_SELECTED',
                compensation_count=1, physical_mapping_verified=False, executable=False)
