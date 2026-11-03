"""原视觉结果的纯离线几何适配；名义全夹爪投影与22mm接触带分开报告。"""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from segmentation.phase17_geometry.chain import rigid,pixels_on_plane,grasp_orientation
from segmentation.phase17_geometry.policy import validate_footprint,contact_box,polygon_clearance,_hull,tcp_mapping
from segmentation.grasp_policy import apply_grasp_policy
from segmentation.terminal_annotation import pixel_sha

DATA=Path(__file__).resolve().parent/'phase17_geometry/data'


@dataclass
class FrozenScene:
    image_path:Path
    camera:dict
    plane:dict
    height_m:float
    T_base_camera:object
    calibration_id:str
    stamp:dict
    original_near_projection_m:float
    source:str
    source_kind:str='HISTORICAL_REPLAY'

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        if data['source_kind']!='HISTORICAL_REPLAY' or data['executable'] is not False:
            raise ValueError('OFFLINE_FROZEN_SCENE_REQUIRED')
        for item in data['inputs'].values():
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('FROZEN_INPUT_HASH_MISMATCH')
        inputs={k:Path(v['path']) for k,v in data['inputs'].items()}
        load=lambda name:json.loads(inputs[name].read_text())
        binding,camera,session,result,pair,rec,frames=[load(k) for k in ['binding','camera','session','result','pair','recognition','frames']]
        for key,alias in [('result','result'),('camera','camera'),('session','session')]:
            if hashlib.sha256(inputs[alias].read_bytes()).hexdigest()!=binding['sha256'][key]:raise ValueError('CALIBRATION_FILE_HASH_MISMATCH')
        if (session['robot']['reference']!='base_flange' or binding['method']!='PARK'
                or camera['serial']!=binding['camera_serial'] or pair['calibration_sha256']!=binding['sha256']):
            raise ValueError('CALIBRATION_REFERENCE_OR_VERSION_MISMATCH')
        image=inputs['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=pair['image_sha256']:raise ValueError('IMAGE_PAIR_MISMATCH')
        matrix=rigid(pair['T_base_flange'])@rigid(result['methods']['PARK']['T_gripper_rgb'])
        return cls(image,camera['rgb'],rec['grasp_policy']['support_plane'],data['center_height_m'],matrix,
            data['calibration_id'],{'ns':int(frames['rgb']['host_receive_timestamp_us'])*1000,'clock':'host_unix_receive','meaning':'historical RGB host receive, not exposure'},
            data['phase17_original_near_projection_m'],str(Path(path).resolve()))

    def backproject(self,points):
        return pixels_on_plane(points,self.camera['K'],self.camera['D'],self.plane,self.height_m,
                               (self.camera['width'],self.camera['height']))

    def project(self,points):
        return cv2.projectPoints(np.asarray(points,float).reshape(-1,3),np.zeros(3),np.zeros(3),
            np.asarray(self.camera['K'],float),np.asarray(self.camera['D'],float))[0].reshape(-1,2)


def _full_model():
    meta=json.loads((DATA/'gripper_25mm_model.json').read_text());raw=(DATA/'gripper_25mm_model.npz').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['mesh_data_sha256']:raise ValueError('GRIPPER_MODEL_HASH_MISMATCH')
    with np.load(DATA/'gripper_25mm_model.npz',allow_pickle=False) as data:v=np.array(data['vertices_flange_m'])
    return v,meta


def full_model_projection(selection,T_camera_grasp,normal,terminal_polygons):
    """保守投影完整法兰/夹爪/两指；高度层被合并，不声称二维交叉就等于3D碰撞。"""
    vertices,meta=_full_model();t=rigid(T_camera_grasp);tool=np.array(tcp_mapping(np.eye(4))['T_flange_grasp_tcp'])
    vertices=vertices-tool[:3,3]
    world=vertices@t[:3,:3].T+t[:3,3]
    tangent=np.array(selection['tangent']);n=np.array(normal);tangent-=n*(n@tangent);tangent/=np.linalg.norm(tangent)
    basis=np.column_stack([tangent,np.cross(n,tangent)]);center=t[:3,3]
    hull=_hull((world-center)@basis)
    report={'quality':'MODEL_DERIVED','geometry_complete_for_fixed_urdf_gripper':True,
        'physical_geometry_verified':False,'jaw_width_m':.025,'projected_polygon_camera_m':(hull@basis.T+center).tolist(),
        'projected_width_along_wire_m':float(np.ptp(hull[:,0])),
        'minimum_nominal_clearance_m':None,'intersects_terminal':None,'geometry_sources':meta['sources'],
        'interpretation':'conservative orthographic projection over all tool heights; not 3D contact proof'}
    if terminal_polygons:
        checks=[polygon_clearance(hull,(np.asarray(p)-center)@basis) for p in terminal_polygons]
        report['intersects_terminal']=any(c[0] for c in checks);report['minimum_nominal_clearance_m']=min(c[1] for c in checks)
    return report


def apply_preview(result,cfg,scene=None,annotation=None,*,orientation_override=None):
    """输入必须保留原始路径/交界；历史派生中心和TCP字段被忽略，不发任何指令。"""
    boundaries={p['candidate_id']:p['boundary_input_xy'] for p in result.grasp_plans if p.get('boundary_confirmed')}
    metrics={}
    if scene is not None:
        original=cv2.imread(str(scene.image_path))
        if original is None or pixel_sha(original)!=pixel_sha(result.image_bgr):raise ValueError('FROZEN_IMAGE_MISMATCH')
        for candidate in result.candidates:
            key=candidate['id']
            if key in boundaries and not candidate.get('path_ambiguous'):
                metrics[key]={'xyz_m':scene.backproject(candidate['inward_path_xy']),
                    'plane_normal':scene.plane['normal_camera'],'boundary_camera_m':scene.backproject([boundaries[key]])[0],
                    'input_quantity':'original_camera_centerline'}
    apply_grasp_policy(result,cfg,boundaries,metrics)
    result.manual_terminal_annotations=[] if annotation is None else [annotation]
    for plan in result.grasp_plans:
        plan.update(executable=False,planning_allowed=False,tcp_target=None,phase2='BLOCKED')
        if plan.get('policy_version')!='wire-v2-total14mm':continue
        if plan['status']!='camera_frame_preview_ready':continue
        selected=plan['selection_v2'];t=np.eye(4)
        t[:3,:3]=grasp_orientation(plan['closing_direction_camera'],plan['support_plane_normal']);t[:3,3]=selected['center_m']
        if orientation_override is not None:
            override=rigid(orientation_override)
            if np.linalg.norm(override[:3,3]-t[:3,3])>1e-9:raise ValueError('PLANNING_ORIENTATION_CENTER_MISMATCH')
            t=override
        plan['grasp_xy_preview']=scene.project([selected['center_m']])[0].tolist()
        terminal=[];annotation_reasons=[];payload=annotation.get('payload') if annotation else None
        if payload:
            if payload['candidate_id']!=plan['candidate_id'] or not np.allclose(payload['boundary_xy'],boundaries[plan['candidate_id']],atol=1e-9,rtol=0):
                annotation_reasons.append('TERMINAL_ANNOTATION_WRONG_INSTANCE_OR_BOUNDARY')
            else:
                from segmentation.terminal_annotation import validate_annotation_payload
                validate_annotation_payload(payload,scene.image_path,result.image_bgr,result.mask,
                    candidate_id=plan['candidate_id'],boundary_xy=boundaries[plan['candidate_id']])
                from segmentation.terminal_annotation import avoidance_geometry
                avoid=avoidance_geometry(payload)
                terminal=[scene.backproject(p) for p in avoid['polygons_xy']]
                if not payload['human_confirmed']:annotation_reasons.append('TERMINAL_ANNOTATION_UNCONFIRMED')
                if not payload['relationship_confirmed']:annotation_reasons.append('TERMINAL_INSULATION_RELATION_UNCONFIRMED')
        else:annotation_reasons.append('TERMINAL_CONTOUR_MISSING')
        contours,_=cv2.findContours(result.mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center_uv=tuple(map(float,plan['grasp_xy_preview']));body=None
        for contour in contours:
            if len(contour)>=3 and cv2.pointPolygonTest(contour,center_uv,False)>=0:
                body=scene.backproject(contour.reshape(-1,2));break
        # 接触带厚度来自Phase1.7固定STL截面；完整实体包络另算，不能用22mm替代整夹爪。
        vertices=contact_box(transverse_m=.025+2*.007029594044850909,depth_m=.002)
        contact=validate_footprint(selected,t,vertices,terminal or None,body,outward=plan['support_plane_normal'],
            scale_evidence='MODEL_DERIVED',geometry_evidence='MODEL_DERIVED',
            terminal_evidence='MISSING' if not terminal else 'MANUAL_POLYGON',scene_geometry_complete=False)
        full=full_model_projection(selected,t,plan['support_plane_normal'],terminal)
        for footprint in (contact,full):
            corners=footprint.get('projected_polygon_camera_m')
            footprint['projected_polygon_xy']=None if corners is None else scene.project(corners).tolist()
        hard=[]
        if full['intersects_terminal']:hard.append('FULL_GRIPPER_PROJECTION_INTERSECTS_TERMINAL')
        if contact['validity']=='INVALID':hard+=contact['rejection_reasons']
        reasons=list(dict.fromkeys(hard+annotation_reasons+contact['rejection_reasons']+[
            'METRIC_SCALE_AND_CENTER_HEIGHT_NOT_PHYSICALLY_VERIFIED','TCP_NOMINAL_USER_SELECTED_ONLY',
            'EXPOSURE_POSE_UNVERIFIED','SDK_BASE_TO_MODEL_BASE_MISSING','FULL_3D_SCENE_CLEARANCE_NOT_VERIFIED','PHASE2_BLOCKED']))
        base=scene.T_base_camera@t
        mapping=tcp_mapping(base)
        mapping.update(frame_id='base_link' if scene.source_kind=='SYNTHETIC' else 'historical_sdk_base',
            stamp=scene.stamp,calibration_id=scene.calibration_id,translation_unit='m')
        plan.update(status='offline_grasp_preview',validity='REJECTED' if hard else 'PREVIEW_ONLY',
            rejection_reasons=reasons,contact_footprint=contact,full_gripper_footprint=full,
            T_camera_grasp=t.tolist(),grasp_base_m=base[:3,3].tolist(),
            nominal_tcp_preview=mapping,frame_id='base_link' if scene.source_kind=='SYNTHETIC' else 'historical_sdk_base',stamp=scene.stamp,
            source_kind=scene.source_kind,
            calibration_id=scene.calibration_id,translation_unit='m',orientation_representation='matrix',
            center_height_m=scene.height_m,metric_evidence='MODEL_DERIVED_UNVERIFIED_HEIGHT',
            nominal_straight_margin_m=.003,phase17_original_near_projection_m=scene.original_near_projection_m,
            current_contact_near_projection_m=contact.get('near_projected_boundary_m'),
            orientation_source='explicit_offline_candidate_pose' if orientation_override is not None else 'local_wire_tangent_nominal_candidate_pose',
            actual_hardware_planned_pose=False,gripper_width_along_wire_m=.022,nominal_tcp_length_m=.1415,
            terminal_evidence='MISSING' if payload is None else payload['annotation_source'],
            terminal_human_confirmed=False if payload is None else payload['human_confirmed'],
            terminal_gap_policy='UNKNOWN_FILLED_CONSERVATIVELY' if terminal else 'MISSING',
            terminal_visible_component_count=0 if payload is None else len(payload['polygons_xy']),
            all_distances_model_only=True,physical_clearance_verified=False)
    result.grasp_policy.update(interpretation='Phase1.8离线预览：全夹爪投影为保守二维检查，非物理安全认证',
        policy_version='wire-v2-total14mm' if cfg.get_path('wire.grasp_policy.offset_mm',14)==14 else 'legacy-comparison-only')
    return result
