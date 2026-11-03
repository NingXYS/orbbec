"""列向量、SI单位、显式frame/time/version的抓取坐标计算。"""
from dataclasses import dataclass, asdict
import math,re
import numpy as np
import cv2


class Rejected(ValueError):
    def __init__(self,code):self.code=code;super().__init__(code)

def require(condition,code):
    if not condition:raise Rejected(code)

def frame(name):
    require(isinstance(name,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_/]*',name) is not None,'INVALID_FRAME_ID')
    return name

def vector(value,size=3):
    result=np.asarray(value,dtype=float)
    require(result.shape==(size,) and np.isfinite(result).all(),'INVALID_VECTOR')
    return result.copy()

def rigid(value):
    t=np.asarray(value,dtype=float)
    require(t.shape==(4,4) and np.isfinite(t).all(),'INVALID_HOMOGENEOUS_MATRIX')
    require(np.allclose(t[3],[0,0,0,1],atol=1e-9,rtol=0),'INVALID_HOMOGENEOUS_ROW')
    r=t[:3,:3]
    require(np.allclose(r.T@r,np.eye(3),atol=1e-7,rtol=0) and abs(np.linalg.det(r)-1)<1e-7,'INVALID_RIGHT_HANDED_ROTATION')
    return t.copy()

def inverse(t):
    t=rigid(t);out=np.eye(4);out[:3,:3]=t[:3,:3].T;out[:3,3]=-t[:3,:3].T@t[:3,3];return out

def pose_matrix(values,translation_unit,angle_unit):
    p=vector(values,6)
    require(translation_unit in ('m','mm') and angle_unit in ('rad','deg'),'UNDECLARED_UNITS')
    p[:3]*=1 if translation_unit=='m' else .001
    x,y,z=p[3:]*(1 if angle_unit=='rad' else math.pi/180)
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    out=np.eye(4);out[:3,:3]=rz@ry@rx;out[:3,3]=p[:3];return rigid(out)

@dataclass(frozen=True)
class Stamp:
    ns:int
    clock:str
    meaning:str
    def __post_init__(self):
        require(type(self.ns) is int and self.ns>0 and bool(self.clock) and bool(self.meaning),'INVALID_TIMESTAMP')

@dataclass(frozen=True)
class Transform:
    parent:str
    child:str
    matrix:object
    version:str
    stamp:object=None
    unit:str='m'
    def __post_init__(self):
        frame(self.parent);frame(self.child)
        require(self.unit=='m','TRANSFORM_NOT_SI')
        require(isinstance(self.version,str) and bool(self.version),'MISSING_TRANSFORM_VERSION')
        object.__setattr__(self,'matrix',rigid(self.matrix))
    def inv(self):return Transform(self.child,self.parent,inverse(self.matrix),self.version,self.stamp)

def compose(a,b):
    require(a.child==b.parent,'TRANSFORM_DIRECTION_MISMATCH')
    return Transform(a.parent,b.child,a.matrix@b.matrix,a.version+'+'+b.version,a.stamp or b.stamp)

def check_time(image,feedback,now,max_age_ns=100_000_000,max_skew_ns=20_000_000):
    require(isinstance(image,Stamp) and isinstance(feedback,Stamp) and isinstance(now,Stamp),'MISSING_TIMESTAMP')
    require(image.clock==feedback.clock==now.clock,'CLOCK_DOMAIN_MISMATCH')
    require(0<=now.ns-image.ns<=max_age_ns,'STALE_OR_FUTURE_IMAGE')
    require(abs(feedback.ns-image.ns)<=max_skew_ns,'POSE_IMAGE_TIME_SKEW')

def grasp_orientation(closing,outward):
    x,z=vector(closing),-vector(outward)
    require(np.linalg.norm(x)>1e-10 and np.linalg.norm(z)>1e-10,'DEGENERATE_GRASP_AXES')
    x/=np.linalg.norm(x);z/=np.linalg.norm(z)
    require(abs(x@z)<1e-5,'NONPERPENDICULAR_GRASP_AXES')
    y=np.cross(z,x);y/=np.linalg.norm(y);x=np.cross(y,z)
    return np.column_stack((x,y,z))

def pixels_on_plane(pixels,k,d,plane,height_m,image_size):
    uv=np.asarray(pixels,float);k=np.asarray(k,float);d=np.asarray(d,float)
    require(uv.ndim==2 and uv.shape[1:]==(2,) and len(uv)>0 and np.isfinite(uv).all(),'INVALID_PIXELS')
    require(k.shape==(3,3) and np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0 and np.allclose(k[2],[0,0,1]),'INVALID_INTRINSICS')
    require(d.shape in ((4,),(5,),(8,)) and np.isfinite(d).all(),'INVALID_BROWN_DISTORTION')
    width,height=image_size
    require(np.all((uv>=0)&(uv<np.array([width,height]))),'PIXEL_OUT_OF_IMAGE')
    n=vector(plane['normal_camera']);origin=vector(plane['origin_camera_m'])
    require(abs(np.linalg.norm(n)-1)<1e-7 and math.isfinite(height_m),'INVALID_PLANE')
    rays=np.c_[cv2.undistortPoints(uv.reshape(-1,1,2),k,d).reshape(-1,2),np.ones(len(uv))]
    denominator=rays@n;require(np.all(np.abs(denominator)>1e-8),'RAY_PARALLEL_TO_PLANE')
    scale=(origin@n+height_m)/denominator
    require(np.isfinite(scale).all() and np.all(scale>0),'POINT_BEHIND_CAMERA')
    return rays*scale[:,None]

@dataclass
class GraspTarget:
    schema_version:int
    observation:dict
    camera_point:object=None
    base_grasp:object=None
    tcp_target:object=None
    approach_direction:object=None
    jaw_width_m:object=None
    validity:str='INVALID'
    rejection_reasons:object=None
    planning_allowed:bool=False
    executable:bool=False
    execution_environment:str='OFFLINE_ONLY'
    def to_dict(self):return asdict(self)

def stamped_pose(t,stamp):
    t=rigid(t);v=cv2.Rodrigues(t[:3,:3])[0].reshape(3);theta=float(np.linalg.norm(v))
    q=np.r_[v*(math.sin(theta/2)/theta if theta>1e-12 else .5),math.cos(theta/2)]
    require(abs(np.linalg.norm(q)-1)<1e-10,'INVALID_QUATERNION')
    return {'position_m':t[:3,3].tolist(),'orientation_xyzw':q.tolist(),'stamp':asdict(stamp),'matrix':t.tolist(),
            'translation_unit':'m','rotation_representation':'unit_quaternion_xyzw'}

def build_target(obs,base_flange,flange_camera,camera_grasp,tcp_grasp,now,tool_quality):
    """未知工具或模型基座映射时只保留预览；永远不产生实机授权。"""
    out=GraspTarget(1,obs,rejection_reasons=[])
    try:
        for key in ('image_id','recognition_id','candidate_id','camera_serial','calibration_id','calibration_binding','image_stamp','source_kind','quantity','tcp_compensations','jaw_width_m','optical_convention'):
            require(key in obs and obs[key] is not None,'MISSING_'+key.upper())
        for key in ('image_id','recognition_id','candidate_id','camera_serial','calibration_id'):
            require(isinstance(obs[key],str) and bool(obs[key]),'EMPTY_'+key.upper())
        require(obs['quantity']=='camera_grasp' and type(obs['tcp_compensations']) is int and obs['tcp_compensations']==0,'DOUBLE_OR_AMBIGUOUS_TCP_COMPENSATION')
        require(obs['source_kind'] in ('SYNTHETIC','HISTORICAL_REPLAY'),'UNSUPPORTED_SOURCE_KIND')
        require(obs['optical_convention']=='x_right_y_down_z_forward','OPTICAL_FRAME_CONVENTION_MISMATCH')
        require(obs['calibration_binding']['camera_serial']==obs['camera_serial'] and obs['calibration_binding']['calibration_id']==obs['calibration_id'],'CAMERA_CALIBRATION_BINDING_MISMATCH')
        stamp=Stamp(**obs['image_stamp'])
        out.observation={**obs,'frame_contract_version':'phase16-v1',
                         'image_capture_time':dict(obs['image_stamp']) if obs['source_kind']=='SYNTHETIC' and obs.get('exposure_pose_verified') else None,
                         'host_receive_time':dict(obs['image_stamp']) if stamp.clock=='host_unix_receive' else None}
        require(all(isinstance(t,Transform) for t in (base_flange,flange_camera,camera_grasp)),'MISSING_TRANSFORM')
        require(flange_camera.version==obs['calibration_id'],'CALIBRATION_VERSION_MISMATCH')
        require(camera_grasp.child=='grasp','GRASP_FRAME_MISMATCH')
        require(isinstance(camera_grasp.stamp,Stamp) and camera_grasp.stamp==stamp,'GRASP_IMAGE_TIMESTAMP_MISMATCH')
        check_time(stamp,base_flange.stamp,now)
        require(isinstance(obs['jaw_width_m'],(int,float)) and not isinstance(obs['jaw_width_m'],bool) and math.isfinite(obs['jaw_width_m']) and 0<obs['jaw_width_m']<=.1,'INVALID_JAW_WIDTH')
        require(camera_grasp.matrix[2,3]>0,'POINT_BEHIND_CAMERA')
        base_camera=compose(base_flange,flange_camera);base_grasp=compose(base_camera,camera_grasp)
        out.camera_point={'frame_id':camera_grasp.parent,'unit':'m','stamp':asdict(stamp),'calibration_id':obs['calibration_id'],'xyz':camera_grasp.matrix[:3,3].tolist()}
        out.base_grasp={'frame_id':base_grasp.parent,'child_frame_id':'grasp','calibration_id':obs['calibration_id'],**stamped_pose(base_grasp.matrix,stamp)}
        out.approach_direction={'frame_id':base_grasp.parent,'stamp':asdict(stamp),'unit':'unit_vector','calibration_id':obs['calibration_id'],'xyz':base_grasp.matrix[:3,2].tolist()}
        out.jaw_width_m=float(obs['jaw_width_m']);reasons=[]
        if tcp_grasp is None:reasons.append('T_TCP_GRASP_MISSING')
        else:
            require(isinstance(tcp_grasp,Transform) and tcp_grasp.parent=='tcp_link' and tcp_grasp.child=='grasp','TCP_GRASP_FRAME_MISMATCH')
            require(tool_quality in ('SYNTHETIC','NOMINAL','VERIFIED'),'UNDECLARED_TOOL_QUALITY')
            target=compose(base_grasp,tcp_grasp.inv())
            if tool_quality in ('SYNTHETIC','VERIFIED'):
                out.tcp_target={'frame_id':target.parent,'child_frame_id':'tcp_link','calibration_id':obs['calibration_id'],**stamped_pose(target.matrix,stamp)}
            else:reasons.append('TOOL_NOMINAL_ONLY')
        if not obs.get('exposure_pose_verified'):reasons.append('EXPOSURE_POSE_UNVERIFIED')
        if base_grasp.parent!='base_link':reasons.append('SDK_BASE_TO_MODEL_BASE_MISSING')
        if obs['source_kind']!='SYNTHETIC':reasons+=['HISTORICAL_REPLAY_NOT_LIVE','ABSOLUTE_ACCURACY_NOT_VERIFIED','PHASE2_BLOCKED']
        out.rejection_reasons=reasons
        out.planning_allowed=(not reasons and obs['source_kind']=='SYNTHETIC' and tool_quality=='SYNTHETIC' and stamp.clock=='synthetic_ros_time')
        out.validity='VALID_OFFLINE' if out.planning_allowed else 'PREVIEW_ONLY'
    except (Rejected,KeyError,TypeError,ValueError) as error:
        out.validity='INVALID';out.rejection_reasons=[getattr(error,'code','MALFORMED_INPUT')]
        out.tcp_target=None;out.planning_allowed=False
    return out

def to_pose_stamped(target):
    """仅转换合格的合成离线目标；返回真实geometry_msgs类型，不发布。"""
    require(target.validity=='VALID_OFFLINE' and target.planning_allowed and not target.executable,'TARGET_NOT_ALLOWED_FOR_OFFLINE_PLANNING')
    p=target.tcp_target;require(p is not None and p['frame_id']=='base_link' and p['child_frame_id']=='tcp_link','MOVEIT_FRAME_MISMATCH')
    expected=stamped_pose(rigid(p['matrix']),Stamp(**p['stamp']))
    require(np.allclose(vector(p['position_m']),expected['position_m'],atol=1e-10,rtol=0)
            and abs(np.linalg.norm(vector(p['orientation_xyzw'],4))-1)<1e-10
            and abs(abs(np.dot(p['orientation_xyzw'],expected['orientation_xyzw']))-1)<1e-10,'POSE_PAYLOAD_INCONSISTENT')
    stamp=Stamp(**p['stamp']);require(stamp.clock=='synthetic_ros_time' and stamp.ns//10**9<2**31,'ROS_CLOCK_BINDING_MISSING')
    from geometry_msgs.msg import PoseStamped
    msg=PoseStamped();msg.header.frame_id='base_link';msg.header.stamp.sec=stamp.ns//10**9;msg.header.stamp.nanosec=stamp.ns%10**9
    msg.pose.position.x,msg.pose.position.y,msg.pose.position.z=map(float,p['position_m'])
    msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w=map(float,p['orientation_xyzw'])
    return msg
