"""原样复用Phase1.7的原始点击到中心线投影。"""
import numpy as np
from .chain import require

def project_boundary_to_path(path,boundary):
    """人工点击投影到原始米制中心线；明确记录横向残差，不当成精确物理边界。"""
    path=np.asarray(path,float);boundary=np.asarray(boundary,float)
    require(path.ndim==2 and path.shape[1:]==(3,) and len(path)>=2
            and boundary.shape==(3,) and np.isfinite(path).all() and np.isfinite(boundary).all(),
            'MISSING_METRIC_CENTERLINE')
    choices=[]
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        d=b-a;length2=float(d@d)
        if length2<=1e-20:continue
        f=float(np.clip((boundary-a)@d/length2,0,1));p=a+f*d
        choices.append((float(np.linalg.norm(p-boundary)),i,f,p))
    require(bool(choices),'DEGENERATE_CENTERLINE')
    distance,i,f,p=min(choices,key=lambda v:v[0]);tail=np.vstack([p,path[i+1:]])
    tail=tail[np.r_[True,np.linalg.norm(np.diff(tail,axis=0),axis=1)>1e-10]]
    return tail,{'centerline_segment':i,'fraction':f,'manual_to_centerline_residual_m':distance,
                 'manual_boundary_camera_m':boundary.tolist(),'anchor_camera_m':p.tolist(),
                 'quality':'MODEL_DERIVED','physical_boundary_verified':False}

