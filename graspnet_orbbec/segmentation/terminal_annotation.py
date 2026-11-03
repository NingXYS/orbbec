"""原始RGB端子多边形：人工证据与自动检测分开，哈希/快照绑定后才能复核。"""
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
import cv2
import numpy as np


def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def pixel_sha(image):return sha_bytes(np.ascontiguousarray(image).tobytes())
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def avoidance_geometry(payload):
    """同一端子的可见片段合为保守凸包；反光缺口不是经验证的空闲空间。"""
    polygons=payload['polygons_xy']
    if not polygons:raise ValueError('TERMINAL_CONTOUR_MISSING')
    points=np.vstack([np.asarray(p,np.float32) for p in polygons])
    hull=cv2.convexHull(points).reshape(-1,2).astype(float).tolist()
    return {'terminal_instance_id':payload.get('terminal_instance_id',payload['candidate_id']+':terminal'),
        'visible_component_count':len(polygons),'polygons_xy':[hull],
        'method':'convex_hull_of_visible_parts','gap_treatment':'UNKNOWN_FILLED_CONSERVATIVELY',
        'physical_shape_verified':False}


def validate_polygon(points,width,height):
    from segmentation.phase17_geometry.policy import _segment_hit
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1:]!=(2,) or not 3<=len(p)<=512 or not np.isfinite(p).all():
        raise ValueError('TERMINAL_POLYGON_INVALID')
    if np.any(p<0) or np.any(p>=np.array([width,height])) or len(np.unique(p,axis=0))!=len(p):
        raise ValueError('TERMINAL_POLYGON_OUT_OF_IMAGE_OR_DUPLICATE')
    if abs(cv2.contourArea(p.astype(np.float32)))<1:
        raise ValueError('TERMINAL_POLYGON_DEGENERATE')
    edges=list(zip(p,np.roll(p,-1,axis=0)))
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if j<=i or j==i+1 or (i==0 and j==len(edges)-1):continue
            if _segment_hit(a,b,c,d):raise ValueError('TERMINAL_POLYGON_SELF_INTERSECTS')
    return p.tolist()


def create_annotation(image_path,image,body_mask,polygons,*,candidate_id,boundary_xy,
                      source,reviewer,human_confirmed=False,relationship_confirmed=False,
                      synthetic=False,previous_annotation_id=None,review_evidence=None):
    """确认必须由本地操作者操作；合成测试和助手草稿不能标成已人工确认。"""
    if source not in ('HUMAN_POLYGON_UI','HUMAN_CHAT_REVIEW','ASSISTANT_VISUAL_DRAFT','SYNTHETIC_TEST'):
        raise ValueError('ANNOTATION_SOURCE_UNDECLARED')
    if any(type(value) is not bool for value in (human_confirmed,relationship_confirmed,synthetic)):
        raise ValueError('ANNOTATION_CONFIRMATION_INVALID')
    if human_confirmed and (source not in ('HUMAN_POLYGON_UI','HUMAN_CHAT_REVIEW') or synthetic or not relationship_confirmed):
        raise ValueError('ANNOTATION_CONFIRMATION_INVALID')
    if source=='HUMAN_CHAT_REVIEW' and not review_evidence:raise ValueError('CHAT_REVIEW_EVIDENCE_MISSING')
    if not reviewer or not candidate_id:raise ValueError('ANNOTATION_REVIEWER_OR_INSTANCE_MISSING')
    height,width=image.shape[:2]
    if np.asarray(body_mask).shape!=(height,width):raise ValueError('ANNOTATION_BODY_MASK_MISMATCH')
    contours=[validate_polygon(p,width,height) for p in polygons]
    if not contours:raise ValueError('TERMINAL_CONTOUR_MISSING')
    boundary=np.asarray(boundary_xy,float)
    if boundary.shape!=(2,) or not np.isfinite(boundary).all() or np.any(boundary<0) or np.any(boundary>=[width,height]):
        raise ValueError('ANNOTATION_BOUNDARY_INVALID')
    mask=np.zeros((height,width),np.uint8)
    cv2.fillPoly(mask,[np.rint(p).astype(np.int32) for p in contours],1)
    payload={'schema_version':'terminal-polygon-v2','annotation_id':'pending',
        'image_id':'sha256:'+sha_bytes(Path(image_path).read_bytes()),'image_file_sha256':sha_bytes(Path(image_path).read_bytes()),
        'image_pixels_sha256':pixel_sha(image),'image_size':[width,height],
        'body_mask_sha256':pixel_sha(np.asarray(body_mask,dtype=np.uint8)),
        'candidate_id':candidate_id,'boundary_xy':boundary.tolist(),'coordinate_space':'original_rgb_pixels',
        'polygons_xy':contours,'annotated_at':datetime.now(timezone.utc).isoformat(),
        'annotation_source':source,'reviewer':reviewer,'human_confirmed':bool(human_confirmed),
        'relationship_confirmed':bool(relationship_confirmed),'synthetic':bool(synthetic),
        'terminal_insulation_relationship':{'kind':'terminal_adjacent_to_insulation',
            'overlap_pixels':int(np.count_nonzero(mask & np.asarray(body_mask,bool))),
            'boundary_test':'manual boundary and polygon are separate evidence; adjacency is not automatic semantic proof'},
        'previous_annotation_id':previous_annotation_id,'automatic_detection':False,
        'physical_geometry_verified':False}
    payload.update(terminal_instance_id=candidate_id+':terminal',visible_parts_of_same_terminal=True,
        confirmation_scope='visible_polygon_location_and_same_terminal_identity',review_evidence=review_evidence,
        reflection_gap_evidence='USER_REPORTED' if source=='HUMAN_CHAT_REVIEW' else 'UNKNOWN',
        full_physical_terminal_boundary_verified=False)
    payload['avoidance_geometry']=avoidance_geometry(payload)
    payload['annotation_id']='sha256:'+sha_bytes(canonical({k:v for k,v in payload.items() if k!='annotation_id'}))
    return payload


def save_annotation(payload,path):
    """JSON与独立SHA文件均独占创建；不存在自引用的整文件哈希字段。"""
    path=Path(path);sidecar=path.with_suffix(path.suffix+'.sha256')
    if path.exists() or sidecar.exists():raise FileExistsError('ANNOTATION_ALREADY_EXISTS')
    path.parent.mkdir(parents=True,exist_ok=True)
    envelope={'payload':payload,'payload_sha256':sha_bytes(canonical(payload))}
    raw=json.dumps(envelope,ensure_ascii=False,indent=2,allow_nan=False).encode()
    with path.open('xb') as f:f.write(raw)
    with sidecar.open('x') as f:f.write(sha_bytes(raw)+'\n')
    return {'annotation_file':str(path.resolve()),'annotation_file_sha256':sha_bytes(raw),'payload':payload}


def validate_annotation_payload(p,image_path,image,body_mask,*,candidate_id,boundary_xy):
    """内存草稿也要复核图像/实例/掩码；不能绕过文件加载器传入跨帧轮廓。"""
    if (p.get('schema_version') not in ('terminal-polygon-v1','terminal-polygon-v2') or p.get('coordinate_space')!='original_rgb_pixels'
            or p.get('automatic_detection') is not False):raise ValueError('ANNOTATION_CONTRACT_MISMATCH')
    if (p['image_file_sha256']!=sha_bytes(Path(image_path).read_bytes()) or p['image_id']!='sha256:'+p['image_file_sha256']
            or p['image_pixels_sha256']!=pixel_sha(image) or p['image_size']!=list(image.shape[1::-1])):
        raise ValueError('ANNOTATION_IMAGE_MISMATCH')
    if (p['body_mask_sha256']!=pixel_sha(np.asarray(body_mask,dtype=np.uint8)) or p['candidate_id']!=candidate_id
            or not np.allclose(np.asarray(p['boundary_xy'],float),boundary_xy,atol=1e-9,rtol=0)):
        raise ValueError('ANNOTATION_INSTANCE_BOUNDARY_OR_MASK_MISMATCH')
    if (p.get('annotation_source') not in ('HUMAN_POLYGON_UI','HUMAN_CHAT_REVIEW','ASSISTANT_VISUAL_DRAFT','SYNTHETIC_TEST')
            or any(type(p.get(k)) is not bool for k in ('human_confirmed','relationship_confirmed','synthetic'))
            or not p.get('reviewer') or not p.get('annotated_at')):
        raise ValueError('ANNOTATION_PROVENANCE_MISSING')
    try:
        stamp=datetime.fromisoformat(p['annotated_at'])
        if stamp.tzinfo is None:raise ValueError('missing timezone')
    except (ValueError,TypeError):raise ValueError('ANNOTATION_TIMESTAMP_INVALID')
    if p['human_confirmed'] and (p['annotation_source'] not in ('HUMAN_POLYGON_UI','HUMAN_CHAT_REVIEW') or p['synthetic'] or not p['relationship_confirmed']):
        raise ValueError('ANNOTATION_CONFIRMATION_INVALID')
    if p['annotation_source']=='HUMAN_CHAT_REVIEW' and not p.get('review_evidence'):raise ValueError('CHAT_REVIEW_EVIDENCE_MISSING')
    expected_id='sha256:'+sha_bytes(canonical({k:v for k,v in p.items() if k!='annotation_id'}))
    if p['annotation_id']!=expected_id:raise ValueError('ANNOTATION_ID_MISMATCH')
    for polygon in p['polygons_xy']:validate_polygon(polygon,*image.shape[1::-1])
    if not p['polygons_xy']:raise ValueError('TERMINAL_CONTOUR_MISSING')
    if p['schema_version']=='terminal-polygon-v2' and p.get('avoidance_geometry')!=avoidance_geometry(p):
        raise ValueError('TERMINAL_CONSERVATIVE_ENVELOPE_MISMATCH')
    return p


def load_annotation(path,image_path,image,body_mask,*,candidate_id,boundary_xy):
    path=Path(path);raw=path.read_bytes();digest=sha_bytes(raw)
    if digest!=path.with_suffix(path.suffix+'.sha256').read_text().strip():raise ValueError('ANNOTATION_FILE_HASH_MISMATCH')
    envelope=json.loads(raw);p=envelope['payload']
    if envelope['payload_sha256']!=sha_bytes(canonical(p)):raise ValueError('ANNOTATION_PAYLOAD_HASH_MISMATCH')
    validate_annotation_payload(p,image_path,image,body_mask,candidate_id=candidate_id,boundary_xy=boundary_xy)
    return {'annotation_file':str(path.resolve()),'annotation_file_sha256':digest,'payload':p}
