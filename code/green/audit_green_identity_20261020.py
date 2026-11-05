"""Read-only identity/geometry checks on preserved green derivatives; not GT certification."""
import hashlib,json,datetime
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
ROOT=Path('/workspace/clear-ec-research')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
g=ROOT/'datasets/clear_ec_official/green_eval'
meta=ROOT/'datasets/clear_ec_fullframe_20260904/fullframe_meta.npz'
with np.load(meta,allow_pickle=True) as z: ids=z['ids'].astype(str)
raw=np.memmap(ROOT/'datasets/clear_ec_fullframe_20260904/img_full_raw.dat',mode='r',dtype=np.uint8,shape=(len(ids),972,1296))
a=json.loads((g/'annotations.json').read_text())
split=ROOT/'experiments/target_fold4_20260912/preflight/fold4.npz'
with np.load(split) as z: tr=set(ids[z['train']]); va=set(ids[z['validation']])
rows=[]
for name,ann in a.items():
 ix=np.flatnonzero(ids==name); assert len(ix)==1,(name,ix)
 p=g/'mha'/f'{name}.mha'; b=p.read_bytes(); marker=b'ElementDataFile = LOCAL\n'; header,data=b.split(marker,1)
 assert b'CompressedData = False' in header and b'ElementType = MET_UCHAR' in header and b'DimSize = 1296 972' in header
 im=np.frombuffer(data,np.uint8).reshape(972,1296)
 assert np.array_equal(im,raw[ix[0]]),name
 pts=np.asarray(ann['markers_mha']); pg=np.asarray(ann['markers_green']); box=np.asarray(ann['roi_box_mha']); bg=np.asarray(ann['roi_box_green'])
 assert np.allclose(pts,np.asarray([[round(float(x)*1.8,1),round(float(y)*1.8,1)] for x,y in pg]),atol=1e-10)
 assert np.array_equal(box,np.asarray([round(float(x)*1.8) for x in bg]))
 assert ((pts>=0)&(pts<np.array([1296,972]))).all()
 assert ((pts>=box[:2])&(pts<=box[2:])).all()
 ov=g/'overlays'/f'{name}_check.png'; overlay=np.array(Image.open(ov).convert('RGB')); h,w=overlay.shape[:2]
 unmarked=(overlay.max(2).astype(float)-overlay.min(2))<15
 gray=overlay.mean(2); corrs={}
 for key,img in [('identity',im),('flip_x',im[:,::-1]),('flip_y',im[::-1,:]),('rot180',im[::-1,::-1])]:
  scaled=cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
  corrs[key]=float(np.corrcoef(scaled[unmarked],gray[unmarked])[0,1])
 rows.append(dict(id=name,fold4='train' if name in tr else 'validation' if name in va else 'excluded',raw_pixel_exact=True,mha_sha256=sha(p),overlay_sha256=sha(ov),overlay_shape=list(overlay.shape),scale_xy=[1296/w,972/h],points=len(pts),coordinate_conversion_exact=True,unmarked_fraction=float(unmarked.mean()),correlations=corrs,highest_correlation_transform=max(corrs,key=corrs.get)))
result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='derivative_identity_audit_completed',inputs={'annotations_sha256':sha(g/'annotations.json'),'metadata_sha256':sha(meta),'split_sha256':sha(split)},n=len(rows),exact_mha_cache_matches=sum(r['raw_pixel_exact'] for r in rows),fold4_counts={k:sum(r['fold4']==k for r in rows) for k in ['train','validation','excluded']},identity_highest=sum(r['highest_correlation_transform']=='identity' for r in rows),min_identity_correlation=min(r['correlations']['identity'] for r in rows),limitations=['Original green TIFF unavailable in searched locations; derivative PNG does not certify source completeness or extraction correctness','Correlation is a diagnostic, not independently prespecified acceptance threshold','No cell boundary masks; no calibration or model training performed','Edge cache ordering remains independently unverified'],rows=rows)
print(json.dumps(result,indent=2,allow_nan=False))
