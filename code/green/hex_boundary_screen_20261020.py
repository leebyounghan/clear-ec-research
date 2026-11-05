"""Frozen CPU-only diagnostic: same HEX metric, boundary change vs seed oracle.
No fitting, no GPU, no write into historical artifacts. Cached order is legacy,
not authenticated by per-image hashes: this cannot promote a submission model.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast,json,hashlib,datetime,sys,time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.draw import polygon
R=Path('/workspace/clear-ec-research');G=R/'datasets/clear_ec_official/green_eval';OUT=R/'experiments/hex_boundary_screen_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def metric(mask):
 props=regionprops(mask);areas=np.array([x.area*(1000/1296)**2 for x in props]);hs=[]
 for c in props:
  r0,c0,r1,c1=c.bbox;ms=mask[r0:r1,c0:c1]==c.label
  ma,mi=c.axis_major_length*.8,c.axis_minor_length*.8;a=c.orientation
  hp=np.array([[ma/2,0],[ma/4,-mi/2],[-ma/4,-mi/2],[-ma/2,0],[-ma/4,mi/2],[ma/4,mi/2]])
  rot=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
  poly=hp@rot+np.array([np.ceil(c.centroid[0]-r0),np.ceil(c.centroid[1]-c0)])
  hm=np.zeros(ms.shape,bool);rr,cc=polygon(poly[:,0],poly[:,1],hm.shape);hm[rr,cc]=True
  union=np.logical_or(ms,hm).sum();hs.append(float(np.logical_and(ms,hm).sum()/union) if union else 0.)
 return [float(1e6/areas.mean()),float(areas.std()/areas.mean()),float(np.mean(hs))] if len(areas) else [0.,0.,0.]
def keep(lab,box):
 x0,y0,x1,y1=box;cand=[]
 for p in regionprops(lab):
  cy,cx=p.centroid;b=p.bbox
  if x0<=cx<x1 and y0<=cy<y1 and b[0]>y0 and b[2]<y1 and b[1]>x0 and b[3]<x1 and 80<=p.area<=4000:cand.append(p)
 areas=np.array([p.area for p in cand]);lut=np.zeros(int(lab.max())+1,np.int32)
 if len(areas):
  med=np.median(areas);mad=np.median(abs(areas-med))+1e-6
  good=[p for p in cand if abs(p.area-med)/(1.4826*mad)<3.]
  for i,p in enumerate(good,1):lut[p.label]=i
 return lut[lab]
def segment(edge,mode,box,points):
 work=ndi.gaussian_filter(edge,.75) if mode=='edge_smooth075' else edge
 interior=ndi.binary_opening(work<.2,iterations=1);distance=ndi.distance_transform_edt(interior)
 coords=peak_local_max(distance,min_distance=7,labels=interior,exclude_border=False)
 markers=np.zeros(edge.shape,np.int32)
 if mode=='oracle_centers':
  x0,y0,x1,y1=box;coords=np.array([[y,x] for y,x in coords if not(x0<=x<x1 and y0<=y<y1)],dtype=int).reshape(-1,2)
  pts=np.array([[int(round(y)),int(round(x))] for x,y in points],dtype=int)
  coords=np.concatenate([coords,pts])
 for i,(y,x) in enumerate(coords,1):
  if 0<=y<edge.shape[0] and 0<=x<edge.shape[1]:markers[y,x]=i
 markers=ndi.label(markers>0)[0]
 return watershed(work,markers=markers)
def main():
 if OUT.exists():raise FileExistsError('No overwrite/rerun: use a new experiment ID')
 OUT.mkdir(parents=True);start=time.time()
 inputs=[G/'annotations.json',G/'fnla_probs.npz',G/'fnla_seg2.py',G/'fnla_seg3.py',R/'experiments/target_fold4_20260912/preflight/fold4.npz',R/'datasets/clear_ec_official/cache_native_meta_512.npz']
 manifest={'source_sha256':sha(__file__),'inputs':{str(p):sha(p) for p in inputs},'arms':['baseline','edge_smooth075','oracle_centers'],'fixed':{'threshold':.2,'peak_distance':7,'gaussian_sigma':.75,'area_range':[80,4000],'mad':3,'fixed_roi':[379,282,917,690]},'role':'CPU cached segmentation diagnostic; oracle unavailable at deployment','order_provenance':'legacy producer fnla_seg2.py uses annotations insertion order; cache contains no IDs; not independently authenticated','no_fit':True,'gpu':0}
 save(OUT/'manifest.json',manifest)
 # Verify HEX implementation against the historical reference on synthetic masks.
 tree=ast.parse((G/'fnla_seg3.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='HEX')
 env={'np':np,'regionprops':regionprops,'polygon':polygon};exec(compile(ast.Module(body=[node],type_ignores=[]),'reference','exec'),env)
 smoke=[]
 for i in range(4):
  m=np.zeros((65,73),np.int32);m[5:15+i*4,8:25]=1;m[30:55,35:45+i*5]=2
  a=metric(m)[2];b=float(env['HEX'](m));assert abs(a-b)<1e-12;(smoke.append({'case':i,'difference':abs(a-b)}))
 assert metric(np.zeros((8,8),np.int32))==[0.,0.,0.]
 save(OUT/'cpu_smoke.json',{'passed':True,'reference_HEX_equivalence':smoke,'empty_mask':True})
 ann=json.loads((G/'annotations.json').read_text());ids=list(ann);z=np.load(G/'fnla_probs.npz');edge=z['edge'];assert edge.shape==(len(ids),972,1296) and np.isfinite(edge).all()
 meta=np.load(inputs[-1],allow_pickle=True);mid=[Path(str(x)).stem for x in meta['ids']];split=np.load(inputs[-2]);train={mid[i] for i in split['train']};val={mid[i] for i in split['validation']}
 rows=[]
 for j,ID in enumerate(ids):
  target=np.array([ann[ID]['gt'][k] for k in ['CD','CV','HEX']]);assert np.isfinite(target).all() and (target>0).all()
  for mode in manifest['arms']:
   lab=segment(edge[j],mode,ann[ID]['roi_box_mha'],ann[ID]['markers_mha'])
   for region,box in [('fixed_central',manifest['fixed']['fixed_roi']),('oracle_roi',ann[ID]['roi_box_mha'])]:
    if mode=='oracle_centers' and region=='fixed_central':continue
    masks=keep(lab,box);pred=metric(masks)
    rows.append({'id':ID,'arm':mode,'region':region,'group':'train_green20' if ID in train else 'validation_green5' if ID in val else 'outside_split','prediction':pred,'target':target.tolist(),'ape':(abs(np.array(pred)-target)/target*100).tolist(),'cells':int(masks.max()),'zero_cells':not bool(masks.max())})
  save(OUT/'progress.json',{'status':'running','images_complete':j+1,'total':len(ids),'elapsed_seconds':time.time()-start});print('image',j+1,len(ids),flush=True)
 summaries=[]
 for mode in manifest['arms']:
  for region in ['fixed_central','oracle_roi']:
   for group in ['train_green20','validation_green5','all25_reused_diagnostic']:
    subset=[x for x in rows if x['arm']==mode and x['region']==region and (group=='all25_reused_diagnostic' or x['group']==group)]
    if not subset:continue
    a=np.mean([x['ape'] for x in subset],axis=0);summaries.append({'arm':mode,'region':region,'group':group,'n':len(subset),'CD':a[0],'CV':a[1],'HEX':a[2],'mean':float(a.mean()),'zero_cells':sum(x['zero_cells'] for x in subset)})
 save(OUT/'result.json',{'status':'completed','manifest_sha256':sha(OUT/'manifest.json'),'summaries':summaries,'rows':rows,'seconds':time.time()-start,'limitations':['25 reused green examples; not independent heldout','edge cache lacks per-image ID hashes; mapping only supported by producer source','no ground truth boundary masks; scalar gains cannot establish segmentation accuracy','oracle centers/ROI are diagnostic only','no new training; fixed HEX not optimized to inflate metric']})
 for path,dig in manifest['inputs'].items():assert sha(path)==dig
 save(OUT/'complete.json',{'status':'completed','result_sha256':sha(OUT/'result.json'),'manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'input_hashes_rechecked':True})
 print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
