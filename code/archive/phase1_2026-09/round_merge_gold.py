"""Merge the two candidate sources (cpsam self-training + original SAM) into round gold.
Per image: pick the source with the LOWER total error vs GT. Gate: keep only if all of CD/CV/HEX
<= TOL. Assemble round{r}/gold = original 26 gold + newly accepted pseudo-gold (winning source's
mask + its full CLAHE image). Reports the source split and error stats. TOL chosen to harvest WANT
masks if not fixed via env."""
import os, json, glob, shutil, numpy as np, tifffile
ROOT="datasets/clear_ec_official"; H,W=972,1296
RD=os.environ.get("RD",f"{ROOT}/round1"); TOL=os.environ.get("TOL",""); WANT=int(os.environ.get("WANT","120"))
SEED_GOLD=os.environ.get("SEED_GOLD",f"{ROOT}/annotation_drafts/gold")
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); N=int(meta["N"])
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
cp=json.load(open(f"{RD}/cpsam_cand.json")); sm=json.load(open(f"{RD}/sam_cand.json"))
ids=sorted(set(cp)|set(sm),key=int)
def maxe(v): return max(v["eCD"],v["eCV"],v["eHEX"])
# collect EVERY passing source-mask independently (an image may add 0/1/2 masks:
# cpsam=good HEX, sam=independent CD/CV -> student learns both)
cands=[]  # (imgid, src, v)
for k in ids:
    c=cp.get(k,{}); s=sm.get(k,{})
    if c.get("ok"): cands.append((k,"cpsam",c))
    if s.get("ok"): cands.append((k,"sam",s))
# choose TOL to hit WANT total source-masks
if TOL=="":
    for t in [8,9,10,11,12,13,14,15,16,18,20]:
        if sum(1 for _,_,v in cands if maxe(v)<=t)>=WANT: TOL=t; break
    else: TOL=20
TOL=float(TOL)
acc=[(k,src,v) for k,src,v in cands if maxe(v)<=TOL]
# assemble gold dir: seed + accepted (source-suffixed name so both sources of one image coexist)
GD=f"{RD}/gold"
if os.path.isdir(GD): shutil.rmtree(GD)
os.makedirs(GD)
nseed=0
for ip in glob.glob(f"{SEED_GOLD}/*_img.tif"):
    ID=os.path.basename(ip).split("_")[0]; mp=f"{SEED_GOLD}/{ID}_mask.tif"
    if os.path.exists(mp): shutil.copy(ip,f"{GD}/s{ID}_img.tif"); shutil.copy(mp,f"{GD}/s{ID}_mask.tif"); nseed+=1
nnew=0; srccount={"cpsam":0,"sam":0}
for k,src,v in acc:
    mp=f"{RD}/{src}/{k}_mask.tif"
    if not os.path.exists(mp): continue
    tag=f"{k}{'c' if src=='cpsam' else 'm'}"
    tifffile.imwrite(f"{GD}/{tag}_img.tif", np.array(full[int(k)]).astype(np.uint8))
    shutil.copy(mp,f"{GD}/{tag}_mask.tif"); srccount[src]+=1; nnew+=1
print(f"candidates: cpsam_ok={sum(1 for v in cp.values() if v.get('ok'))} sam_ok={sum(1 for v in sm.values() if v.get('ok'))}")
print(f"TOL={TOL:.0f}%  accepted source-masks={len(acc)}  (cpsam {srccount['cpsam']} / sam {srccount['sam']})")
print(f"gold dir -> {GD}: seed {nseed} + new {nnew} = {nseed+nnew} masks")
if acc:
    print(f"  accepted mean err: CD {np.mean([v['eCD'] for _,_,v in acc]):.1f}% CV {np.mean([v['eCV'] for _,_,v in acc]):.1f}% HEX {np.mean([v['eHEX'] for _,_,v in acc]):.1f}%")
rows=cands
for t in [8,10,12,15,20]:
    print(f"  harvest@{t}% (source-masks): {sum(1 for _,_,v in cands if maxe(v)<=t)}")
json.dump({"tol":TOL,"accepted":[k for k,_,_ in acc],"nseed":nseed,"nnew":nnew}, open(f"{RD}/merge.json","w"))
