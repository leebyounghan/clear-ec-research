"""Ensemble: blend regression + fNLA-seg per-image preds on common set.
Fit per-metric weight on val-half (500), evaluate on test-half (500). Also report error
correlation (does seg add independent signal?) and a simple stacking check."""
import numpy as np, json
ROOT="datasets/clear_ec_official"
E=np.load(f"{ROOT}/ens_common.npz",allow_pickle=True)
split=E["split"]; gt=E["gt"].astype(float); reg=E["regress"].astype(float); seg=E["seg"].astype(float)
vm=split=="val"; tm=split=="test"
names=["CD","CV","HEX"]
def mpe(p,g): return np.abs(p-g)/g*100
# error correlation per metric
print("per-metric error correlation (reg vs seg): lower=more independent=better ensemble")
for j,nm in enumerate(names):
    er=mpe(reg[:,j],gt[:,j]); es=mpe(seg[:,j],gt[:,j])
    print(f"  {nm}: corr={np.corrcoef(er,es)[0,1]:+.2f}  reg_MPE={er.mean():.1f} seg_MPE={es.mean():.1f}")
# fit per-metric weight w on val: pred=w*reg+(1-w)*seg, minimize val MPE
ws=np.linspace(0,1,101); best_w=np.zeros(3)
for j in range(3):
    errs=[mpe(w*reg[vm,j]+(1-w)*seg[vm,j],gt[vm,j]).mean() for w in ws]
    best_w[j]=ws[int(np.argmin(errs))]
print(f"\nfitted weights (w*reg+(1-w)*seg), per metric: {dict(zip(names,best_w.round(2)))}")
# evaluate on test-half
def ov(P,mask): return np.array([mpe(P[mask,j],gt[mask,j]).mean() for j in range(3)])
reg_t=ov(reg,tm); seg_t=ov(seg,tm)
blend=np.column_stack([best_w[j]*reg[:,j]+(1-best_w[j])*seg[:,j] for j in range(3)])
bl_t=ov(blend,tm)
print(f"\n=== TEST-half (500) OV MPE ===")
print(f"{'':6} {'CD':>6} {'CV':>6} {'HEX':>6} {'OV':>6}")
print(f"{'reg':6} {reg_t[0]:6.2f} {reg_t[1]:6.2f} {reg_t[2]:6.2f} {reg_t.mean():6.2f}")
print(f"{'seg':6} {seg_t[0]:6.2f} {seg_t[1]:6.2f} {seg_t[2]:6.2f} {seg_t.mean():6.2f}")
print(f"{'blend':6} {bl_t[0]:6.2f} {bl_t[1]:6.2f} {bl_t[2]:6.2f} {bl_t.mean():6.2f}")
json.dump({"weights":best_w.tolist(),"reg_test":reg_t.tolist(),"seg_test":seg_t.tolist(),
           "blend_test":bl_t.tolist()},open(f"{ROOT}/ens_blend_results.json","w"),indent=1)
