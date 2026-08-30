"""Stacking fusion of regression + fNLA-seg. Per metric, a gradient-boosting meta-learner predicts the
true value from [reg preds (3), fNLA preds (3), n_cells, log focus]; the aux features let it GATE fNLA
(down-weight on low-focus/few-cell images where fNLA's heavy tail fails). Train on pool tr rows, eval
on pool val/test. Compare vs regression-alone and vs a simple tuned linear blend. If it beats reg on
val, fusion is the lever."""
import json, numpy as np
ROOT="datasets/clear_ec_official"
F=json.load(open(f"{ROOT}/fnla_pool_scalars.json")); R=json.load(open(f"{ROOT}/reg_pool_preds.json"))
rows=[r for r in F if r in R]
def feats(r):
    fn=F[r]["fnla"]; rg=R[r]; return rg+fn+[F[r]["ncell"], np.log(F[r]["focus"]+1)]
X=np.array([feats(r) for r in rows]); G=np.array([F[r]["gt"] for r in rows]); SP=np.array([F[r]["sp"] for r in rows])
RG=np.array([R[r] for r in rows]); FN=np.array([F[r]["fnla"] for r in rows])
tr=SP=="tr"; va=SP=="va"; te=SP=="te"
print(f"rows {len(rows)}  tr{tr.sum()} va{va.sum()} te{te.sum()}")
def mape(p,g): return np.mean(np.abs(p-g)/g*100,0)
names=["CD","CV","HEX"]
# baselines
for nm,P in [("reg-alone",RG),("fNLA-alone",FN)]:
    e=mape(P[va],G[va]); print(f"[{nm}] VAL CD{e[0]:.2f} CV{e[1]:.2f} HEX{e[2]:.2f} OV{e.mean():.2f}")
try:
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    HAS=True
except Exception as e: HAS=False; print("no sklearn:",e)
def blend_opt(m):  # simple: w*reg+(1-w)*fnla, w tuned on tr
    best=(1.0,9e9)
    for w in np.linspace(0,1,41):
        p=w*RG[tr,m]+(1-w)*FN[tr,m]; e=mape(p,G[tr,m])
        if e<best[1]: best=(w,e)
    return best[0]
valm=[];tem=[];valb=[];teb=[]
for m in range(3):
    # (a) tuned linear blend
    w=blend_opt(m); pv=w*RG[va,m]+(1-w)*FN[va,m]; pt=w*RG[te,m]+(1-w)*FN[te,m]
    valb.append(mape(pv,G[va,m])); teb.append(mape(pt,G[te,m]))
    # (b) gradient boosting meta (predict ratio gt/reg for stability)
    if HAS:
        y=G[tr,m]/np.clip(RG[tr,m],1e-6,None)
        gb=HistGradientBoostingRegressor(max_iter=300,max_depth=3,learning_rate=0.05,l2_regularization=1.0)
        gb.fit(X[tr],y)
        pv=gb.predict(X[va])*RG[va,m]; pt=gb.predict(X[te])*RG[te,m]
        valm.append(mape(pv,G[va,m])); tem.append(mape(pt,G[te,m]))
        print(f"  {names[m]}: blend(w={w:.2f}) VAL {valb[-1]:.2f}/TEST {teb[-1]:.2f} | GBmeta VAL {valm[-1]:.2f}/TEST {tem[-1]:.2f}")
    else:
        print(f"  {names[m]}: blend(w={w:.2f}) VAL {valb[-1]:.2f}/TEST {teb[-1]:.2f}")
rge=mape(RG[va],G[va])
print(f"\n=== FUSION (pool subset) ===\nreg-alone     VAL OV {rge.mean():.2f}")
print(f"tuned-blend   VAL OV {np.mean(valb):.2f}  TEST OV {np.mean(teb):.2f}")
if HAS: print(f"GB-meta-gate  VAL OV {np.mean(valm):.2f}  TEST OV {np.mean(tem):.2f}  (reg-alone test {mape(RG[te],G[te]).mean():.2f})")
