"""Instant tests once member_preds.npz exists (no GPU):
A) per-metric multiplicative calibration (beta=-1 shading generalization): k tuned on val, test transfer.
B) per-metric greedy member selection incl. fullframe (redo of ens_specialist selection with ff).
C) FFT-feature stack: per metric, linear model on val? -> NO, fit on val overfits. Instead: ridge fit on
   TRAIN is impossible (no member preds on train). So test additive value via val-tuned single blend
   weight w: pred = (1-w)*ens + w*fft_cd (CD only), and report val->test transfer. Small param count (1)
   tuned on val is acceptable (same protocol as member selection).
"""
import json, numpy as np
ROOT="datasets/clear_ec_official"
z=np.load(f"{ROOT}/member_preds.npz",allow_pickle=True)
va=z["va_idx"]; te=z["te_idx"]; Gva=z["gt_va"].astype(float); Gte=z["gt_te"].astype(float)
names=[k[:-3] for k in z.files if k.endswith("_va") and not k.startswith(("gt","va","te"))]
Pv={n:z[f"{n}_va"].astype(float) for n in names}; Pt={n:z[f"{n}_te"].astype(float) for n in names}
ok_va=(Gva>0).all(1); ok_te=(Gte>0).all(1)
def mape(p,g): return float(np.mean(np.abs(p-g)/g*100))
M=["CD","CV","HEX"]
print("members:",names)
for n in names:
    print(f"[{n}] val",[round(mape(Pv[n][ok_va,m],Gva[ok_va,m]),2) for m in range(3)],
          "test",[round(mape(Pt[n][ok_te,m],Gte[ok_te,m]),2) for m in range(3)])
# B) greedy per-metric selection
sel_all={}
print("\n=== greedy per-metric (incl ff) ===")
val_f=[];test_f=[]
for m in range(3):
    sel=[];cur=1e9
    while True:
        best=None
        for n in names:
            if n in sel: continue
            e=mape(np.mean([Pv[j][ok_va,m] for j in sel+[n]],0),Gva[ok_va,m])
            if e<cur-1e-4: cur=e;best=n
        if best is None: break
        sel.append(best)
    pv=np.mean([Pv[j][ok_va,m] for j in sel],0); pt=np.mean([Pt[j][ok_te,m] for j in sel],0)
    sel_all[m]=sel
    val_f.append(mape(pv,Gva[ok_va,m])); test_f.append(mape(pt,Gte[ok_te,m]))
    print(f"{M[m]}: {'+'.join(sel)}  VAL {val_f[-1]:.2f} TEST {test_f[-1]:.2f}")
print(f"ENSEMBLE OV: VAL {np.mean(val_f):.2f} TEST {np.mean(test_f):.2f}")
# A) multiplicative shading per metric on the selected ensemble
print("\n=== multiplicative calibration (tuned on val) ===")
val_c=[];test_c=[]
for m in range(3):
    pv=np.mean([Pv[j][ok_va,m] for j in sel_all[m]],0); pt=np.mean([Pt[j][ok_te,m] for j in sel_all[m]],0)
    ks=np.linspace(0.94,1.04,101)
    errs=[mape(pv*k,Gva[ok_va,m]) for k in ks]
    k=ks[int(np.argmin(errs))]
    val_c.append(min(errs)); test_c.append(mape(pt*k,Gte[ok_te,m]))
    print(f"{M[m]}: k={k:.3f}  VAL {val_c[-1]:.2f} TEST {test_c[-1]:.2f}")
print(f"CALIBRATED OV: VAL {np.mean(val_c):.2f} TEST {np.mean(test_c):.2f}")
# C) FFT blend for CD (and stack check for CV/HEX proxies)
F=json.load(open(f"{ROOT}/fft_morphometry2.json"))["rows"]
fva=np.array([F.get(str(i),{}).get("cd",np.nan) for i in va]); fte=np.array([F.get(str(i),{}).get("cd",np.nan) for i in te])
mva=ok_va&~np.isnan(fva); mte=ok_te&~np.isnan(fte)
pv=np.mean([Pv[j][:,0] for j in sel_all[0]],0); pt=np.mean([Pt[j][:,0] for j in sel_all[0]],0)
print("\n=== FFT-CD blend ===")
ws=np.linspace(0,0.3,31)
errs=[mape((1-w)*pv[mva]+w*fva[mva],Gva[mva,0]) for w in ws]
w=ws[int(np.argmin(errs))]
print(f"w={w:.2f}  VAL {min(errs):.2f} (ens-alone {mape(pv[mva],Gva[mva,0]):.2f})  "
      f"TEST {mape((1-w)*pt[mte]+w*fte[mte],Gte[mte,0]):.2f} (ens-alone {mape(pt[mte],Gte[mte,0]):.2f})")
json.dump({"sel":{M[m]:sel_all[m] for m in range(3)},"val":val_f,"test":test_f,
           "val_cal":val_c,"test_cal":test_c},open(f"{ROOT}/ens_ff_results.json","w"),indent=1)
