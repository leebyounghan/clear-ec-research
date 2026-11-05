"""Naive direct value regression on frozen cpsam features. Split 1000 test/1000 val/7000 train."""
import numpy as np, torch, torch.nn as nn, json
ROOT="datasets/clear_ec_official"
d=np.load(f"{ROOT}/cpsam_feats.npz", allow_pickle=True)
feats,Y,ids=d["feats"],d["Y"],d["ids"]
N=len(feats); rng=np.random.default_rng(42); perm=rng.permutation(N)
te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
print(f"split: train {len(tr)} val {len(va)} test {len(te)}")
# standardize features on train
mu,sd=feats[tr].mean(0),feats[tr].std(0)+1e-6
X=(feats-mu)/sd
# scale targets by train-mean so all ~1 (MPE is scale-invariant)
ymean=Y[tr].mean(0); Yn=Y/ymean
dev="cuda"
Xt=torch.tensor(X,dtype=torch.float32,device=dev); Yt=torch.tensor(Yn,dtype=torch.float32,device=dev)
class MLP(nn.Module):
    def __init__(s,d=512):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.3),
                            nn.Linear(256,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,3))
    def forward(s,x): return s.net(x)+1.0   # bias ~1 (targets ~1)
net=MLP(X.shape[1]).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
def mpe(pred,idx):
    # pred,Yn in scaled units; MPE identical in raw units
    e=(pred-Yt[idx]).abs()/Yt[idx].clamp(min=1e-6)
    return e.mean(0)  # per-metric
trI=torch.tensor(tr,device=dev); vaI=torch.tensor(va,device=dev); teI=torch.tensor(te,device=dev)
best=1e9; best_state=None; bestep=0
for ep in range(400):
    net.train(); opt.zero_grad()
    p=net(Xt[trI]); loss=((p-Yt[trI]).abs()/Yt[trI].clamp(min=1e-6)).mean()
    loss.backward(); opt.step()
    if ep%5==0:
        net.eval()
        with torch.no_grad():
            vp=net(Xt[vaI]); vm=mpe(vp,vaI); vov=vm.mean().item()
        if vov<best: best=vov; bestep=ep; best_state={k:v.clone() for k,v in net.state_dict().items()}
net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    tp=net(Xt[teI]); tm=mpe(tp,teI).cpu().numpy()*100
    vp=net(Xt[vaI]); vm=mpe(vp,vaI).cpu().numpy()*100
print(f"\nbest val epoch {bestep}")
print(f"{'':>6} {'CD':>7} {'CV':>7} {'HEX':>7} {'OVERALL':>8}")
print(f"{'VAL':>6} {vm[0]:>6.1f}% {vm[1]:>6.1f}% {vm[2]:>6.1f}% {vm.mean():>7.1f}%")
print(f"{'TEST':>6} {tm[0]:>6.1f}% {tm[1]:>6.1f}% {tm[2]:>6.1f}% {tm.mean():>7.1f}%")
# baseline: predict train-mean (const) -> its MPE
with torch.no_grad():
    const=torch.ones_like(Yt[teI])
    cm=((const-Yt[teI]).abs()/Yt[teI].clamp(min=1e-6)).mean(0).cpu().numpy()*100
print(f"{'const':>6} {cm[0]:>6.1f}% {cm[1]:>6.1f}% {cm[2]:>6.1f}% {cm.mean():>7.1f}%  (predict train-mean)")
json.dump({"test":tm.tolist(),"val":vm.tolist(),"const":cm.tolist(),"best_epoch":bestep},
          open(f"{ROOT}/regress_results.json","w"),indent=1)
