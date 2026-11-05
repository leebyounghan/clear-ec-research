import json
X='/workspace/clear-ec-research/experiments/'
R=dict(recipeA=X+'phase2_pooling_replication_20261026/f0_s46_std',enc_lr5x=X+'meta_ceiling_20261103/bridge2/f0_enc_lr5x',token_mlp=X+'meta_ceiling_20261103/bridge2/f0_token_mlp')
for n,r in R.items():
    h=json.load(open(r+'/runs/fold0/cpdino_vitb/history.json'))
    for e in h:
        if e['epoch'] in (5,10,15,20):
            t=e['train_probe_score']; v=e['score']
            print(n,e['epoch'],'train',{k:round(t[k],2) for k in ('CD','CV','HEX','mean')},'val',{k:round(v[k],2) for k in ('CD','CV','HEX','mean')},'loss',round(e['online_train_loss'],4),'enc_lr',round(e['encoder_lr'],7))
