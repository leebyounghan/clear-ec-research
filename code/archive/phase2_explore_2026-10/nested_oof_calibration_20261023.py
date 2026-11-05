"""Fixed prospective log-residual calibration; no outer-label fit or tuning."""
import numpy as np
METHODS=('identity','scalar','shrinkage_power','quality_residual')

def positive(a):
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not (a>0).all():
        raise ValueError('Expected finite positive N x 3 values')
    return a

def quality(a,n):
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(n,3) or not np.isfinite(a).all():
        raise ValueError('Expected three finite image-only quality features')
    return a

def image_quality(raw):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.dtype!=np.uint8 or min(raw.shape)<3:
        raise ValueError('Native grayscale uint8 image required')
    a=raw.astype(np.float64)/255.
    lap=a[1:-1,:-2]+a[1:-1,2:]+a[:-2,1:-1]+a[2:,1:-1]-4*a[1:-1,1:-1]
    return np.array([a.mean(),a.std(),np.log1p(lap.var())])


def fit(method,pred,target,features,fit_ids,outer_ids):
    if method not in METHODS: raise ValueError('Unknown method')
    p=positive(pred);y=positive(target);q=quality(features,len(p))
    if p.shape!=y.shape or len(fit_ids)!=len(p) or len(set(fit_ids))!=len(p) or set(fit_ids)&set(outer_ids):
        raise ValueError('Calibration fit IDs overlap outer holdout or mismatch')
    if len(p)<10: raise ValueError('Insufficient calibration rows')
    # Base-model exclusion must additionally be established by nested manifests.
    z=np.concatenate([np.log(p),q],axis=1)
    center=z.mean(0);scale=z.std(0);scale=np.where(scale>1e-12,scale,1.)
    z=(z-center)/scale
    residual=np.log(y)-np.log(p);coef=[]
    for j in range(3):
        cols=[] if method in ('identity','scalar') else [j] if method=='shrinkage_power' else list(range(6))
        x=np.column_stack([np.ones(len(p)),z[:,cols]])
        # Mean squared log residual + fixed identity-centered ridge penalty.
        penalty=np.diag([.01]+[.1]*len(cols))
        w=np.zeros(x.shape[1]) if method=='identity' else np.linalg.solve(x.T@x/len(p)+penalty,x.T@residual[:,j]/len(p))
        coef.append(dict(columns=cols,weights=w.tolist()))
    return dict(schema='nested_log_residual_v1',method=method,center=center.tolist(),scale=scale.tolist(),coefficients=coef,fit_n=len(p),penalties=dict(intercept=.01,slope=.1))

def predict(model,pred,features):
    if model.get('schema')!='nested_log_residual_v1':raise ValueError('Unknown schema')
    p=positive(pred);q=quality(features,len(p))
    if model['method']=='identity':return p.copy()
    z=(np.concatenate([np.log(p),q],axis=1)-np.array(model['center']))/np.array(model['scale'])
    out=p.copy()
    for j,c in enumerate(model['coefficients']):
        x=np.column_stack([np.ones(len(p)),z[:,c['columns']]])
        out[:,j]*=np.exp(x@np.array(c['weights']))
    return positive(out)

def tests():
    import json
    np.testing.assert_array_equal(image_quality(np.zeros((5,5),dtype=np.uint8)),np.zeros(3))
    rng=np.random.default_rng(46);p=np.exp(rng.normal(size=(80,3)));q=rng.normal(size=(80,3));ids=list(range(80));outer=list(range(80,100))
    for method in METHODS:
        m=fit(method,p,p,q,ids,outer);np.testing.assert_allclose(predict(m,p,q),p,rtol=1e-12)
        mm=json.loads(json.dumps(m));np.testing.assert_array_equal(predict(m,p,q),predict(mm,p,q))
        changed=q.copy();changed[-1]+=100
        # Prediction calls cannot mutate train-only normalization or coefficients.
        before=json.dumps(m,sort_keys=True);predict(m,p,changed);assert json.dumps(m,sort_keys=True)==before
    m=fit('scalar',p,2*p,q,ids,outer)
    np.testing.assert_allclose(predict(m,p,q)/p,np.exp(np.log(2)/1.01),rtol=1e-12)
    for bad_ids,bad_target in [(ids[:-1],p),(ids,p*0),(ids,np.full_like(p,np.nan)),(list(range(1,81)),p)]:
        try:fit('scalar',p,bad_target,q,bad_ids,outer)
        except ValueError:pass
        else:raise AssertionError('Invalid or leaking fit accepted')
    return dict(status='PASS',methods=list(METHODS),identity_mapping=True,serialization_exact=True,train_normalization_immutable=True,known_scalar_solution=True,invalid_fit_rejections=4)

if __name__=='__main__':
    import json
    print(json.dumps(tests()))
