"""Point localization contracts. ROI bounds supervise evaluation support, not cell masks."""
import numpy as np
from scipy.optimize import linear_sum_assignment

STRIDE = 8
OFFSET = 3.5  # kernel16, padding4: (16-1)/2-4
SIGMA = 8.0
MATCH_RADIUS = 12.0

def point_target(points, roi, grid_shape=(124,164)):
    points = np.asarray(points, dtype=np.float64).reshape(-1,2)
    box = np.asarray(roi, dtype=np.float64)
    assert box.shape == (4,) and np.isfinite(box).all()
    assert box[0] <= box[2] and box[1] <= box[3]
    assert len(points) and np.isfinite(points).all()
    assert ((points >= box[:2]) & (points <= box[2:])).all()
    yy, xx = np.indices(grid_shape, dtype=np.float64)
    xx = OFFSET + STRIDE*xx; yy = OFFSET + STRIDE*yy
    valid = (xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
    target = np.zeros(grid_shape, dtype=np.float64)
    for px,py in points:
        target = np.maximum(target, np.exp(-((xx-px)**2+(yy-py)**2)/(2*SIGMA**2)))
    assert valid.any()
    return target.astype(np.float32), valid

def match_points(predicted, annotated, radius=MATCH_RADIUS):
    """Maximum-cardinality radius matching, then minimum distance; no double matches."""
    p=np.asarray(predicted,dtype=np.float64).reshape(-1,2)
    a=np.asarray(annotated,dtype=np.float64).reshape(-1,2)
    assert np.isfinite(p).all() and np.isfinite(a).all() and radius>0
    if len(p) and len(a):
        d=np.linalg.norm(p[:,None]-a[None,:],axis=2)
        # Every forbidden edge costs more than all feasible distances combined.
        penalty=(min(len(p),len(a))+1)*(radius+1)
        i,j=linear_sum_assignment(np.where(d<=radius,d,penalty))
        good=d[i,j]<=radius; distances=d[i[good],j[good]]
    else: distances=np.empty(0)
    tp=len(distances); fp=len(p)-tp; fn=len(a)-tp
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(len(p),1),
                recall=tp/max(len(a),1),f1=2*tp/max(len(p)+len(a),1),
                matched_mean_distance=float(distances.mean()) if tp else None)

def cpu_tests():
    target,mask=point_target([[19.5,27.5]],[0,0,64,64],(10,10))
    assert np.unravel_index(target.argmax(),target.shape)==(3,2)
    assert target[3,2]==1 and not mask[-1,-1]
    assert match_points([[0,0],[1,0]],[[0,0]])['tp']==1
    assert match_points([],[[0,0]])['fn']==1
    assert match_points([[0,0]],[])['fp']==1
    assert match_points([[0,0]],[[13,0]])['tp']==0
    # Greedy nearest-neighbor matching would lose a true positive here.
    assert match_points([[0,0],[10,0]],[[1,0],[-10,0]])['tp']==2
    return dict(status='passed',checks=['patch_center_mapping','gaussian_peak',
        'ROI_exclusion','one_to_one_matching','empty_cases','radius_cutoff','maximum_cardinality'],
        stride=STRIDE,offset=OFFSET,sigma=SIGMA,match_radius=MATCH_RADIUS,
        cell_mask_ground_truth=False)
if __name__=='__main__':
    import json
    print(json.dumps(cpu_tests()))
