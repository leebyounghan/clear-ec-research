"""Label-free native-256 region policies; no resized global image or ROI labels."""
import numpy as np
from fullframe_views import ContextViews, dihedral

POLICIES = ("full_uniform", "full_quality", "anchor_quality")


def diverse_origins(origins, scores, count=8, anchors=()):
    """Greedy quality ranking with spatial separation, deterministic fallback."""
    origins=np.asarray(origins,dtype=np.int32); scores=np.asarray(scores)
    if len(scores)!=len(origins) or not np.isfinite(scores).all():
        raise ValueError("Invalid unlabeled quality scores")
    chosen=[]; anchor_list=[np.asarray(a) for a in anchors]
    order=np.argsort(-scores,kind="stable")
    for minimum in (192,128,0):
        for ix in order:
            point=origins[ix]
            if any(np.array_equal(point,x) for x in chosen+anchor_list):
                continue
            if any(np.linalg.norm(point-x)<minimum for x in chosen+anchor_list):
                continue
            chosen.append(point)
            if len(chosen)==count:
                return np.asarray(chosen)
    raise ValueError("Insufficient distinct crop origins")


class NativeRegionViews(ContextViews):
    def __init__(self,central,ids,policy,raw_root="datasets/clear_ec_fullframe_20260904"):
        if policy not in POLICIES:
            raise ValueError("Unknown native region policy")
        super().__init__(central,ids,scope="full",quality="focus",two_views=False,raw_root=raw_root)
        self.policy=policy
        with np.load(str(raw_root)+"/context_quality.npz") as data:
            self.scores=data["scores"].copy()
        self.central_origin=np.array([(self.height-512)//2,(self.width-512)//2],dtype=np.int32)
        self.central_c5=self.central_origin+np.array([(0,0),(0,256),(256,0),(256,256),(128,128)])
        self.selected=np.stack([diverse_origins(self.origins,s,8) for s in self.scores])

    def train_origin(self,index):
        if self.policy=="full_uniform":
            return np.array([np.random.randint(self.height-255),np.random.randint(self.width-255)])
        if self.policy=="anchor_quality" and np.random.rand()<.6:
            return self.central_origin+np.random.randint(0,257,size=2)
        origin=self.selected[index,np.random.randint(8)]+np.random.randint(-32,33,size=2)
        return np.clip(origin,[0,0],[self.height-256,self.width-256])

    def tile(self,index,origin,variant):
        row,col=map(int,origin)
        raw=self.images[index,row:row+256,col:col+256]
        if raw.shape!=(256,256):
            raise ValueError("Out-of-bounds native crop")
        image=dihedral(self.normalize(raw),variant)
        return np.stack((image,)*3)

    def train(self,index,k=2):
        return np.stack([self.tile(index,self.train_origin(index),np.random.randint(8)) for _ in range(k)])

    def eval_origins_for(self,index):
        if self.policy=="full_uniform":
            h,w=self.height-256,self.width-256
            return np.array([(0,0),(0,w),(h,0),(h,w),(h//2,w//2)])
        if self.policy=="full_quality":
            return self.selected[index,:5]
        anchors=self.central_c5[[0,3,4]]
        extra=diverse_origins(self.origins,self.scores[index],2,anchors=anchors)
        return np.concatenate((anchors,extra))

    def evaluate(self,index,d8):
        return np.stack([self.tile(index,origin,v) for origin in self.eval_origins_for(index)
                         for v in range(8 if d8 else 1)])
