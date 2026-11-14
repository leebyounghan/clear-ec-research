"""PyTorch port of the Vigueras-Guillen TVST 2020 Dense U-Net (CNN-Edge and CNN-ROI) (HEX lane, 2026-11-11).
Written from the architecture in the authors' notebook (TVST_paper_testing.ipynb, get_unet): growth 8, blocks
(6, 12, 24, 48, 24, 12, 6), 4x4 'same' convs (TF padding: 1 before, 2 after), BN eps 1.001e-5 at inference (moving
statistics; renorm only changes training), ELU, average-pool downsampling with the input pyramid concatenated, 4x4 stride-2
'same' transposed convs (= PyTorch padding 1), final 1x1 conv + ELU + softmax over 2 classes; class 1 = edge (CNN-Edge) or
ROI (CNN-ROI). Input: intensity / 255 (CNN-Edge) or the edge probability (CNN-ROI), (B, 1, H, W), H and W multiples of 8.
Weights: vig_convert.py output, loaded with torch.load(weights_only=True).
usage: vig_net.py --verify   (runs both nets on the authors' example image and compares with their saved outputs)"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
WDIR = LANE + '/deploy/vigueras'
EX = '/workspace/clear-ec-research/datasets/external_20261106/vigueras_tvst_weights/src/images/2016-23_Patient013_m12_CE_i02_'
BLOCKS = (6, 12, 24, 48, 24, 12, 6)


def load(name, dev='cuda'):
    W = torch.load(f'{WDIR}/{name}.pt', map_location='cpu', weights_only=True)
    fin = [k.rsplit('.', 1)[0] for k in W if not k.startswith(('block_', 'down_', 'up_')) and k.endswith('.kernel')]
    assert len(fin) == 1, fin
    W = {k: v.to(dev) for k, v in W.items()}; W['_final'] = fin[0]; return W


def _bn(x, W, n, eps=1.001e-5):
    g, b, m, v = (W[f'{n}.{k}'] for k in ('gamma', 'beta', 'moving_mean', 'moving_variance'))
    return (x - m[:, None, None]) * (g / torch.sqrt(v + eps))[:, None, None] + b[:, None, None]


def forward(W, x):
    def dense(x, name, n):
        for i in range(n):
            nm = f'{name}_block{i + 1}'
            x1 = F.elu(_bn(F.conv2d(F.pad(x, (1, 2, 1, 2)), W[nm + '_conv.kernel']), W, nm + '_bn'))
            x = torch.cat([x, x1], 1)
        return x

    def down(x, name): return F.avg_pool2d(F.elu(_bn(F.conv2d(x, W[name + '_conv.kernel']), W, name + '_bn')), 2)

    def up(x, name): return F.elu(_bn(F.conv_transpose2d(x, W[name + '_conv.kernel'], stride=2, padding=1), W, name + '_bn'))

    assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0, x.shape
    x1 = dense(x, 'block_1', BLOCKS[0]); x1d = down(x1, 'down_1')
    x2i = F.avg_pool2d(x, 2); x2 = dense(torch.cat([x2i, x1d], 1), 'block_2', BLOCKS[1]); x2d = down(x2, 'down_2')
    x3i = F.avg_pool2d(x2i, 2); x3 = dense(torch.cat([x3i, x2d], 1), 'block_3', BLOCKS[2]); x3d = down(x3, 'down_3')
    x4i = F.avg_pool2d(x3i, 2); x4 = dense(torch.cat([x4i, x3d], 1), 'block_4', BLOCKS[3]); u4 = up(x4, 'up_4')
    x5 = dense(torch.cat([x3, u4], 1), 'block_5', BLOCKS[4]); u5 = up(x5, 'up_5')
    x6 = dense(torch.cat([x2, u5], 1), 'block_6', BLOCKS[5]); u6 = up(x6, 'up_6')
    x7 = dense(torch.cat([x1, u6], 1), 'block_7', BLOCKS[6])
    f = W['_final']; z = F.elu(F.conv2d(x7, W[f + '.kernel'], W[f + '.bias']))
    return torch.softmax(z, 1)[:, 1]


@torch.no_grad()
def run(W, imgs, batch=4):
    """imgs: float32 array (N, H, W) -> class-1 probability (N, H, W) float32"""
    out = np.zeros(imgs.shape, np.float32); dev = W[W['_final'] + '.kernel'].device
    for s in range(0, len(imgs), batch):
        out[s:s + batch] = forward(W, torch.from_numpy(imgs[s:s + batch, None]).to(dev)).cpu().numpy()
    return out


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    if '--verify' in sys.argv:
        import cv2
        im = cv2.imread(EX + 'Intn.tif', cv2.IMREAD_UNCHANGED); ref_e = cv2.imread(EX + 'NN_Edges.tif', cv2.IMREAD_UNCHANGED); ref_r = cv2.imread(EX + 'NN_ROI.tif', cv2.IMREAD_UNCHANGED)
        print('example', im.shape, im.dtype, 'refs', ref_e.shape, ref_e.dtype, ref_r.dtype, flush=True)
        E = run(load('tvst_edge'), (im.astype(np.float32) / 255)[None])[0]; R = run(load('tvst_roi'), E[None])[0]
        rep = {}
        for nm, P, ref in (('edge', E, ref_e), ('roi', R, ref_r)):
            ref = ref.astype(np.float64)
            for how, q in (('round', np.round(P * 255)), ('floor', np.floor(P * 255))):
                d = np.abs(q - ref); rep[f'{nm}_{how}'] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()), frac_gt1=float((d > 1).mean()))
            rep[f'{nm}_corr'] = float(np.corrcoef(P.ravel(), ref.ravel())[0, 1])
        print(json.dumps(rep, indent=1), flush=True); json.dump(rep, open(LANE + '/logs/vig_verify.json', 'w'), indent=1)
