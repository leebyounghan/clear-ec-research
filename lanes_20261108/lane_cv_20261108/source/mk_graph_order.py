import re
src = open('source/graph_train_cv_20261108.py').read()
hdr = '"""Order-loss copy (2026-11-10, CV lane, label-noise direction version 3): --order W adds W x pairwise BCE on the CV column (noise-weighted order loss, s = 0.065, lee2026stochorder; same formula as family_cv_20261108.py CD_LOSS=order) to the L1 residual loss; GRAPH_DEV=cpu runs on CPU. Otherwise identical to graph_train_cv_20261108.py.\n'
assert src.startswith('"""')
src = hdr + src[3:]
src = src.replace("import torch\n", "import torch, math, os\nDEV = os.environ.get('GRAPH_DEV', 'cuda')\n", 1)
n = src.count('.cuda()'); src = src.replace('.cuda()', '.to(DEV)')
src = src.replace("ap.add_argument('name');", "ap.add_argument('name'); ap.add_argument('--order', type=float, default=0.0);", 1)
old = "loss = ((p - Tt[ri]).abs() * Mt[ri]).sum() / Mt[ri].sum();"
assert src.count(old) == 1
new = "loss = ((p - Tt[ri]).abs() * Mt[ri]).sum() / Mt[ri].sum() + A_.order * order_cv(p, ri);"
src = src.replace(old, new)
fn = '''

LHt = torch.from_numpy(lH.astype(np.float32)).to(DEV)


def order_cv(p, ri):
    if A_.order == 0: return p.sum() * 0
    m = Mt[ri, 1] > 0
    if int(m.sum()) < 2: return p.sum() * 0
    lp = (LHt[ri, 1] + p[:, 1])[m]; lt = (LHt[ri, 1] + Tt[ri, 1])[m]
    dp = lp[:, None] - lp[None, :]; dt = lt[:, None] - lt[None, :]; q = 0.5 * (1 + torch.erf(dt / (2 * 0.065))); off = ~torch.eye(len(lp), dtype=torch.bool, device=lp.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(dp[off] / (math.sqrt(2) * 0.065), q[off])


def run('''
assert src.count('\n\ndef run(') == 1
src = src.replace('\n\ndef run(', fn, 1)
src = src.replace("torch.set_num_threads(2)", "torch.set_num_threads(int(os.environ.get('GRAPH_THREADS', '2')))")
open('source/graph_order_cv_20261110.py', 'w').write(src); print('cuda replaced', n)
