"""Bounded synthetic CUDA test of original SSL/regression loops, NOT CPSAM retraining.

Uses a tiny encoder and generated arrays, writes only to a new temporary directory.
No challenge data, release weights, old experiments, or full-data fits are used.
"""
from pathlib import Path
import tempfile
import json
import numpy as np
import train_ssl_raw


def main():
    old, shared, ssl, _ = train_ssl_raw.load_api()
    torch = old.setup_torch()
    if not torch.cuda.is_available():
        raise RuntimeError('Synthetic smoke requires CUDA')
    nn = torch.nn

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Conv2d(3, 256, 8, stride=8)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    class Views:
        def __init__(self):
            self.selected = np.zeros((64, 1, 2), dtype=int)

        def tile(self, index, origin, variant):
            return np.random.default_rng(int(index)).random((3, 256, 256), dtype=np.float32)

        def train(self, index, k=2):
            return np.stack([self.tile(index, None, 0) for _ in range(k)])

    with tempfile.TemporaryDirectory(prefix='clearec-loop-smoke-') as tmp:
        root = Path(tmp)
        shared.seed(77)
        views = Views()
        pre = root/'pretrain'
        pre.mkdir()
        result = ssl.pretrain(TinyEncoder().cuda(), views, np.arange(64), pre, 1, shared)
        from final_candidate_core import CandidateNet
        net = CandidateNet(TinyEncoder()).cuda()
        ck = torch.load(result['final_checkpoint'], map_location='cpu', weights_only=True)
        net.encoder.load_state_dict(ck['teacher_encoder'], strict=True)
        labels = np.tile(np.array([2000., .35, .55], dtype=np.float32), (64, 1))
        cfg = old.config_for(shared, 1, True)
        fitted = old.fit(shared, net, views, np.arange(64), labels, root/'regression', cfg, 1e-5)
        assert fitted['strict_reload_passed']
        print(json.dumps(dict(status='passed', ssl_steps=1, regression_steps=1,
              ssl_batch=32, regression_batch=48, synthetic_tiny_encoder=True,
              real_cpsam_training_tested=False, release_modified=False)))


if __name__ == '__main__':
    main()
