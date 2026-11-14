"""Recipe A regression from an SSL arm of this lane (GPU 1, 2026-11-06). The frozen std-pooling runner on one GPU through lane GPU 0's wrapper; only the SSL evidence check changes, because arms do not use the frozen SSL runner.
make_input <arm> writes <arm>/regression_input from run/last.pt and the arm's config, in the shape the frozen runner reads. Everything else as the frozen runner, with --ssl-root pointing at that folder."""
import hashlib, json, sys
from pathlib import Path

import torch

import phase2_pooling_regression_20261019 as R
import recipeA_single_gpu_20261106  # noqa: F401, patches R for one GPU
import phase2_lejepa_regression_20261003 as prior

RAW_SHA = '93e67ac6df5887a1a5396c85e236b8398297dd3e728cdb6513238c1628c1c88a'
_SOURCES = R.source_files


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_input(arm):
    root = Path('experiments/lane_gpu1_20261106/ssl') / arm; d = root / 'regression_input'; (d / 'run').mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / 'config.json').read_text()); c = torch.load(root / 'run/last.pt', map_location='cpu', weights_only=True)
    m = dict(arm=arm, fold=cfg['fold'], seed=cfg['seed'], config=cfg, raw_sha256=RAW_SHA, raw_bytes=8954 * 972 * 1296, note='adapter for the frozen regression runner')
    (d / 'manifest.json').write_text(json.dumps(m, indent=1))
    torch.save(dict(encoder=c['encoder'], epoch=5, step=c['step'], arm_epochs=c['epoch'], manifest_sha=sha(d / 'manifest.json')), d / 'run/last.pt')
    r = dict(steps=c['step'], epochs=c['epoch'], manifest_sha=sha(d / 'manifest.json'), checkpoint_sha=sha(d / 'run/last.pt'), validation_images_used=0)
    (d / 'run/result.json').write_text(json.dumps(r, indent=1))
    (d / 'run/complete.json').write_text(json.dumps(dict(status='completed', result_sha=sha(d / 'run/result.json'))))
    print('input ready', d, flush=True)


def ssl_evidence(ids, tr, va):
    sm = R.read(R.SSL / 'manifest.json')
    assert sm['fold'] == R.ARGS.fold and sm['config']['train_n'] == len(tr) and sm['config']['val_n'] == len(va)
    sr = prior.validate_result(R.SSL / 'run', sha(R.SSL / 'manifest.json'))
    assert sr['validation_images_used'] == 0
    c = torch.load(R.SSL / 'run/last.pt', map_location='cpu', weights_only=True); s = prior.state_sha(c['encoder']); del c
    return sm, sr, s


R.ssl_evidence = ssl_evidence
R.source_files = lambda: _SOURCES() + [Path(__file__).resolve()]
if __name__ == '__main__':
    if sys.argv[1] == 'make_input':
        make_input(sys.argv[2])
    else:
        R.main()
