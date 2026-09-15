"""Standalone adapter for the released sixth-member fit; no old queue manifests.

Reuses original pretraining and regression functions without changing their loops.
Full training remains explicit and never replaces released checkpoints.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def load_api():
    # Absolute repository source, not source folders from past experiment runs.
    sys.path.insert(0, str(REPO / 'code'))
    import next_candidate_20260913 as legacy
    import train_specialize_screen as shared
    import ssl_native_core as ssl
    import train_deploy_region as regions
    for module in (legacy, shared, ssl, regions):
        if Path(module.__file__).resolve().parent != REPO / 'code':
            raise RuntimeError('Unexpected source module: ' + module.__file__)
    return legacy, shared, ssl, regions


def source_inventory():
    names = ('next_candidate_20260913.py', 'train_specialize_screen.py',
             'ssl_native_core.py', 'train_deploy_region.py', 'final_candidate_core.py',
             'method_training_core.py', 'native_region_views.py', 'fullframe_views.py')
    return {name: hashlib.sha256((REPO/'code'/name).read_bytes()).hexdigest() for name in names}


def train(workspace, output, smoke=False):
    if output.exists() or output.is_symlink():
        raise FileExistsError('Use a new output directory; never overwrite a run or release')
    if not (workspace/'data_ready.json').is_file():
        raise ValueError('Run workspace.py init and prepare first')
    foundation = workspace/'cpsam_models/cpsam_v2'
    from workspace import FOUNDATION_SHA, sha
    if sha(foundation) != FOUNDATION_SHA:
        raise ValueError('Foundation checksum mismatch')
    os.chdir(workspace)
    old, shared, ssl, regions = load_api()
    torch = old.setup_torch()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training requires a BF16-capable NVIDIA GPU')
    import numpy as np
    with np.load('datasets/clear_ec_final9000_20260904/meta.npz') as d:
        ids, labels, indices = d['ids'].copy(), d['Y'].copy(), d['train_indices'].copy()
    if len(ids) != 9000 or len(indices) != 8998 or not np.isfinite(labels).all():
        raise ValueError('Expected 9000 images and 8998 valid supervised rows')
    np.testing.assert_array_equal(indices, np.flatnonzero((labels > 0).all(1)))
    output.mkdir(parents=True, exist_ok=False)
    old.save(output/'adapter_manifest.json', dict(source_sha256=source_inventory(),
        smoke=smoke, historical_queue_artifacts_required=False,
        bitwise_retraining_certified=False, released_checkpoints_modified=False))
    shared.seed(77)
    views = regions.build_regions('full_quality')
    pre = output/'pretrain'
    pre.mkdir()
    net = shared.build_model('cpsam_models/cpsam_v2')
    seen = set()
    original = views.train
    def recording_train(index, k=2):
        seen.add(int(index))
        return original(index, k)
    views.train = recording_train
    result = ssl.pretrain(net.encoder, views, np.arange(9000, dtype=np.int64),
                          pre, 4 if smoke else 6000, shared)
    views.train = original
    if not smoke and len(seen) != 9000:
        raise RuntimeError('Full SSL did not sample all 9000 images')
    old.save(pre/'coverage.json', dict(eligible_images=9000, unique_images_seen=len(seen), labels_used=False))
    del net
    gc.collect()
    torch.cuda.empty_cache()
    net = old.fresh_net(shared, result['final_checkpoint'])
    cfg = old.config_for(shared, 2 if smoke else 5215, smoke, train_rows=8998,
        validation_rows=0, encoder_lr=1e-5, head_lr=1e-3,
        initialization='standard_SSL_all9000_terminal_EMA', ssl_images=9000,
        ssl_updates=4 if smoke else 6000, ssl_checkpoint_sha256=result['checkpoint_sha256'],
        full_data_final_fit=True, independent_validation=False)
    old.fit(shared, net, views, indices, labels, output/'raw_deploy', cfg, 1e-5, None, ids)
    print('Fit complete. Rerun checkpoint is NOT automatically packaged or submitted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-imports', action='store_true')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true', help='4 SSL + 2 regression steps; original batch sizes')
    args = parser.parse_args()
    if args.check_imports:
        load_api()
        print(json.dumps(dict(status='passed', source_sha256=source_inventory(), training_started=False)))
        return
    if args.workspace is None or args.output is None:
        parser.error('--workspace and --output are required for training')
    if args.output.is_symlink():
        raise ValueError('Output cannot be a symlink')
    train(args.workspace.resolve(strict=True), args.output.absolute(), args.smoke)


if __name__ == '__main__':
    main()
