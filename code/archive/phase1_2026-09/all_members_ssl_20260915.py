"""Two bounded queues: original five recipes with one shared all9000 SSL encoder.

No architecture/label/crop/loss changes, no OOF claim, no submission or packaging.
Original trainers are copied into this experiment's frozen source directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('experiments/all_members_ssl_20260915')
RELEASE = Path('experiments/ssl_release_20260914')
ENCODER = RELEASE / 'pretrain/encoder_step6000.pt'
ENCODER_SHA = '971bcaba9f66cbf9466b6e4bfd3e5e26f593ef56384dee1a2368133647bded32'
DATA = Path('datasets/clear_ec_final9000_20260904')
QUEUES = {0: ['predbag2_35', 'seed77'], 1: ['metric35', 'seed42', 'seed2026']}
SELECTED = {'seed42': 3576, 'seed77': 2831, 'seed2026': 2980,
            'metric35': 4470, 'predbag2_35': 3874}
SOURCES = ['all_members_ssl_20260915.py', 'train_final_candidate.py',
           'train_deploy_methods.py', 'final_candidate_core.py', 'method_training_core.py']


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    m = load(ROOT / 'manifest.json')
    for group in ['source_sha256', 'input_sha256']:
        for p, expected in m[group].items():
            assert sha(p) == expected, p
    assert sha(ENCODER) == ENCODER_SHA
    return m


def head_sha(net):
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        if not name.startswith('encoder.'):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def inject_ssl(net):
    import torch
    before = head_sha(net)
    rng = torch.get_rng_state().clone()
    ck = torch.load(ENCODER, map_location='cpu', weights_only=True)
    assert ck['step'] == 6000 and ck['config']['validation_images_used'] == 0
    assert set(net.encoder.state_dict()) == set(ck['teacher_encoder'])
    net.encoder.load_state_dict(ck['teacher_encoder'], strict=True)
    assert all(torch.equal(value.detach().cpu(), ck['teacher_encoder'][name])
               for name, value in net.encoder.state_dict().items())
    assert head_sha(net) == before and torch.equal(torch.get_rng_state(), rng)
    return net, before


def prepare():
    import numpy as np
    import torch
    import final_candidate_core as core
    assert not (ROOT / 'manifest.json').exists()
    assert not load(RELEASE / 'gpu_complete.json')['failed']
    assert load(RELEASE / 'pretrain/coverage.json') == dict(
        eligible_images=9000, unique_images_seen=9000, labels_used=False)
    assert sha(ENCODER) == ENCODER_SHA
    with np.load(DATA / 'meta.npz') as d:
        tr = d['train_indices']; y = d['Y_train']
        assert len(tr) == len(np.unique(tr)) == 8998
        assert len(d['ids']) == 9000 and np.isfinite(y[tr]).all() and (y[tr] > 0).all()
    assert (DATA / 'img_native_512.dat').stat().st_size == 9000 * 512 * 512
    torch.manual_seed(77)
    net, heads = inject_ssl(core.build_model('cpsam_models/cpsam_v2', device='cpu'))
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())
    del net
    source_hashes = {str(ROOT / 'source' / n): sha(ROOT / 'source' / n) for n in SOURCES}
    inputs = [DATA / 'meta.npz', DATA / 'img_native_512.dat',
              Path('cpsam_models/cpsam_v2'), ENCODER,
              RELEASE / 'manifest.json', RELEASE / 'pretrain/coverage.json',
              RELEASE / 'stage_manifest.json', RELEASE / 'gpu_complete.json']
    save(ROOT / 'plan.json', dict(selected_steps={str(s): SELECTED[f'seed{s}'] for s in [42,77,2026]}))
    inputs.append(ROOT / 'plan.json')
    m = dict(created_at=time.strftime('%FT%T%z'), source_sha256=source_hashes,
        input_sha256={str(p): sha(p) for p in inputs}, queues=QUEUES,
        selected_steps=SELECTED, ssl_encoder=str(ENCODER), ssl_sha256=ENCODER_SHA,
        ssl_unique_images=9000, ssl_updates=6000, train_rows=8998,
        target_policy='original_first5_slide_smoothed_Y_train_preserved',
        crop_policy='original_central512_random_native256_preserved',
        initialization_only_change=True, backbone_architecture_unchanged=True,
        ssl_independent_pretrains=1, fresh_heads_each_member=True,
        sixth_preserved=str(RELEASE / 'model/full_quality_seed77.pt'),
        independent_validation=False, performance_gain_not_established=True,
        no_packaging_or_submission=True, no_calibration_fit=True,
        maximum_new_gpu_jobs=2, maximum_concurrent_new_gpus=2,
        queue_limit_seconds=23400, maximum_runtime_gpu_hours=13)
    save(ROOT / 'manifest.json', m)
    save(ROOT / 'preflight.json', dict(passed=True, strict_encoder_load=True,
         fresh_head_sha_preserved=heads, cpu_rng_preserved=True, all_parameters_finite=True,
         tested_on='K8s CPU', at=time.strftime('%FT%T%z')))
    print(json.dumps(dict(preflight_passed=True, manifest_sha256=sha(ROOT/'manifest.json'))), flush=True)


def worker(member, smoke):
    import numpy as np
    import torch
    import final_candidate_core as core
    m = verify()
    module_name = 'train_final_candidate' if member.startswith('seed') else 'train_deploy_methods'
    mod = __import__(module_name)
    mod.ROOT = ROOT
    run = ROOT / ('smoke' if smoke else 'runs') / member
    assert not run.exists(), 'No automatic retry or overwrite'
    original_builder = mod.build_model
    constructor_audit = {}

    def build(*args, **kwargs):
        net, heads = inject_ssl(original_builder(*args, **kwargs))
        constructor_audit.update(fresh_head_sha256=heads, strict_encoder_load=True,
                                 encoder_sha256=ENCODER_SHA)
        return net

    mod.build_model = build
    original_save = mod.save_json

    def annotated_save(path, value):
        value = dict(value)
        if Path(path).name == 'config.json':
            value.update(initialization='all9000_standard_masked_EMA_SSL_encoder',
                ssl_encoder_sha256=ENCODER_SHA, ssl_images=9000, ssl_updates=6000,
                target_policy=m['target_policy'], crop_policy=m['crop_policy'],
                selection='inherited_original_precommitted_steps_not_retuned_for_SSL',
                independent_validation=False, source_sha256=m['source_sha256'],
                constructor_audit=constructor_audit)
        original_save(path, value)

    mod.save_json = annotated_save
    sys.argv = [module_name]
    if member.startswith('seed'):
        sys.argv += ['--seed', member.removeprefix('seed')]
    else:
        sys.argv += ['--recipe', member, '--output-root', str(ROOT)]
    if smoke:
        sys.argv.append('--smoke')
    mod.main()
    torch.cuda.empty_cache()
    ck = torch.load(run / 'selected.pt', map_location='cpu', weights_only=True)
    assert ck['step'] == (1 if smoke else SELECTED[member])
    net = core.build_model('cpsam_models/cpsam_v2')
    net.load_state_dict(ck['model'], strict=True)
    net.eval()
    images = np.memmap(DATA / 'img_native_512.dat', dtype='uint8', mode='r', shape=(9000,512,512))
    with torch.inference_mode():
        for index in [0, 1285, 5142, 8999]:
            pred = core.predict_one(net, images[index])
            assert pred.shape == (3,) and np.isfinite(pred).all() and (pred > 0).all()
    save(run / 'verified.json', dict(member=member, smoke=smoke,
        selected_steps=ck['step'], checkpoint_sha256=sha(run/'selected.pt'),
        strict_reload_passed=True, finite_positive_four_image_40view=True,
        fresh_heads_and_exact_ssl_encoder=constructor_audit,
        accuracy_evaluation=False, official_runtime_test=False,
        at=time.strftime('%FT%T%z')))


def run_queue(queue):
    verify()
    out = ROOT / f'queue{queue}'
    out.mkdir(exist_ok=False)
    start = time.monotonic(); active = None; done = []

    def stop(sig, frame):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
        save(out/'failure.json', dict(signal=sig, done=done, at=time.strftime('%FT%T%z')))
        raise SystemExit(128+sig)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        for member in QUEUES[queue]:
            for smoke in [True, False]:
                remaining = 23400 - (time.monotonic()-start)
                assert remaining > 0, 'Queue budget exhausted'
                save(out/'status.json', dict(member=member, smoke=smoke, done=done,
                     at=time.strftime('%FT%T%z')))
                command = [sys.executable, str(ROOT/'source/all_members_ssl_20260915.py'),
                           'worker', '--member', member] + (['--smoke'] if smoke else [])
                with (out/f'{member}_{"smoke" if smoke else "full"}.log').open('x') as stream:
                    active = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    active.wait(timeout=min(900 if smoke else 14400, remaining))
                assert active.returncode == 0, f'{member} smoke={smoke} exit={active.returncode}'
                assert load(ROOT/('smoke' if smoke else 'runs')/member/'verified.json')['strict_reload_passed']
            done.append(member)
        save(out/'complete.json', dict(done=done, elapsed_seconds=time.monotonic()-start,
                                       at=time.strftime('%FT%T%z')))
    except Exception as exc:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
        save(out/'failure.json', dict(error=repr(exc), done=done, at=time.strftime('%FT%T%z')))
        raise


def audit():
    m = verify()
    assert not (ROOT/'completion.json').exists()
    assert all((ROOT/f'queue{q}/complete.json').exists() and not (ROOT/f'queue{q}/failure.json').exists() for q in QUEUES)
    records = {}
    for member, step in SELECTED.items():
        v = load(ROOT/'runs'/member/'verified.json')
        assert v['selected_steps'] == step and v['strict_reload_passed']
        assert sha(ROOT/'runs'/member/'selected.pt') == v['checkpoint_sha256']
        records[member] = v
    save(ROOT/'completion.json', dict(models=records, all_five_ssl_retrained=True,
        sixth_preserved=m['sixth_preserved'], no_internal_validation_score=True,
        no_submission_package_created=True, at=time.strftime('%FT%T%z')))
    print(json.dumps(dict(completed_members=list(records), accuracy_gain_not_established=True)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare','worker','run','audit'])
    p.add_argument('--member', choices=SELECTED)
    p.add_argument('--queue', type=int, choices=QUEUES)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.command == 'worker': worker(a.member, a.smoke)
    elif a.command == 'run': run_queue(a.queue)
    else: globals()[a.command]()
