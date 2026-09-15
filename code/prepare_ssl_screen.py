"""CPU-only provenance and split audit; exports aggregates only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('experiments/ssl_screen_20260911')
INITIAL = '0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def main():
    pre = ROOT/'preflight'; pre.mkdir(exist_ok=False)
    assert sha('cpsam_models/cpsam_v2') == INITIAL
    prior = json.loads(Path('experiments/crop_dino_screen_20260910/manifest.json').read_text())
    for path, digest in prior['source_sha256'].items():
        assert sha(path) == digest, path
    inputs = {}
    old_audit = Path('experiments/crop_dino_screen_20260910/screen_audit.json')
    audit = json.loads(old_audit.read_text())
    # Frozen audit covers aligned OOF, splits and exact control checkpoint bytes.
    for path, digest in audit['input_sha256'].items():
        assert sha(path) == digest, path
        inputs[path] = digest
    inputs[str(old_audit)] = sha(old_audit)
    data = Path('datasets/clear_ec_official')
    with np.load(data/'cache_native_meta_512.npz', allow_pickle=True) as metadata:
        ids = metadata['ids'].astype(str)
    assert len(ids) == len(set(ids)) == 8954
    with (data/'final_train_ids.csv').open() as f:
        rows = list(csv.DictReader(f))
    clean = lambda x: Path(str(x)).stem
    mapping = {clean(r['ID']): str(r['slide_id']) for r in rows}
    slides = np.array([mapping[clean(i)] for i in ids])
    raw = Path('datasets/clear_ec_fullframe_20260904')
    frame_manifest = json.loads((raw/'fullframe_manifest.json').read_text())
    assert frame_manifest['central_pixels_exact_match']
    with np.load(raw/'fullframe_meta.npz', allow_pickle=True) as metadata:
        np.testing.assert_array_equal(ids, metadata['ids'].astype(str))
    shape = tuple(frame_manifest['cache_shape'])
    assert shape[0] == len(ids)
    images = np.memmap(raw/'img_full_raw.dat', dtype=np.uint8, mode='r', shape=shape)
    pixel_hashes = []
    start = time.monotonic()
    for index in range(len(ids)):
        pixel_hashes.append(hashlib.sha256(images[index].tobytes()).hexdigest())
        if (index+1) % 2000 == 0:
            print(json.dumps(dict(stage='exact_duplicate_audit', images=index+1,
                                  seconds=time.monotonic()-start)), flush=True)
    # Merge slide groups connected by identical full images (including transitive links).
    parents = {slide:slide for slide in slides}
    def find(slide):
        while parents[slide] != slide:
            parents[slide] = parents[parents[slide]]
            slide = parents[slide]
        return slide
    seen = {}
    for slide,digest in zip(slides,pixel_hashes):
        if digest in seen:
            parents[find(slide)] = find(seen[digest])
        else:
            seen[digest] = slide
    connected = np.array([find(slide) for slide in slides])
    records = {}
    for fold in (0, 1):
        control = Path('experiments/crop_dino_screen_20260910/runs')/f'control_fold{fold}'
        cfg = json.loads((control/'config.json').read_text())
        result = json.loads((control/'result.json').read_text())
        assert cfg['policy'] == 'control' and not cfg['smoke']
        assert cfg['seed'] == 77 and cfg['updates'] == 5215
        assert cfg['batch_size'] == 48 and cfg['k_crops'] == 2
        assert cfg['initial_sha256'] == INITIAL and result['strict_reload_passed']
        with np.load(control/'split_indices.npz') as split:
            tr, va = split['train'], split['validation']
        assert len(va) == 1791 and not set(tr)&set(va)
        assert not set(slides[tr])&set(slides[va])
        historic_tr = tr.copy()
        bad_groups = set(connected[tr])&set(connected[va])
        tr = tr[~np.isin(connected[tr],list(bad_groups))]
        primary_validation = va[~np.isin(connected[va],list(bad_groups))]
        assert not set(connected[tr])&set(connected[va])
        assert not {pixel_hashes[i] for i in tr}&{pixel_hashes[i] for i in va}
        old = Path('experiments/region_seedcheck_20260906/runs')/f'full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth'
        assert sha(old/'split_indices.npz') == cfg['split_sha256']
        with np.load(old/'split_indices.npz') as split:
            np.testing.assert_array_equal(historic_tr, split['train'])
            np.testing.assert_array_equal(va, split['validation'])
        np.savez_compressed(pre/f'fold{fold}.npz', train=tr, validation=va,
                            primary_validation=primary_validation, historical_train=historic_tr)
        inputs[str(pre/f'fold{fold}.npz')] = sha(pre/f'fold{fold}.npz')
        records[str(fold)] = dict(train_rows=len(tr), validation_rows=len(va),
            historical_split_sha256=cfg['split_sha256'], split_sha256=sha(pre/f'fold{fold}.npz'),
            removed_train_rows=len(historic_tr)-len(tr), primary_validation_rows=len(primary_validation),
            excluded_primary_validation_rows=len(va)-len(primary_validation),
            slide_overlap=0, exact_pixel_duplicate_overlap=0,
            control_sample_sha256=result['training']['first_three_batches_sha256'],
            control_training_seconds=result['training']['elapsed_seconds'])
    for path in (data/'cache_native_meta_512.npz', data/'final_train_ids.csv',
                 raw/'fullframe_manifest.json', raw/'fullframe_meta.npz',
                 raw/'context_quality.npz', raw/'context_views_manifest.json'):
        inputs[str(path)] = sha(path)
    np.savez_compressed(pre/'pixel_fingerprints.npz', sha256=np.array(pixel_hashes))
    source_dirs = (ROOT/'source', Path('experiments/crop_dino_screen_20260910/source'),
                   Path('experiments/specialize_screen_20260908/source/frozen'),
                   Path('experiments/region_search_20260905/source/code'))
    sources = {str(p): sha(p) for directory in source_dirs for p in directory.rglob('*.py')}
    manifest = dict(created_at=time.strftime('%FT%T%z'), initial_sha256=INITIAL,
        source_sha256=sources, input_sha256=inputs, folds=records, requested_gpus=4,
        ssl_steps=6000, ssl_batch=32, downstream_epochs=35, extended_control_epochs=55,
        max_queue_seconds=11*3600, max_worker_seconds=10*3600, smoke_timeout_seconds=1200,
        primary_blend='replace6=(5*current5+candidate)/6', minimum_improvement_points=.02,
        gate='On duplicate-connected-group-clean primary validation: SSL replace6 improves current6 BOTH folds and aggregate>=0.02; beats NEW clean35 and continued55 replace6 BOTH folds',
        control_recipe='fresh clean35, then20 additional epochs with reset optimizer/cosine and encoder LR3e-6',
        historical_control_reused_for_scoring=False,
        no_external_data=True, no_new_pretrained_weights=True, no_resize=True,
        grouping='slide_id; patient identity not established; near duplicates not exhaustively audited',
        auto_retry=False, deployment_approved=False, confirmation_authorized=False,
        historic_validation_reuse=True, data_archive_sha256=frame_manifest['archive_sha256'])
    save(ROOT/'manifest.json', manifest)
    print(json.dumps(dict(preflight_passed=True, folds=records, requested_gpus=4,
                          source_files=len(sources), protected_inputs=len(inputs))), flush=True)


if __name__ == '__main__':
    main()
