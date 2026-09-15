"""Create an isolated training workspace from Git-tracked source and explicit inputs.

No Kubernetes, historical run outputs, automatic training, or checkpoint replacement.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / 'review_submission_20260910/training'
FOUNDATION_SHA = '0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_inputs(raw_dir, labels):
    images = sorted(raw_dir.glob('*.mha'))
    with labels.open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'ID', 'slide_id', 'CD', 'CV', 'HEX'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires ID, slide_id, CD, CV, HEX')
        rows = list(reader)
    image_ids = {p.stem for p in images}
    label_ids = {r['ID'] for r in rows}
    if len(images) != 9000 or len(rows) != 9000 or len(label_ids) != 9000:
        raise ValueError('Expected 9000 original MHA images and 9000 unique CSV rows')
    if len(image_ids & label_ids) != 8954:
        raise ValueError('Expected 8954 exact original IDs; do not use the reconciled CSV here')
    return len(images)


def initialize(workspace, raw_dir, labels, foundation):
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError('Choose a new workspace; existing paths are never replaced')
    validate_inputs(raw_dir, labels)
    if sha(foundation) != FOUNDATION_SHA:
        raise ValueError('Foundation does not match the released cpsam_v2')
    shutil.copytree(FROZEN, workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    official = workspace / 'datasets/clear_ec_official'
    official.mkdir(parents=True)
    (official / 'train_mha').symlink_to(raw_dir, target_is_directory=True)
    shutil.copyfile(labels, official / 'final_train_ids.csv')
    (workspace / 'cpsam_models').mkdir()
    (workspace / 'cpsam_models/cpsam_v2').symlink_to(foundation)
    sources = {str(p.relative_to(workspace)): sha(p)
               for p in workspace.rglob('*') if p.is_file() and not p.is_symlink()
               and p.suffix in ('.py', '.json')}
    record = dict(raw_dir=str(raw_dir), original_labels_sha256=sha(labels),
                  foundation_sha256=FOUNDATION_SHA, sources=sources,
                  historical_run_artifacts_required=False, training_started=False)
    with (workspace / 'workspace.json').open('x') as f:
        json.dump(record, f, indent=2)
    print('Workspace initialized. No data caches or training runs started.')


def prepare(workspace, archive, expected):
    record = json.loads((workspace / 'workspace.json').read_text())
    if archive.name.endswith(('.part', '.partial')):
        raise ValueError('Use a completed archive, not a partial download')
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected.lower()):
        raise ValueError('Provide the independently recorded raw-archive SHA256')
    if sha(archive) != expected.lower():
        raise ValueError('Raw archive SHA256 mismatch')
    for name, digest in record['sources'].items():
        if sha(workspace / name) != digest:
            raise ValueError('Workspace source changed: ' + name)
    labels = workspace / 'datasets/clear_ec_official/final_train_ids.csv'
    if sha(labels) != record['original_labels_sha256']:
        raise ValueError('Original labels changed')
    validate_inputs(workspace/'datasets/clear_ec_official/train_mha', labels)
    marker = workspace / 'preparation_started.json'
    with marker.open('x') as f:
        json.dump(dict(archive=str(archive), sha256=expected.lower()), f)
    # Frozen prep scripts own only new caches below this workspace.
    commands = [
        ['code/cache_native.py'],
        ['code/prepare_fullframe_cache.py', '--archive', str(archive), '--sha256', expected.lower()],
        ['code/audit_raw_id_aliases.py'],
        ['code/prepare_final9000_candidate.py'],
        ['experiments/region_search_20260905/source/code/fullframe_views.py'],
    ]
    for index, command in enumerate(commands):
        with (workspace / f'prepare_{index}.log').open('x') as log:
            subprocess.run([sys.executable, *command], cwd=workspace,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    with (workspace / 'data_ready.json').open('x') as f:
        json.dump(dict(status='passed', raw_archive_sha256=expected.lower(),
                       training_started=False), f)
    print('Data preparation passed. Training must be launched explicitly.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--workspace', type=Path, required=True)
    init.add_argument('--raw-dir', type=Path, required=True)
    init.add_argument('--labels', type=Path, required=True)
    init.add_argument('--foundation', type=Path, required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--workspace', type=Path, required=True)
    prep.add_argument('--archive', type=Path, required=True)
    prep.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.command == 'init':
        if args.workspace.is_symlink():
            raise ValueError('Workspace cannot be a symlink')
        initialize(args.workspace.absolute(), args.raw_dir.resolve(strict=True),
                   args.labels.resolve(strict=True), args.foundation.resolve(strict=True))
    else:
        prepare(args.workspace.resolve(strict=True), args.archive.resolve(strict=True), args.sha256)


if __name__ == '__main__':
    main()
