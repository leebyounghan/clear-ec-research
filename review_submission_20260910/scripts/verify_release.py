"""Read-only release verification; optionally write a new aggregate QA report."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = {
    'inference.py': 'da7b48cb36402d5c95b8baefaf438da9c13d7ed4e82f2838f76e716ae92c7939',
    'base_inference.py': 'ab281c92334c1592b4bca25be9fb9648b4e30fe2b677c84126f7360d42a5fd94',
    'final_candidate_core.py': '92207fff4ad29feeef1cbc37746de730a2f7fbd076d00207fc6795bcece268f4',
    'fullframe_views.py': '9e689bc02d3c3e35ffa3cc651ec128f941360d98781c04ef3fd3eed712b7594f',
    'native_region_views.py': '28faab9a057e6b16fd1956efbed6ca57b8acd3f72a9e9630c61344bc6a80dbb2',
    'region_inference_core.py': '19d2d104aa3f025370c3d2ef4ebbdbf9aa387a2a24b99cc16fe76d877dbaa2bb',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected, size=None):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing or symlinked release file: {path}')
    if size is not None and path.stat().st_size != size:
        raise ValueError(f'Byte size mismatch: {path}')
    if digest(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path)
    parser.add_argument('--archives', type=Path)
    parser.add_argument('--model-archive', type=Path, help='Verify model-only download; Docker archive not required')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    for name, expected in APP.items():
        check(ROOT / 'app' / name, expected)
    model_manifest = json.loads((ROOT / 'provenance/model_archive.json').read_text())
    config_name = 'candidate6_full_quality.json'
    config_entry = model_manifest['members'][config_name]
    check(ROOT / 'configs' / config_name, config_entry['sha256'], config_entry['bytes'])
    historical_checks = 0
    # Explicit membership also excludes macOS AppleDouble ._*.json sidecars.
    for name in ('seed42', 'seed77', 'seed2026', 'metric35', 'predbag2_35', 'full_quality_seed77'):
        config_path = ROOT / 'provenance/training' / f'{name}.json'
        config = json.loads(config_path.read_text())
        regional = config_path.stem == 'full_quality_seed77'
        base = ROOT / 'training'
        if regional:
            base /= 'experiments/region_search_20260905/source/code'
        for relative, expected in config['source_sha256'].items():
            check(base / relative, expected)
            historical_checks += 1
        if regional:
            check(ROOT / 'training/experiments/deploy_region_20260907/source/train_deploy_region.py',
                  config['adapter_sha256'])
            historical_checks += 1
    sources = [p for folder in ('app', 'training', 'scripts')
               for p in (ROOT / folder).rglob('*.py')]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    if args.model:
        for name, entry in model_manifest['members'].items():
            check(args.model / name, entry['sha256'], entry['bytes'])
    if args.archives:
        for filename in ('model_archive.json', 'image_archive.json'):
            entry = json.loads((ROOT / 'provenance' / filename).read_text())
            check(args.archives / entry['name'], entry['sha256'], entry['bytes'])
    if args.model_archive:
        check(args.model_archive, model_manifest['sha256'], model_manifest['bytes'])
    report = {
        'status': 'passed', 'frozen_inference_modules': len(APP),
        'historical_training_source_checks': historical_checks,
        'python_syntax_files': len(sources), 'config_hash_verified': True,
        'model_files_verified': len(model_manifest['members']) if args.model else 0,
        'archives_verified': 2 if args.archives else 0,
        'model_archive_verified': bool(args.model_archive or args.archives),
        'clean_gpu_inference_tested': False, 'clean_retraining_tested': False,
        'reviewer_access_verified': False,
        'scope': 'Byte identities and syntax only; not a numerical reproduction certificate.'
    }
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
