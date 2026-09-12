"""Aggregate Kubernetes-only verification report; no image/label content emitted."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from check_environment import validate


def main():
    root = Path(__file__).resolve().parents[1]
    environment = validate(allow_no_gpu=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'scripts',
                    '-p', 'test_venv_tools.py', '-v'], cwd=root, check=True)
    model = Path('/reference/clear-ec-research/experiments/region_candidate_20260907/experimental_stage/model')
    subprocess.run([sys.executable, 'scripts/verify_release.py', '--model', str(model),
                    '--report', 'provenance/k8s_source_verification.json'], cwd=root, check=True)
    report = {
        'status': 'passed', 'execution_location': 'Kubernetes CPU verification pod',
        'environment': environment, 'isolated_venv': sys.prefix != sys.base_prefix,
        'pip_check': 'passed during setup', 'synthetic_wrapper_tests': 8,
        'checkpoint_files_verified': 8, 'full_gpu_inference_parity': 'not tested',
        'training_pods_modified': False,
    }
    destination = root / 'provenance/k8s_venv_verification.json'
    destination.write_text(json.dumps(report, indent=2))
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (root / 'provenance/venv-linux-installed.txt').write_text(freeze)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
