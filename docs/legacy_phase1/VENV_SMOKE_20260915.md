# Fresh source-venv inference verification

On September 15, a new HTTPS checkout of commit
`d80342d6c09a6fab6097a9b928e7104c1bf0681b` was installed into a new isolated venv
on a separate Linux x86_64 RTX 4090 server. No system-site packages, old venv, or
Docker runtime were used for this inference. The Python interpreter was the host's
Python 3.12.7; all application packages were freshly installed from the repository
instructions, with pip caching disabled. This was not a newly installed operating
system, and the host NVIDIA driver remained a prerequisite.

| Check | Result |
|---|---|
| Python / PyTorch / CUDA | 3.12.7 / 2.11.0+cu126 / 12.6 |
| Installation / `pip check` / CUDA FP16 | Passed |
| Frozen inference / training source integrity | Passed, including 16 historical training-source checks |
| September 14 model-resource hashes | All eight passed |
| Direct source inference | Six members, three finite positive JSON scalars |
| Comparison with released Docker | CD, CV and HEX exactly equal, including output file hashes |
| Timings | Application 12.700 seconds; wrapper including checks/startup 17.802 seconds |
| Checkout | Unmodified; no training started |

The same one-image input and previously verified Drive-downloaded model resources
were reused for a controlled Docker/source comparison. No individual predictions
or challenge images are distributed. This verifies the installed source inference
path, not full dataset accuracy, a T4 benchmark, Docker image rebuilding, or refitting.

## Reproduce the tested source path

From the repository root on a Linux NVIDIA host with Python 3.12 available:

```bash
LAMDA_PYTHON=python3.12 bash review_submission_20260910/scripts/setup_venv.sh
source review_submission_20260910/.venv/bin/activate
python review_submission_20260910/scripts/check_environment.py
```

Download/extract the current archives as in the [Docker guide](../../docker_review_20260914/README.md).
For this source path, Docker loading is not required. Verify the latest model
resources explicitly, because the preserved September 10 optional model verifier
targets the old sixth checkpoint:

```bash
(cd docker_review_20260914/checkpoints/model && sha256sum --check ../../MODEL_SHA256SUMS.txt)
python review_submission_20260910/scripts/predict_image.py \
  --image /absolute/path/to/authorized-case.mha \
  --model docker_review_20260914/checkpoints/model \
  --output /absolute/path/to/new-empty-output \
  --gpu 0
```

The wrapper's source verification remains valid because the application is unchanged.
It does not validate the September 14 checkpoint hashes itself; the separate current
manifest command above is required. Expected outputs are the same three scalar JSONs
documented in the Docker guide. Never use the old verifier's `--model` option for the
replacement September 14 sixth checkpoint.
