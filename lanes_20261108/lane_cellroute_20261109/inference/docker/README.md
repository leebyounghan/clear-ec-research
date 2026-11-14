# Container draft for the new-recipe inference (cell-route lane, 2026-11-10). Not built, not pushed.

Files:
- `Dockerfile`: base `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` (the pod runs torch 2.11.0+cu130), packages pinned in `requirements.txt` to the pod venv.
  - Entry: `recipe_infer.py --manifest deploy` on the GC interface: one .mha/.tif in, the three json files out.
- `collect_files.py`: runs one deploy prediction with file-open hooks and writes `files.txt`, every repo file the deploy path reads (code, params, weights) at its repo-relative path.
  - 11-10 census: 552 files, 31.1 GB.
  - Not built yet: ViT-L s46/s47 full data (10 files) and CPSAM 512 full data (5 files).
  - The hex6 corrector's files are added from the GPU deploy run (it is GPU-only).

Build steps (when asked):
1. `python collect_files.py` on the pod after every full-data weight exists.
2. Copy the listed files into `context/workspace/clear-ec-research/` keeping their paths (`rsync --files-from=files.txt`). The repo subset keeps the absolute root `/workspace/clear-ec-research` because several lane modules hard-code it.
3. `docker build -t clear-ec-recipe:<YYYYMMDD>-<sha7 of files.txt> .` (never `latest`), on the build box, after the coordinator's OK.
4. Smoke inside the container on 2-3 training frames written as .mha, compared with the pod entry's output for the same frames.
5. Check once on a T4-class GPU at packaging. Members pick bf16 where supported, fp16 otherwise (`members.ADT`).
