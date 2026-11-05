"""Download the approved foundation on the cluster; credential from stdin only.

Never call login(), persist a token, put credentials in argv/environment, or
print exception messages/URLs (which may contain signed credentials).
"""
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

REPO = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
REVISION = '5931719e67bbdb9737e363e781fb0c67687896bc'
DESTINATION = Path('/workspace/clear-ec-research/experiments/phase2_foundation_20261002/weights/dinov3-vitb16')
FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors', 'LICENSE.md', 'README.md')


def main():
    disable_progress_bars()
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    token = sys.stdin.readline().strip()
    if not token.startswith('hf_'):
        raise ValueError('Invalid credential input')
    api = HfApi(token=token)
    info = api.model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Unexpected model revision')
    available = {entry.rfilename: entry for entry in info.siblings}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name in FILES:
        path = Path(hf_hub_download(REPO, name, revision=REVISION, token=token,
                                   local_dir=DESTINATION))
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        lfs = getattr(available[name], 'lfs', None)
        expected = getattr(lfs, 'sha256', None) if lfs else None
        if expected is not None and actual != expected:
            raise ValueError('Downloaded file checksum mismatch')
        verified[name] = dict(bytes=path.stat().st_size, sha256=actual,
                              upstream_lfs_sha256=expected,
                              upstream_sha_verified=bool(expected and expected == actual))
        print(json.dumps({'verified_file': name, 'bytes': path.stat().st_size}), flush=True)
    del token, api
    result = dict(repo=REPO, revision=REVISION, destination=str(DESTINATION),
                  files=verified, credential_saved=False, downloaded_on='k8s-mlx',
                  verified_utc=time.strftime('%FT%TZ', time.gmtime()))
    manifest = DESTINATION/'artifact_manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['files'] != verified or previous['revision'] != REVISION:
            raise ValueError('Existing artifact manifest differs; refusing overwrite')
    else:
        with manifest.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({'status': 'download_verified', 'manifest': str(manifest),
                      'model_sha256': verified['model.safetensors']['sha256']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        response = getattr(error, 'response', None)
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'http_status': getattr(response, 'status_code', None)}), flush=True)
        sys.exit(1)
