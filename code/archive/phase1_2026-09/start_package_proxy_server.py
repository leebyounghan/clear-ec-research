"""Start a private allow-listed transfer endpoint with an ephemeral application key."""
import os
from pathlib import Path
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "submission_candidate_20260909/provenance/.transfer-key"
if not KEY.exists():
    descriptor = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(secrets.token_hex(32))
subprocess.run(["kubectl", "-n", "p-ajou-mlxp-project", "cp", str(ROOT / "code/serve_package_weights.py"),
                "clear-ec-package-reader:/tmp/serve_package_weights.py"], check=True)
process = subprocess.Popen(["kubectl", "-n", "p-ajou-mlxp-project", "exec", "-i", "clear-ec-package-reader", "--",
    "/workspace/venv/bin/python", "/tmp/serve_package_weights.py", "--api-proxy"], stdin=subprocess.PIPE)
process.stdin.write((KEY.read_text() + "\n").encode())
process.stdin.close()
raise SystemExit(process.wait())
