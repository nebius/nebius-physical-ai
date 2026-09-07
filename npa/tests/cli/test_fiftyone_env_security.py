"""Execute the generated service setup against real temporary files."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from npa.cli.fiftyone import _service_setup_script


@pytest.mark.parametrize("fail_secret_write", [False, True])
def test_service_env_is_private_before_credentials_and_published_atomically(
    tmp_path: Path, fail_secret_write: bool
) -> None:
    tools = tmp_path / "bin"
    tools.mkdir()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (tmp_path / "systemd").mkdir()
    source = upstream / ".env"
    source.write_text("AWS_SECRET_ACCESS_KEY=synthetic-secret\n")
    source.chmod(0o600)
    destination = tmp_path / "service" / "env"
    prior = b"FIFTYONE_DATASET_NAME=previous\n"
    if fail_secret_write:
        destination.parent.mkdir()
        destination.write_bytes(prior)
        destination.chmod(0o600)
    observations = tmp_path / "observations.json"
    scripts = {
        "sudo": '#!/bin/sh\nexec "$@"\n',
        "systemctl": "#!/bin/sh\nexit 0\n",
        "curl": "#!/bin/sh\nexit 0\n",
    }
    for name, body in scripts.items():
        target = tools / name
        target.write_text(body)
        target.chmod(0o755)
    tee = tools / "tee"
    tee.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, stat, subprocess, sys\n"
        "data = sys.stdin.buffer.read()\n"
        "result = subprocess.run(['/usr/bin/tee', *sys.argv[1:]], input=data, stdout=subprocess.PIPE)\n"
        "sys.stdout.buffer.write(result.stdout)\n"
        "if b'AWS_SECRET_ACCESS_KEY=' in data:\n"
        "    path = pathlib.Path(sys.argv[-1])\n"
        "    event = {'file_mode': stat.S_IMODE(path.stat().st_mode), "
        "'parent_mode': stat.S_IMODE(path.parent.stat().st_mode)}\n"
        f"    pathlib.Path({str(observations)!r}).write_text(json.dumps(event))\n"
        f"    if {fail_secret_write!r}: raise SystemExit(17)\n"
        "raise SystemExit(result.returncode)\n"
    )
    tee.chmod(0o755)
    script = (
        _service_setup_script(5151)
        .replace("/etc/npa-fiftyone", str(destination.parent))
        .replace("/etc/systemd/system", str(tmp_path / "systemd"))
        .replace("/opt/lerobot", str(upstream))
    )
    completed = subprocess.run(
        ["/bin/bash", "-c", "set -euo pipefail\numask 022\n" + script],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert json.loads(observations.read_text()) == {
        "file_mode": 0o600,
        "parent_mode": 0o700,
    }
    assert not list(destination.parent.glob(".env.*"))
    if fail_secret_write:
        assert completed.returncode == 17
        assert destination.read_bytes() == prior
    else:
        assert completed.returncode == 0, completed.stderr
        assert "AWS_SECRET_ACCESS_KEY=synthetic-secret" in destination.read_text()
        assert destination.stat().st_mode & 0o777 == 0o600
