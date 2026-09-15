"""Verify generated readiness commands against an actual loopback HTTP server."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import shlex
import subprocess
import threading

import pytest

from npa.cli import fiftyone
from npa.clients.ssh import SSHError


@pytest.mark.parametrize("kind", ["service", "native_import", "container_import"])
@pytest.mark.parametrize("http_status", [200, 503])
def test_readiness_returns_failure_unless_app_responds(kind, http_status, tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(http_status)
            self.end_headers()

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    monkeypatch.setattr(fiftyone, "FIFTYONE_READY_ATTEMPTS", 1)
    if kind == "service":
        script = fiftyone._service_setup_script(port)
        script = script[script.index("for _ in $(seq 1"):]
    else:
        builder = fiftyone._build_load_dataset_command if kind == "native_import" else fiftyone._build_container_load_dataset_command
        script = shlex.split(builder("review", "s3://example-bucket/images"))[-1]
        script = script[script.index('app_port="$(grep'):]
    env_file = tmp_path / "service.env"
    env_file.write_text(f"FIFTYONE_DEFAULT_APP_PORT={port}\n")
    script = script.replace("/etc/npa-fiftyone/env", str(env_file))
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, contents in {
        "sudo": '#!/bin/sh\nexec "$@"\n',
        "systemctl": '#!/bin/sh\nprintf "old journal: NPA_FIFTYONE_APP_READY\\n"\nexit 0\n',
        "docker": '#!/bin/sh\nprintf "old container: NPA_FIFTYONE_APP_READY\\n"\nexit 0\n',
        "sleep": '#!/bin/sh\nexit 0\n',
    }.items():
        path = tools / name
        path.write_text(contents)
        path.chmod(0o755)
    try:
        result = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + script],
            env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"},
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == (0 if http_status == 200 else 1)
        final_line = result.stdout.rstrip().splitlines()[-1]
        if http_status == 200:
            assert final_line == fiftyone.FIFTYONE_READY_MARKER
        else:
            assert "readiness timeout" in result.stderr
            assert final_line != fiftyone.FIFTYONE_READY_MARKER
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("output", [
    "old journal: NPA_FIFTYONE_APP_READY\n",
    "NPA_FIFTYONE_APP_READY\nsubsequent failure\n",
])
def test_historical_or_embedded_marker_cannot_mask_command_failure(output, mocker):
    ssh = mocker.MagicMock()
    ssh.run.return_value = (1, output, "readiness timeout")
    with pytest.raises(SSHError, match="exit 1"):
        fiftyone._run_fiftyone_command(ssh, "launch")


def test_fiftyone_commands_do_not_invoke_login_logout_hooks():
    # Ubuntu's login-shell logout handling can override a successful command.
    # These helpers already establish their own runtime and storage environment.
    command = shlex.split(fiftyone._remote_bash("printf ready; exit 0"))
    assert command[:2] == ["bash", "-c"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout == "ready"
