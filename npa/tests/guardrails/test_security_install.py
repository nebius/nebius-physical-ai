"""Exercise scanner download recovery and checksum rejection with actual curl."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import threading

import pytest


_INSTALLER = Path(__file__).resolve().parents[3] / "scripts/security_install.sh"


def _scanner_archive():
    content = (
        b'#!/bin/sh\nprintf "fixture scanner\\n"\ntouch "$NPA_TEST_SCANNER_MARKER"\n'
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        entry = tarfile.TarInfo("trivy")
        entry.mode = 0o700
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    return output.getvalue()


@contextmanager
def _release_server(responses, payload):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            code = responses[min(len(requests), len(responses) - 1)]
            requests.append(code)
            body = payload if code == 200 else b"synthetic release-host error"
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/scanner.tar.gz", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _run_installer(tmp_path, url, digest):
    # Change only the external fixture identity; run the real download, checksum,
    # extraction and launch commands, including curl's actual retry behavior.
    script = _INSTALLER.read_text()
    script, urls = re.subn(r"https://github\.com/\S+\.tar\.gz", url, script)
    script, hashes = re.subn(r"[a-f0-9]{64}(?=  trivy\.tar\.gz)", digest, script)
    assert urls == hashes == 1
    path = tmp_path / "installer.sh"
    path.write_text(script)
    marker = tmp_path / "scanner-executed"
    result = subprocess.run(
        ["bash", str(path), str(tmp_path / "tools")],
        env={
            **os.environ,
            "NPA_TEST_SCANNER_MARKER": str(marker),
            "no_proxy": "127.0.0.1",
        },
        capture_output=True,
        text=True,
    )
    return result, marker


@pytest.mark.parametrize("status", [429, 504])
def test_transient_release_failure_recovers_before_verified_execution(tmp_path, status):
    """Retry a transient response and execute only the verified archive.

    Args:
        tmp_path: Isolated installer output.
        status: Transient HTTP response from the local release fixture.
    Returns:
        None.
    Raises:
        AssertionError: Recovery or verified execution fails.
    """
    payload = _scanner_archive()
    with _release_server([status, 200], payload) as (url, requests):
        result, marker = _run_installer(
            tmp_path, url, hashlib.sha256(payload).hexdigest()
        )
    assert result.returncode == 0, result.stderr
    assert requests == [status, 200]
    assert marker.exists()


@pytest.mark.parametrize(
    "responses,checksum_valid", [([404], True), ([504], True), ([504, 200], False)]
)
def test_failed_release_or_checksum_never_extracts_or_executes(
    tmp_path, responses, checksum_valid
):
    """Fail closed on permanent/exhausted HTTP errors or corrupt downloads.

    Args:
        tmp_path: Isolated installer output.
        responses: HTTP results emitted by the local release fixture.
        checksum_valid: Whether the expected hash identifies the fixture bytes.
    Returns:
        None.
    Raises:
        AssertionError: Unverified bytes are extracted or executed.
    """
    payload = _scanner_archive()
    digest = hashlib.sha256(payload).hexdigest() if checksum_valid else "0" * 64
    with _release_server(responses, payload) as (url, requests):
        result, marker = _run_installer(tmp_path, url, digest)
    assert result.returncode != 0
    assert not marker.exists()
    assert not (tmp_path / "tools/trivy").exists()
    assert len(requests) == (6 if responses == [504] else len(responses))
