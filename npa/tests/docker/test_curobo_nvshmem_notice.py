"""Bind the additional NVSHMEM release notice and reject changed upstream bytes."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest


IMAGE = Path(__file__).resolve().parents[3] / "npa/docker/workbench/curobo"
NOTICE_URL = "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt"
NOTICE_PATH = "/usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt"
# NVIDIA's 3.3.24 CUDA13 archive and its linked product license have these bytes.
NOTICE_SHA256 = "43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f"


def _notice_instruction():
    instructions = re.sub(r"[ \t]*\\\n\s*", " ", (IMAGE / "Dockerfile").read_text()).splitlines()
    matches = [line for line in instructions if line.startswith("RUN ") and NOTICE_URL in line]
    assert len(matches) == 1
    return matches[0].removeprefix("RUN ")


def test_notice_is_additional_and_bound_to_reviewed_release_bytes():
    command = _notice_instruction()
    assert command == (
        f"curl --fail --location {NOTICE_URL} -o {NOTICE_PATH} "
        f"&& echo '{NOTICE_SHA256}  {NOTICE_PATH}' | sha256sum -c - "
        f"&& chmod 0444 {NOTICE_PATH}"
    )
    record = (IMAGE / "REDISTRIBUTION.md").read_text()
    assert NOTICE_SHA256 in record
    assert NOTICE_PATH in record
    assert "alongside the wheel's bundled CUDA license" in record


@pytest.mark.parametrize("download_fails", [False, True])
def test_actual_notice_shell_rejects_changed_bytes_or_failed_fetch(tmp_path, download_fails):
    """Mock only HTTP transport; use the real shell and SHA256 verifier."""
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        + ("exit 22\n" if download_fails else "for arg; do output=$arg; done\n"
           "printf 'changed or incomplete upstream notice\\n' > \"$output\"\n")
    )
    curl.chmod(0o755)
    destination = tmp_path / "NVSHMEM-LICENSE.txt"
    command = _notice_instruction().replace(NOTICE_PATH, str(destination))
    result = subprocess.run(
        ["/bin/bash", "-o", "pipefail", "-c", command],
        env={**os.environ, "PATH": f"{tmp_path}:{os.defpath}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    if download_fails:
        assert result.returncode == 22
        assert not destination.exists()
    else:
        assert "FAILED" in result.stdout
        assert destination.stat().st_mode & 0o200  # Final read-only chmod was not reached.
