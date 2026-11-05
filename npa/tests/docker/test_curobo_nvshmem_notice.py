"""Bind the additional NVSHMEM release notice and reject changed upstream bytes."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest


IMAGE = Path(__file__).resolve().parents[3] / "npa/docker/workbench/curobo"
ARCHIVE_URL = (
    "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/"
    "linux-x86_64/libnvshmem-linux-x86_64-3.4.5_cuda13-archive.tar.xz"
)
ARCHIVE_PATH = "/tmp/nvshmem-linux-x86_64-3.4.5_cuda13.tar.xz"
ARCHIVE_MEMBER = "libnvshmem-linux-x86_64-3.4.5_cuda13-archive/LICENSE"
ARCHIVE_SHA256 = "18f12807c23bc5b270581af1f9bafd540369da783e557171558887623d77842c"
NOTICE_PATH = "/usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt"
# NVIDIA's 3.4.5 CUDA13 archive contains the complete release notice at this path.
NOTICE_SHA256 = "1f5b7ada702926bc73327e6eb02dc2d41facc844cc4512ac900451bda06a459e"


def _notice_instruction():
    instructions = re.sub(
        r"[ \t]*\\\n\s*", " ", (IMAGE / "Dockerfile").read_text()
    ).splitlines()
    matches = [
        line for line in instructions if line.startswith("RUN ") and ARCHIVE_URL in line
    ]
    assert len(matches) == 1
    return matches[0].removeprefix("RUN ")


def test_notice_is_additional_and_bound_to_reviewed_release_bytes():
    command = _notice_instruction()
    assert command == (
        f"nvshmem_archive={ARCHIVE_PATH} "
        f"&& curl --fail --location {ARCHIVE_URL} -o \"${{nvshmem_archive}}\" "
        f"&& echo '{ARCHIVE_SHA256}  {ARCHIVE_PATH}' | sha256sum -c - "
        f"&& tar -xJOf \"${{nvshmem_archive}}\" {ARCHIVE_MEMBER} > {NOTICE_PATH} "
        f"&& echo '{NOTICE_SHA256}  {NOTICE_PATH}' | sha256sum -c - "
        f"&& rm -f \"${{nvshmem_archive}}\" "
        f"&& chmod 0444 {NOTICE_PATH}"
    )
    record = (IMAGE / "REDISTRIBUTION.md").read_text()
    assert NOTICE_SHA256 in record
    assert NOTICE_PATH in record
    assert "alongside the wheel's bundled CUDA license" in record


@pytest.mark.parametrize("download_fails", [False, True])
def test_actual_notice_shell_rejects_changed_bytes_or_failed_fetch(
    tmp_path, download_fails
):
    """Mock only HTTP transport; use the real shell and SHA256 verifier."""
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        + (
            "exit 22\n"
            if download_fails
            else "for arg; do output=$arg; done\n"
            "printf 'changed or incomplete upstream notice\\n' > \"$output\"\n"
        )
    )
    curl.chmod(0o755)
    destination = tmp_path / "NVSHMEM-LICENSE.txt"
    archive = tmp_path / "nvshmem.tar.xz"
    command = _notice_instruction().replace(NOTICE_PATH, str(destination)).replace(
        ARCHIVE_PATH, str(archive)
    )
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
        assert archive.exists()
        assert not destination.exists()
