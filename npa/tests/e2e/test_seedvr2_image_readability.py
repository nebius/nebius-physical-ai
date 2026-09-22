"""Exercise SeedVR's actual COPY modes with a local image and UID 1000."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/seedvr2/Dockerfile"
NOTICE = "NVSHMEM-License-v3.4.5-0.txt"


def _local_base() -> tuple[str, tuple[str, ...]]:
    image = os.environ.get("NPA_E2E_SEEDVR_READABILITY_BASE_IMAGE", "")
    if not image:
        pytest.skip("Set NPA_E2E_SEEDVR_READABILITY_BASE_IMAGE to a local image ID")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image), "Use an immutable local ID"
    rows = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))
    assert len(rows) == 1 and rows[0]["Id"] == image
    tags = rows[0].get("RepoTags")
    assert isinstance(tags, list) and tags, "Base image needs a pre-existing tag"
    assert all(isinstance(tag, str) and tag != "<none>:<none>" for tag in tags)
    return image, tuple(tags)


def _assert_base_preserved(image: str, tags: tuple[str, ...]) -> None:
    rows = json.loads(subprocess.check_output(["docker", "image", "inspect", *tags]))
    assert len(rows) == len(tags) and all(row["Id"] == image for row in rows)


def _context(root: Path, mask: int) -> None:
    previous = os.umask(mask)
    try:
        for relative, payload in {
            "notices/nvshmem.txt": "fixture permission notice\n",
            "pyproject.toml": "[project]\nname = 'permission-probe'\n",
            "README.md": "fixture source metadata\n",
            "src/npa/adapter/nested/probe.py": "value = 1\n",
            "src/npa/run-probe": "#!/bin/sh\nprintf 'executable-preserved\\n'\n",
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        (root / "src/npa/run-probe").chmod(0o755)
    finally:
        os.umask(previous)


def _copy_lines() -> list[str]:
    lines = DOCKERFILE.read_text().splitlines()
    destinations = (
        f" /usr/share/doc/npa-seedvr2/{NOTICE}",
        " /opt/npa-src/",
        " /opt/npa-src/src/npa",
    )
    selected = [
        line
        for line in lines
        if line.startswith("COPY ") and line.endswith(destinations)
    ]
    assert len(selected) == 3
    return selected


def _recipe(image: str) -> str:
    return (
        "\n".join(
            [
                f"FROM {image} AS seedvr2-build",
                "USER root",
                f"COPY notices/nvshmem.txt /opt/npa-legal/{NOTICE}",
                f"FROM {image}",
                "USER root",
                "RUN rm -rf /opt/npa-src",
                *_copy_lines(),
                "USER 1000",
                "ENTRYPOINT []",
            ]
        )
        + "\n"
    )


def _verify_readable(tag: str) -> None:
    script = (
        "set -eu; test $(id -u) = 1000; "
        f"test -r /usr/share/doc/npa-seedvr2/{NOTICE}; "
        "test -r /opt/npa-src/pyproject.toml; "
        "test -r /opt/npa-src/README.md; "
        "test -r /opt/npa-src/src/npa/adapter/nested/probe.py; "
        "cat /opt/npa-src/src/npa/adapter/nested/probe.py; "
        "/opt/npa-src/src/npa/run-probe"
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            tag,
            "sh",
            "-c",
            script,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "value = 1" in result.stdout and "executable-preserved" in result.stdout


def _build_fixture(tag: str, context: Path) -> None:
    result = subprocess.run(
        [
            "docker",
            "build",
            "--pull=false",
            "--network=none",
            "--provenance=false",
            "-t",
            tag,
            str(context),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_seedvr_runtime_copies_are_readable_under_source_umask(
    tmp_path: Path, mask: int
) -> None:
    """Use native Docker COPY semantics and a non-root process, without a GPU."""
    image, original_tags = _local_base()
    _context(tmp_path, mask)
    tag = "npa-seedvr-readability-test:" + uuid4().hex
    base_tag = tag + "-base"
    try:
        subprocess.run(["docker", "tag", image, base_tag], check=True)
        actual = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", base_tag], text=True
        ).strip()
        assert actual == image
        (tmp_path / "Dockerfile").write_text(_recipe(base_tag))
        _build_fixture(tag, tmp_path)
        _verify_readable(tag)
    finally:
        _assert_base_preserved(image, original_tags)
        subprocess.run(
            ["docker", "image", "rm", tag, base_tag], capture_output=True, check=False
        )
        _assert_base_preserved(image, original_tags)
