"""The baked NPA interpreter and source identity survive SkyPilot setup."""

from pathlib import Path
import subprocess

import pytest


DOCKERFILE = (
    Path(__file__).resolve().parents[3] / "npa/docker/workbench/curobo/Dockerfile"
)
PIP_BOOTSTRAP = DOCKERFILE.with_name("pip-bootstrap.lock")


def test_baked_identity_uses_checked_build_input_and_absolute_interpreter():
    text = DOCKERFILE.read_text()
    assert "ARG NPA_SOURCE_SHA" in text
    assert "ARG UBUNTU_SNAPSHOT=20260920T000000Z" in text
    assert "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" in text
    assert "archive.ubuntu.com" not in text
    assert "security.ubuntu.com" not in text
    assert "/etc/apt/sources.list.d/*.sources" in text
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in text
    assert "NPA_BAKED_PYTHON=/opt/npa-venv/bin/python" in text
    assert "PYTHONPATH=" not in text
    assert text.index('RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]') < text.index(
        "RUN pip install"
    )


def test_pip_bootstrap_distribution_is_content_pinned():
    text = DOCKERFILE.read_text()
    assert (
        PIP_BOOTSTRAP.read_text()
        == "# PyPI wheel: https://files.pythonhosted.org/packages/f3/6e/"
        "1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/"
        "pip-26.2.1-py3-none-any.whl\n"
        "pip==26.2.1 \\\n"
        "    --hash=sha256:71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e\n"
    )
    assert "COPY docker/workbench/curobo/pip-bootstrap.lock" in text
    assert "--require-hashes -r /opt/pip-bootstrap.lock" in text
    assert "pip install --no-cache-dir --upgrade 'pip==" not in text


@pytest.mark.parametrize("sha", ["", "a" * 39, "a" * 41, "g" * 40, "a" * 40])
def test_docker_source_gate_executes_and_rejects_nonfull_sha(sha):
    instructions = DOCKERFILE.read_text().replace("\\\n", " ").splitlines()
    instruction = next(line for line in instructions if line.startswith("RUN [[ "))
    result = subprocess.run(
        ["/bin/bash", "-c", instruction.removeprefix("RUN ")],
        env={
            "NPA_SOURCE_SHA": sha,
            "SOURCE_DATE_EPOCH": "1700000000",
            "UBUNTU_SNAPSHOT": "20260920T000000Z",
        },
        check=False,
        capture_output=True,
    )
    assert (result.returncode == 0) is (sha == "a" * 40)


@pytest.mark.parametrize(
    "snapshot", ["", "20260919T000000Z", "20260920T000000Z", "latest"]
)
def test_docker_source_gate_rejects_unreviewed_snapshot(snapshot):
    instructions = DOCKERFILE.read_text().replace("\\\n", " ").splitlines()
    instruction = next(line for line in instructions if line.startswith("RUN [[ "))
    result = subprocess.run(
        ["/bin/bash", "-c", instruction.removeprefix("RUN ")],
        env={
            "NPA_SOURCE_SHA": "a" * 40,
            "SOURCE_DATE_EPOCH": "1700000000",
            "UBUNTU_SNAPSHOT": snapshot,
        },
        check=False,
        capture_output=True,
    )
    assert (result.returncode == 0) is (snapshot == "20260920T000000Z")
